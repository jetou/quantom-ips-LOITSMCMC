import logging
import os,sys
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Optional
from omegaconf import MISSING
import numpy as np


import time

from quantom_ips import make
from quantom_ips.utils.torch_nn_registry import get_optimizer
from quantom_ips.utils.stateful_module import StatefulModule

logger = logging.getLogger(__name__)


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GANOptimizer(StatefulModule):
    def __init__(
        self,
        discriminator: Any = MISSING,  # Can be set when registering or on command-line
        generator: Any = MISSING,
        n_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        gen_opt: str = "Adam",
        gen_lr: float = 1e-5,
        disc_opt: str = "Adam",
        disc_lr: float = 9e-6,
        #disc_lr: float = 1e-6,
        batch_size: int = 10,
        logdir: str = "${hydra:runtime.output_dir}",
        progress_bar: bool = False,
        label_noise: float = 0.1,
        noise_dim: int = 10,
        train_on: str = "events",  # "events" or "xsec"
        sample_diagnostics: bool = True,  # keep epoch0/epochN event dumps
        mse_freq: int = 1,
        diag_n_events: int = int(1e5),    
        diag_n_repeats: int = 10,     # number of diagnostic reps
        diag_seed: int = 0,           # base seed for deterministic reps
        ) -> None:
        
        super().__init__()

        assert train_on in ["events", "xsec"]
        self.train_on = train_on
        self.sample_diagnostics = sample_diagnostics
        self.diag_n_events = diag_n_events
        self.diag_n_repeats = diag_n_repeats
        self.diag_seed = diag_seed

        
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.gen_opt = gen_opt
        self.gen_lr = gen_lr
        self.disc_opt = disc_opt
        self.disc_lr = disc_lr
        self.batch_size = batch_size
        self.logdir = logdir
        self.progress_bar = progress_bar
        self.label_noise = label_noise
        self.noise_dim = noise_dim
        self.mse_freq = mse_freq

        self.discriminator = make(discriminator)
        self.generator = make(generator)


    


    def _dump_fake_event_reps(self, env, *, filename_prefix: str, dtype, device):
        """
        Full-pipeline diagnostics:
          For r in [0..diag_n_repeats-1]:
            noise_r -> generator -> theory.forward -> sampler.forward -> events_r
        Saves:
          <logdir>/<prefix>_fake_events_reps.npy with shape (R, N, 5)
          <logdir>/<prefix>_fake_events.npy      with shape (N, 5)  (rep 0)
        """
        assert env.sampler is not None, "Need a sampler for event diagnostics."
        assert self.diag_n_events is not None and self.diag_n_events > 0
    
        generator = self.generator
        generator.eval()
    
        R = int(self.diag_n_repeats)
        N = int(self.diag_n_events)
        B = int(self.batch_size)
    
        fake_reps = []
    
        with torch.no_grad():
            for r in range(R):
                seed_all(int(self.diag_seed) + r)
    
                # 1) sample noise (new rep)
                noise = torch.normal(
                    0.0, 1.0, size=(B, self.noise_dim), device=device
                )
    
                # 2) generator -> params
                params = generator(noise)
    
                # 3) theory forward -> probabilities / grid_axes
                # We call env.step so it does any flavor broadcasting you implemented
                # BUT we need to tell env how many events to generate.
                #
                # We'll pass a DUMMY real_samples tensor only to provide N, unless your env supports n_samples.
                dummy_real = torch.empty((1, N, 5), device=device, dtype=dtype)
    
                obs, reward, terminated, truncated, info = env(params, dummy_real)
                # Use raw (physical) events for eval; obs["fake"] is batch-normalized
                fake_ev = info.get("fake_events_raw", obs["fake"])  # expected (B, N, 5)
                if fake_ev.ndim != 3 or fake_ev.shape[-1] != 5:
                    raise ValueError(f"Expected fake events (B,N,5), got {tuple(fake_ev.shape)}")
    
                # flatten batch clouds together -> (B*N, 5)
                fake_flat = fake_ev.reshape(-1, fake_ev.shape[-1]).detach().cpu().numpy()
    
                # Keep exactly N events per rep (not B*N) if you want fixed N:
                # Option A: keep all B*N (more statistics per rep)
                # Option B: keep exactly N by slicing (recommended for consistency with truth sampling)
                fake_flat = fake_flat[:N]
    
                fake_reps.append(fake_flat)
    
        fake_reps = np.stack(fake_reps, axis=0)  # (R, N, 5)
    
        # Save
        reps_path = os.path.join(self.logdir, f"{filename_prefix}_fake_events_reps.npy")
        single_path = os.path.join(self.logdir, f"{filename_prefix}_fake_events.npy")
        np.save(reps_path, fake_reps)
        np.save(single_path, fake_reps[0])
    
        logger.info(f"[diagnostics] saved fake reps: {reps_path} shape={fake_reps.shape}")


    
    def train(self, env, data_parser):
        os.makedirs(Path(self.logdir), exist_ok=True)
    
        generator = self.generator
        discriminator = self.discriminator
        dtype = next(generator.parameters()).dtype
        device = next(generator.parameters()).device
    
        free_tmd_mode = getattr(env.theory, "free_tmd_mode", False)
        print("Free TMD mode?", free_tmd_mode)
    
        gen_optimizer = get_optimizer(self.gen_opt, generator.parameters(), self.gen_lr)
        disc_optimizer = get_optimizer(self.disc_opt, discriminator.parameters(), self.disc_lr)
        criterion = nn.BCELoss()
        #criterion = nn.MSELoss()
    
        n_epochs = self.n_epochs
        g_losses = torch.zeros(n_epochs)
        d_losses = torch.zeros(n_epochs)
        
        tmd_mse   = torch.full((n_epochs,), float("nan"))
        xsec_mse  = torch.full((n_epochs,), float("nan"))
        input_mse = torch.full((n_epochs,), float("nan"))

        tmd_mse_std   = torch.full((n_epochs,), float("nan"))
        xsec_mse_std  = torch.full((n_epochs,), float("nan"))
        input_mse_std = torch.full((n_epochs,), float("nan"))
        
                

    
        # --------------------------
        # Helper: make params compatible with SIDIS theory flavor expectations
        # --------------------------
        def ensure_flavor_dim(x, n_flav: int = 6):
            """
            Ensure x has shape (B, n_flav, ..., ...) for SIDIS theory.
            Accepts (B,1,...) and broadcasts to (B,n_flav,...).
            """
            if x.ndim != 4:
                raise ValueError(f"Expected 4D tensor (B,f,x,b), got shape {tuple(x.shape)}")
        
            B, f, nx, nb = x.shape
        
            if f == n_flav:
                return x
            if f == 1:
                return x.repeat(1, n_flav, 1, 1)
        
            raise ValueError(f"Expected {n_flav} channels (flavors) or 1, got {f}")


    
        # --------------------------
        # Load "true" TMD from density.npy
        # --------------------------
        use_log = False
        eps = 1e-12
    
        sample_dir = env.sample_dir
        density_path = os.path.join(sample_dir, "density.npy")
        D = np.load(density_path, allow_pickle=True).item()
        density_raw = D["density"][0][0]  # (nx, nbt) TMD(x,b), no bT factor
    
        real_tmd = torch.tensor(density_raw, device=device, dtype=dtype)
        if real_tmd.shape[0] == env.theory.nbt:
            real_tmd = real_tmd.T
        real_tmd = real_tmd.unsqueeze(0).unsqueeze(0)  # (1,1,x,b)
    
        real_tmd_input = real_tmd.squeeze(0).squeeze(0).detach().cpu().clone()  # (x,b)
    
        # Grid axes for get_tmd_pdf (still uses x,b based shape)
        grid_axes = env.theory.create_grid(real_tmd.shape[2:], real_tmd.dtype, real_tmd.device)
    
        # For SIDIS (non-free), make a "6-flavor truth" for any call that needs f=6
        real_tmd_for_sidistheory = ensure_flavor_dim(real_tmd)  # -> (1,6,x,b) if needed
    
        
        
        # --------------------------
        # Compute target evolved TMD for physics MSE (x,Q,b)
        # --------------------------
        with torch.no_grad():
            # NOTE: get_tmd_pdf expects flavor dimension in SIDIS mode
            tmd_real = env.theory.get_tmd_pdf(real_tmd_for_sidistheory, grid_axes)  # (B,f,x,Q,b)
            real_tmd_for_mse = tmd_real.mean(dim=(0, 1))  # (x,Q,b)
    
            if use_log:
                real_tmd_for_mse = torch.log10(torch.clamp(real_tmd_for_mse, min=eps))
    
        real_tmd_evolved = real_tmd_for_mse.detach().cpu().clone()  # (x,Q,b)
    
        # --------------------------
        # Compute "true" cross section tensor as xsec target
        # --------------------------
        with torch.no_grad():
            xsec_true, xsec_grid, _ = env.theory.forward(real_tmd_for_sidistheory)  # (1,x,Q,z,q,phi) if average=True
            if xsec_true.ndim == 5:
                xsec_true = xsec_true.unsqueeze(0)
        xsec_true_cpu = xsec_true.detach().cpu().clone()
    
        # --------------------------
        # Storage for first/last epoch snapshots & ensembles 
        # --------------------------
        first_gen_tmd_input = None
        first_gen_tmd       = None
        last_gen_tmd_input  = None
        last_gen_tmd        = None
    
        first_gen_tmd_input_ens = None
        first_gen_tmd_ens       = None
        last_gen_tmd_input_ens  = None
        last_gen_tmd_ens        = None
    
        # --------------------------
        # Snapshot: UNTRAINED generator (epoch 0)
        # --------------------------
        with torch.no_grad():
            n_eval = self.batch_size
            noise_eval = torch.normal(0.0, 1.0, size=(n_eval, self.noise_dim), device=device)
            gen_out_0 = generator(noise_eval)
    
            if not free_tmd_mode:
                gen_tmd_0 = gen_out_0
                if gen_tmd_0.ndim == 3:
                    gen_tmd_0 = gen_tmd_0.unsqueeze(1)  # (B,1,x,b)
                # for evolved TMD we need f=6
                gen_tmd_0_for_theory = ensure_flavor_dim(gen_tmd_0)
    
                first_gen_tmd_input_ens = gen_tmd_0.squeeze(1).detach().cpu().clone()  # (B,x,b)
    
                tmd_gen_0 = env.theory.get_tmd_pdf(gen_tmd_0_for_theory, grid_axes)     # (B,f,x,Q,b)
                first_gen_tmd_ens = tmd_gen_0.mean(dim=1).detach().cpu().clone()        # (B,x,Q,b) mean over flavor
    
                first_gen_tmd_input = first_gen_tmd_input_ens.mean(dim=0).clone()      # (x,b)
                first_gen_tmd       = first_gen_tmd_ens.mean(dim=0).clone()            # (x,Q,b)
    
            else:
                gen_tmd_0 = gen_out_0
                if gen_tmd_0.ndim == 4:
                    gen_tmd_0 = gen_tmd_0.unsqueeze(1)  # (B,1,x,Q,b)
    
                first_gen_tmd_ens = gen_tmd_0.detach().cpu().clone()                   # (B,f,x,Q,b) with f=1 maybe
                iQ2_mid = gen_tmd_0.shape[3] // 2
    
                first_gen_tmd_input = gen_tmd_0.mean(dim=(0, 1))[:, iQ2_mid, :]        # (x,b)
                first_gen_tmd       = gen_tmd_0.mean(dim=(0, 1))                       # (x,Q,b)
    
                first_gen_tmd_input_ens = gen_tmd_0[:, :, :, iQ2_mid, :].mean(dim=1).detach().cpu().clone()  # (B,x,b)
    
        # --------------------------
        # Optional epoch0 diagnostics (EVENT MODE ONLY)
        # --------------------------
        if self.train_on == "events" and self.sample_diagnostics:
            self._dump_fake_event_reps(
                env,
                filename_prefix="epoch0",
                dtype=dtype,
                device=device,
            )

    
        # --------------------------
        # Main training loop
        # --------------------------
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
    
            if self.train_on == "events":
                # decide how many steps this epoch
                steps_this_epoch = self.steps_per_epoch if self.steps_per_epoch is not None else len(data_parser)
                
                data_iter = iter(data_parser)
                iter_steps = range(steps_this_epoch)
                if self.progress_bar:
                    iter_steps = tqdm(iter_steps)
                
                for _ in iter_steps:
                    try:
                        real_events = next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_parser)
                        real_events = next(data_iter)
                    
                    real_events = real_events.to(dtype=dtype, device=device)
    
                    generator.zero_grad()        

                    noise = torch.normal(0.0, 1.0, size=(self.batch_size, self.noise_dim), device=device)
                    params = generator(noise)
                    
    
                    obs, reward, terminated, truncated, info = env(params, real_events)

    
                    fake_ev = obs["fake"]  # (B,N,5)
                    real_ev = obs["real"]  # (B,N,5)
               
                    B, N, D = fake_ev.shape
                    fake_in = fake_ev.reshape(B * N, D)
                    real_in = real_ev.reshape(B * N, D)


                    # if epoch == 0:
                    #     print("=== DISCRIMINATOR INPUT DEBUG ===")
                    #     print("fake_in.shape:", fake_in.shape)
                    #     print("real_in.shape:", real_in.shape)
                    #     print("fake_in mean/std:",
                    #           fake_in.mean().item(),
                    #           fake_in.std().item())
                    #     print("real_in mean/std:",
                    #           real_in.mean().item(),
                    #           real_in.std().item())
                    #     print("===============================")

                    #     print(
                    #             f"[Epoch {epoch}] "
                    #             f"B_gen={params.shape[0]}, "
                    #             f"B_real={real_events.shape[0]}, "
                    #             f"N_events={real_events.shape[-2]}"
                    #         )


    
                    # ---- G update
                    fake_labels = discriminator(fake_in)
                    labels = torch.full((B * N, 1), 1.0, device=device)
                    labels -= self.label_noise * torch.rand_like(labels)
    
                    gen_loss = criterion(fake_labels, labels)
                    gen_loss.backward()

                    #! Does grad exist? Yes.
                    # total = 0.0
                    # for p in generator.parameters():
                    #     if p.grad is not None:
                    #         total += p.grad.detach().float().norm().item()
                    # print("GEN grad-norm:", total)
                    
                    gen_optimizer.step()
    
                    # ---- D update
                    discriminator.zero_grad()
    
                    fake_labels = discriminator(fake_in.detach())
                    labels = torch.full((B * N, 1), 0.0, device=device)
                    labels += self.label_noise * torch.rand_like(labels)
                    disc_loss_fake = criterion(fake_labels, labels)
    
                    real_labels = discriminator(real_in)
                    labels = torch.full((B * N, 1), 1.0, device=device)
                    labels -= self.label_noise * torch.rand_like(labels)
                    disc_loss_real = criterion(real_labels, labels)
    
                    disc_loss = disc_loss_fake + disc_loss_real
                    disc_loss.backward()
                    disc_optimizer.step()
    
            else:
                # xsec mode: ONE step per epoch
                generator.zero_grad()
    
                noise = torch.normal(0.0, 1.0, size=(self.batch_size, self.noise_dim), device=device)
                params = generator(noise)
    
                # params must be theory-compatible (f=6) in SIDIS mode
                params_for_theory = ensure_flavor_dim(params)
    
                fake_xsec, _, _ = env.theory.forward(params_for_theory)  # (B,x,Q,z,q,phi) or (1,...) if average=True
    
                # If theory averages over batch, expand it back to B for GAN training
                if fake_xsec.shape[0] == 1 and self.batch_size > 1:
                    fake_xsec = fake_xsec.repeat(self.batch_size, *([1] * (fake_xsec.ndim - 1)))
    
                real_xsec = xsec_true.to(device=device, dtype=dtype)
                if real_xsec.shape[0] == 1 and fake_xsec.shape[0] > 1:
                    real_xsec = real_xsec.repeat(fake_xsec.shape[0], *([1] * (real_xsec.ndim - 1)))
    
                def norm_xsec(x, epsn=1e-8):
                    dims = tuple(range(1, x.ndim))
                    mu = x.mean(dim=dims, keepdim=True)
                    sig = x.std(dim=dims, keepdim=True).clamp_min(epsn)
                    return (x - mu) / sig
    
                fake_xsec_n = norm_xsec(fake_xsec)
                real_xsec_n = norm_xsec(real_xsec)
    
                B = fake_xsec_n.shape[0]
                fake_in = fake_xsec_n.reshape(B, -1)
                real_in = real_xsec_n.reshape(B, -1)
    
                # ---- G update
                fake_labels = discriminator(fake_in)
                labels = torch.full((B, 1), 1.0, device=device)
                labels -= self.label_noise * torch.rand_like(labels)
    
                gen_loss = criterion(fake_labels, labels)
                gen_loss.backward()

              

                gen_optimizer.step()

                
    
                # ---- D update
                discriminator.zero_grad()
    
                fake_labels = discriminator(fake_in.detach())
                labels = torch.full((B, 1), 0.0, device=device)
                labels += self.label_noise * torch.rand_like(labels)
                disc_loss_fake = criterion(fake_labels, labels)
    
                real_labels = discriminator(real_in)
                labels = torch.full((B, 1), 1.0, device=device)
                labels -= self.label_noise * torch.rand_like(labels)
                disc_loss_real = criterion(real_labels, labels)
    
                disc_loss = disc_loss_fake + disc_loss_real
                disc_loss.backward()
                disc_optimizer.step()
    
            # record epoch losses
            g_losses[epoch] = gen_loss.detach().cpu().item()
            d_losses[epoch] = disc_loss.detach().cpu().item()

            do_mse = ((epoch % self.mse_freq) == 0) or (epoch == n_epochs - 1)
            # --------------------------
            # End-of-epoch physics MSE + final snapshots (your existing logic)
            # --------------------------
            if do_mse:
                with torch.no_grad():
                    n_eval = self.batch_size
                    noise_eval = torch.normal(0.0, 1.0, size=(n_eval, self.noise_dim), device=device)
                    gen_out = generator(noise_eval)
        
                    if not free_tmd_mode:
                        gen_tmd = gen_out
                        if gen_tmd.ndim == 3:
                            gen_tmd = gen_tmd.unsqueeze(1)  # (B,1,x,b)
        
                        # evolve requires flavor dim
                        gen_tmd_for_theory = ensure_flavor_dim(gen_tmd)
        
                        gen_input_ens = gen_tmd.squeeze(1)  # (B,x,b)
        
                        tmd_gen = env.theory.get_tmd_pdf(gen_tmd_for_theory, grid_axes)  # (B,f,x,Q,b)
                        tmd_gen_ens = tmd_gen.mean(dim=1)  # (B,x,Q,b) mean over flavor
        
                        gen_input_mean = gen_input_ens.mean(dim=0)  # (x,b)
                        tmd_gen_mean   = tmd_gen_ens.mean(dim=0)    # (x,Q,b)
    
                       
        
                    else:
                        gen_tmd = gen_out
                        if gen_tmd.ndim == 4:
                            gen_tmd = gen_tmd.unsqueeze(1)  # (B,1,x,Q,b)
        
                        tmd_gen_ens = gen_tmd  # (B,f,x,Q,b)
        
                        iQ2_mid = gen_tmd.shape[3] // 2
                        gen_input_ens = gen_tmd[:, :, :, iQ2_mid, :].mean(dim=1)  # (B,x,b)
        
                        gen_input_mean = gen_input_ens.mean(dim=0)    # (x,b)
                        tmd_gen_mean   = tmd_gen_ens.mean(dim=(0, 1)) # (x,Q,b)
        
                    tmd_gen_mean_for_mse = tmd_gen_mean
                    if use_log:
                        tmd_gen_mean_for_mse = torch.log10(torch.clamp(tmd_gen_mean_for_mse, min=eps))
        
                    
                    # TMD MSE across eval batch: per-sample MSE -> mean/std
                    # ----------------------------------------------------
                    # tmd_gen_ens is (B,x,Q,b) in non-free mode, or (B,f,x,Q,b) in free mode
                    if not free_tmd_mode:
                        tmd_for_mse_ens = tmd_gen_ens  # (B,x,Q,b)
                    else:
                        tmd_for_mse_ens = tmd_gen_ens.mean(dim=1)  # (B,x,Q,b) mean over flavor if present
                    
                    if use_log:
                        tmd_for_mse_ens = torch.log10(torch.clamp(tmd_for_mse_ens, min=eps))
                    
                    # broadcast truth (x,Q,b) -> (B,x,Q,b)
                    diff2 = (tmd_for_mse_ens - real_tmd_for_mse.unsqueeze(0)) ** 2
                    mse_per = diff2.mean(dim=(1, 2, 3))  # (B,)
                    
                    tmd_mse[epoch]     = mse_per.mean().detach().cpu()
                    tmd_mse_std[epoch] = mse_per.std(unbiased=False).detach().cpu()

    
                    # ----------------------------------------------------
                    # Input-scale MSE across eval batch: per-sample -> mean/std
                    # gen_input_ens is (B,x,b)
                    # ----------------------------------------------------
                    real_in = real_tmd_input.detach().cpu().float()  # (x,b)
                    gen_in_ens = gen_input_ens.detach().cpu().float()  # (B,x,b)
                    
                    if use_log:
                        real_in = torch.log10(torch.clamp(real_in, min=eps))
                        gen_in_ens = torch.log10(torch.clamp(gen_in_ens, min=eps))
                    
                    diff2_in = (gen_in_ens - real_in.unsqueeze(0)) ** 2
                    mse_in_per = diff2_in.mean(dim=(1, 2))  # (B,)
                    
                    input_mse[epoch]     = mse_in_per.mean()
                    input_mse_std[epoch] = mse_in_per.std(unbiased=False)
                    
                    print(
                        f"  [Epoch {epoch+1}] INPUT MSE (all x, b_T): "
                        f"{input_mse[epoch].item():.3e} +/- {input_mse_std[epoch].item():.3e}"
                    )

    
                    # ----------------------------------------------------
                    # Cross-section MSE at this epoch (compare to xsec_true)
                    # ----------------------------------------------------
                    if not free_tmd_mode:
                        # gen_tmd_for_theory already has shape (B,6,x,b) here
                        fake_xsec_eval, _, _ = env.theory.forward(gen_tmd_for_theory)  # (B,... ) or (1,... ) if average=True
                    else:
                        # free_tmd_mode: gen_out is already (B,1,x,Q,b) (or similar) and theory.forward expects "params"
                        fake_xsec_eval, _, _ = env.theory.forward(gen_out)
                
                    # Ensure shape is (B,x,Q,z,qt,phi)
                    if fake_xsec_eval.ndim == 5:
                        fake_xsec_eval = fake_xsec_eval.unsqueeze(0)
                
                    # If theory averaged over batch, expand back to B so "ensemble mean" is well-defined
                    if fake_xsec_eval.shape[0] == 1 and n_eval > 1:
                        fake_xsec_eval = fake_xsec_eval.repeat(n_eval, *([1] * (fake_xsec_eval.ndim - 1)))
                
                    # Match truth batch shape
                    real_xsec_eval = xsec_true.to(device=device, dtype=dtype)
                    if real_xsec_eval.ndim == 5:
                        real_xsec_eval = real_xsec_eval.unsqueeze(0)
                    if real_xsec_eval.shape[0] == 1 and fake_xsec_eval.shape[0] > 1:
                        real_xsec_eval = real_xsec_eval.repeat(fake_xsec_eval.shape[0], *([1] * (real_xsec_eval.ndim - 1)))
                
                    # ----------------------------------------------------
                    # XSEC MSE across eval batch: per-sample -> mean/std
                    # fake_xsec_eval, real_xsec_eval are (B,x,Q,z,qt,phi)
                    # ----------------------------------------------------
                    diff2_x = (fake_xsec_eval - real_xsec_eval) ** 2
                    mse_x_per = diff2_x.mean(dim=tuple(range(1, diff2_x.ndim)))  # (B,)
                    
                    xsec_mse[epoch]     = mse_x_per.mean().detach().cpu()
                    xsec_mse_std[epoch] = mse_x_per.std(unbiased=False).detach().cpu()
                    
                    print(
                        f"  [Epoch {epoch+1}] XSEC MSE (all bins): "
                        f"{xsec_mse[epoch].item():.3e} +/- {xsec_mse_std[epoch].item():.3e}"
                    )
    
        
                    if epoch == n_epochs - 1:
                        last_gen_tmd_input = gen_input_mean.detach().cpu().clone()
                        last_gen_tmd       = tmd_gen_mean.detach().cpu().clone()
                        last_gen_tmd_input_ens = gen_input_ens.detach().cpu().clone()
                        last_gen_tmd_ens       = (tmd_gen_ens.detach().cpu().clone()
                                                  if free_tmd_mode else tmd_gen_ens.detach().cpu().clone())
        
                    print(
                            f"  [Epoch {epoch+1}] TMD MSE (all x, Q^2, b_T): "
                            f"{tmd_mse[epoch].item():.3e} +/- {tmd_mse_std[epoch].item():.3e}"
                            )

    
        # --------------------------
        # Optional epochN diagnostics (EVENT MODE ONLY)
        # --------------------------

        if self.train_on == "events" and self.sample_diagnostics:
            self._dump_fake_event_reps(
                env,
                filename_prefix="epochN",
                dtype=dtype,
                device=device,
            )


    
        # --------------------------
        # Return dict (same keys you had)
        # --------------------------
        return {
            "g_losses": g_losses,
            "d_losses": d_losses,
            "tmd_mse": tmd_mse,
            "xsec_mse": xsec_mse,
            "input_mse": input_mse,
            "tmd_mse_std": tmd_mse_std,
            "xsec_mse_std": xsec_mse_std,
            "input_mse_std": input_mse_std,
            "real_tmd_evolved": real_tmd_evolved,
            "first_gen_tmd": first_gen_tmd,
            "last_gen_tmd": last_gen_tmd,
            "real_tmd_input": real_tmd_input,
            "first_gen_tmd_input": first_gen_tmd_input,
            "last_gen_tmd_input": last_gen_tmd_input,
            "first_gen_tmd_input_ens": first_gen_tmd_input_ens,
            "first_gen_tmd_ens": first_gen_tmd_ens,
            "last_gen_tmd_input_ens": last_gen_tmd_input_ens,
            "last_gen_tmd_ens": last_gen_tmd_ens,
            "xsec_true": xsec_true_cpu,
        }



