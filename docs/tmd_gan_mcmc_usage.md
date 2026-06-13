# TMD GAN and MCMC Sampler Evaluation

This project keeps the original QuantOm-IPS workflows intact and adds a separate
TMD/SIDIS path copied from the diffusion snapshot project.

## Dependencies

The TMD/SIDIS theory modules require the normal project dependencies plus
`numpy`, `scipy`, `numba`, and `pyyaml`. They also require a working LHAPDF
Python installation and the `JAM20-SIDIS_PDF_proton_nlo` and
`JAM20-SIDIS_FF_pion_nlo` sets on the training machine.

## Train TMD GAN

Put `density.npy` in the same directory as `sidis_dataset.npy`, or pass
`env.sample_dir=D:\path\to\sample_data`. The TMD GAN optimizer uses
`density.npy` for physics diagnostics during training.

```bash
cd D:\PersonalFiles\codex\LOITS\quantom-ips-main\quantom-ips-main

python examples\tmd_workflow.py opt=TMDGANCC env=TMDQuantomEnv ^
  env/theory=SIDIS_masked_cc env/sampler=ITS ^
  dataloader=tmd_events dataloader.dataset.path=D:\path\to\sidis_dataset.npy ^
  opt.n_epochs=50 opt.train_on=events env.mode=events ^
  opt.progress_bar=True opt.log_every=10 opt.mse_freq=10 ^
  hydra.run.dir=D:\path\to\run_dir
```

Training uses `ITS` because the MCMC sampler is evaluation-only and is not
intended to provide training gradients.

You can also use the submission helper script:

```bash
bash examples/submission_scripts/training_tmd_gan_loitsmcmc.sh \
  --epochs 500 \
  --run-dir examples/results/tmd_gan_mcmc_500
```

For long runs, `opt.progress_bar=True` keeps the terminal compact by showing
losses in a tqdm postfix instead of printing several lines every epoch. If you
turn the progress bar off, `opt.log_every=10` controls how often one summary
line is printed. `opt.mse_freq=10` computes the slower TMD/XSEC diagnostics
every 10 epochs while still computing them at the final epoch.

## Compare Sampler Filters

```bash
python examples\tmd_evaluate_sampler_filters.py ^
  --rundir D:\path\to\run_dir ^
  --data D:\path\to\sidis_dataset.npy ^
  --n-events 10000 ^
  --n-repeats 10 ^
  --outdir D:\path\to\eval_dir ^
  --samplers ITS MCMCLOITSND NFMCMCND ^
  --nf-train-steps 300 ^
  --nf-train-samples 4096 ^
  --projections x,qT x,Q2 z,qT qT,phi
```

The evaluation script saves:

- `sampler_filter_metrics.json`
- `real_events.npy`
- `its_events.npy`
- `its_events_reps.npy`
- `mcmcloitsnd_events.npy`
- `mcmcloitsnd_events_reps.npy`
- `nfmcmcnd_events.npy`
- `nfmcmcnd_events_reps.npy`
- `tmd_true_vs_model_last_epoch.png`
- `sampler_marginals.png`
- `sampler_event_distributions_uncertainty.png`
- `sampler_projection_x_qT_true_its_mcmc.png`
- `sampler_projection_x_qT_differences.png`
- matching `true_its_mcmc` and `differences` plots for each requested projection

The sampler figures annotate the mean acceptance rates for accept/reject
samplers such as `MCMCLOITSND` and `NFMCMCND`. `ITS` is not an accept/reject
sampler, so its acceptance rate is not reported.

To compare only the two MCMC-corrected samplers:

```bash
python examples\tmd_evaluate_sampler_filters.py ^
  --rundir D:\path\to\run_dir ^
  --data D:\path\to\sidis_dataset.npy ^
  --n-events 10000 ^
  --n-repeats 10 ^
  --outdir D:\path\to\eval_dir ^
  --samplers MCMCLOITSND NFMCMCND
```

The TMD plot is the model sanity check: it compares truth and generator TMDs
before and after theory evolution. The sampler plots compare physical event
clouds after the SIDIS cross-section density has been sampled, so their axes are
event variables such as `x`, `Q2`, `z`, `qT`, and `phi`; there is no sampler
`b_T` output.
