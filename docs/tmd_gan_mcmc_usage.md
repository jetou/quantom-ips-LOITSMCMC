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
  hydra.run.dir=D:\path\to\run_dir
```

Training uses `ITS` because the MCMC sampler is evaluation-only and is not
intended to provide training gradients.

## Compare Sampler Filters

```bash
python examples\tmd_evaluate_sampler_filters.py ^
  --rundir D:\path\to\run_dir ^
  --data D:\path\to\sidis_dataset.npy ^
  --n-events 10000 ^
  --outdir D:\path\to\eval_dir
```

The evaluation script saves:

- `sampler_filter_metrics.json`
- `real_events.npy`
- `its_events.npy`
- `mcmcloitsnd_events.npy`
- `sampler_marginals.png`
- `sampler_x_qt_projection.png`
