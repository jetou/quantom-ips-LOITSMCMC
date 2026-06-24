# Monte Carlo Methods and Applications Draft

This folder contains a submission-ready LaTeX draft skeleton for the manuscript proposed by Yaohang Li.

## Journal Format

`Monte Carlo Methods and Applications` is published by De Gruyter and accepts research articles through its ScholarOne submission portal. The public journal page and the publisher's author guidance do not provide an MCMA-specific LaTeX template. The current `article`-class manuscript is therefore an appropriate drafting format; apply any journal-specific production formatting requested during submission or after acceptance.

- Journal page: <https://www.degruyter.com/journal/key/mcma/html>
- Submission portal: <https://mc.manuscriptcentral.com/mcma>
- Publisher guidance: <https://www.degruyter.com/publishing/for-authors/journal-authors/prepare-your-journal-submission>

## Section Ownership

| Section | Primary owner | Status |
|---|---|---|
| Introduction and inverse-inference motivation | Yaohang | Placeholder retained |
| Differentiability limitations of inverse CDF and MCMC | Yaohang | Placeholder retained |
| LOITS with MCMC correction | Jitao | Initial draft written |
| NF with MCMC correction | Jitao | Initial draft written |
| Experimental protocol, figures, and tables | Jitao | Structure written; needs final results |
| Discussion and conclusion | Joint | Placeholder retained |

## Build

Compile `main.tex` with `pdflatex`, then `bibtex`, then `pdflatex` twice. For example:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Figures Included

The following figures have been copied into `paper/figures/` and are referenced by `main.tex`:

| File | Paper role | Source |
|---|---|---|
| `ackley_nf_mcmc_correction.png` | Two-dimensional NF proposal and MCMC-correction diagnostic | `examples/results/nf_paired_d10_smooth/` |
| `tmd_event_marginals_its_mcmc_loits.png` | TMD one-dimensional event marginals | `Figures/` |
| `tmd_x_qt_projection_its_mcmc_loits.png` | TMD $x$--$q_T$ event projection | `Figures/` |

## Figures Still Needed for the Final Manuscript

1. One finalized, reproducible acceptance-versus-dimension figure for **LOITS-MCMC vs NF-MCMC**, with the exact proposal definitions and repeated-run uncertainty stated.
2. A matched TMD comparison containing **real events, MCMCLOITSND, and NFMCMCND** from the same density, sample count, and random-seed protocol.
3. The TMD model diagnostic comparing theory input/evolved TMD against the trained model input/evolved TMD.
4. A machine-readable metrics table (acceptance rate, JS divergence, $L^1$ error, density MAE, and any ESS/autocorrelation metric) corresponding to every final figure.

### Ackley Before/After Figure Run

The Ackley demo now writes the complete before/after figure set for both samplers:

```bash
python examples/nf_mcmc_ackley_demo.py \
  --device cuda \
  --outdir examples/results/ackley_final_before_after \
  --n-events 100000 \
  --grid-n 100 \
  --truth-burn-in 5000 \
  --nf-train-steps 3000 \
  --nf-train-samples 100000 \
  --batch-size 2048 \
  --flow-layers 8 \
  --hidden-dim 256 \
  --nf-burn-in 50 \
  --nf-thin 10 \
  --loits-burn-in 50 \
  --loits-thin 10
```

This produces `raw_nf_samples.npy`, `nf_mcmc_samples.npy`, `raw_loits_samples.npy`, and `loits_mcmc_samples.npy`, along with:

- `ackley_nf_mcmc_marginals.png`
- `ackley_loits_mcmc_marginals.png`
- `ackley_nf_loits_corrected_marginals.png`
- `ackley_nf_loits_before_after_2d.png`

## Before Submission

1. Replace every `TODO` marker.
2. Copy the final plots into this folder or use `\includegraphics` paths in `main.tex`.
3. Add inverse-inference references supplied by Yaohang.
4. Verify all numerical claims from repeated runs, not a single seed.
5. Confirm author names, affiliations, funding, and data/code availability.
