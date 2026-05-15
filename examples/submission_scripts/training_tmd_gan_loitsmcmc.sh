#!/bin/bash
set -e

# Run from repo root, or set REPO_DIR=/path/to/quantom-ips.
repoDir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Optional: activate the same environment you used for the old TMD runs.
# Example:
#   VENV_PATH=/w/jam-sciwork24/kmbraga/SciDAC/Quantom/quantom-ips/.venv bash examples/submission_scripts/training_tmd_gan_loitsmcmc.sh
if [ -n "${VENV_PATH:-}" ]; then
  source "$VENV_PATH/bin/activate"
fi

# Prefer conda's C++ runtime when using a conda environment with LHAPDF.
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${LHAPDF_DATA_PATH:-}" ]; then
  for p in \
    "$CONDA_PREFIX/share/LHAPDF" \
    "/home/jxu004/miniconda3/envs/tmd/share/LHAPDF" \
    "/home/jxu004/miniconda3/envs/pyhigh2/share/LHAPDF" \
    "/w/jam-sciwork24/ccocuzza/lhapdf/python3/sets"; do
    if [ -d "$p/JAM20-SIDIS_PDF_proton_nlo" ] && [ -d "$p/JAM20-SIDIS_FF_pion_nlo" ]; then
      export LHAPDF_DATA_PATH="$p"
      break
    fi
  done
fi

echo "LHAPDF_DATA_PATH=${LHAPDF_DATA_PATH:-unset}"

export PYTHONPATH="$repoDir/src:${PYTHONPATH:-}"
export QUANTOM_IPS_GRIDS_DIR="${QUANTOM_IPS_GRIDS_DIR:-$repoDir/examples/submission_scripts/grids}"
mkdir -p "$QUANTOM_IPS_GRIDS_DIR"

KIN_YAML="${KIN_YAML:-$repoDir/examples/sidis_11GeV.yaml}"
KIN_OVERRIDES="$(python "$repoDir/examples/resolve_kinematics.py" --yaml "$KIN_YAML" 2>/dev/null || true)"

if [ -z "$KIN_OVERRIDES" ]; then
  echo "WARNING: resolve_kinematics failed; using fallback grid."
  KIN_OVERRIDES="env.theory.nx=20 env.theory.nQ2=20 env.theory.nz=10 env.theory.nbt=30 env.theory.nqt=10 env.theory.nphi=5 env.theory.Ebeam=11.0 env.theory.x_min=0.048 env.theory.x_max=1.0 env.theory.Q2_min=1.0 env.theory.Q2_max=20.6 env.theory.z_min=0.2 env.theory.z_max=0.8 env.theory.bt_min=0.001 env.theory.bt_max=6.0 env.theory.qt_min=0.0 env.theory.qt_max=1.0 env.theory.phi_min=0.0 env.theory.phi_max=6.283 env.theory.W2_min=4.0 env.theory.log_x=True env.theory.log_Q2=False env.theory.log_z=False env.theory.qT_lim=0.3"
fi

nx=$(echo "$KIN_OVERRIDES"  | sed -n 's/.*env.theory.nx=\([^ ]*\).*/\1/p')
nbt=$(echo "$KIN_OVERRIDES" | sed -n 's/.*env.theory.nbt=\([^ ]*\).*/\1/p')

resultDir="${RESULT_DIR:-$repoDir/examples/results/tmd_gan_mcmc}"
sampleDir="${SAMPLE_DIR:-$repoDir/examples/sample_data}"
dataDir="${DATA_PATH:-$sampleDir/sidis_dataset.npy}"

nEpochs="${N_EPOCHS:-50}"
B_gen="${B_GEN:-40}"
B_real="${B_REAL:-40}"
N_events="${N_EVENTS:-1000}"
steps_per_epoch="${STEPS_PER_EPOCH:-1}"
noise_dim="${NOISE_DIM:-128}"
nf="${N_FLAV_OUT:-1}"
device="${DEVICE:-cuda}"
event_norm="${EVENT_NORM:-False}"
sampler="${SAMPLER:-ITS}"
theory="${THEORY:-SIDIS_masked_cc}"

CMD="python $repoDir/examples/tmd_workflow.py hydra.run.dir=$resultDir"
CMD="$CMD device=$device"
CMD="$CMD opt=TMDGANCC env=TMDQuantomEnv"
CMD="$CMD env/sampler=$sampler env/theory=$theory"
CMD="$CMD dataloader=tmd_events dataloader.dataset.path=$dataDir"
CMD="$CMD env.sample_dir=$sampleDir"
CMD="$CMD opt.n_epochs=$nEpochs opt.train_on=events env.mode=events"
CMD="$CMD env.normalize_events=$event_norm"
CMD="$CMD $KIN_OVERRIDES"
CMD="$CMD model@opt.discriminator=MLPDiscriminatorSigOut"
CMD="$CMD opt.discriminator.input_dim=5 opt.discriminator.output_dim=1"
CMD="$CMD model@opt.generator=TMDConv2DTGenerator"
CMD="$CMD opt.batch_size=$B_gen opt.noise_dim=$noise_dim"
CMD="$CMD opt.generator.mlp_in=$noise_dim"
CMD="$CMD opt.generator.image_size=[$nx,$nbt]"
CMD="$CMD opt.generator.out_channels=$nf"
CMD="$CMD dataloader.batch_size=$B_real"
CMD="$CMD dataloader.dataset.n_samples=$N_events"
CMD="$CMD opt.steps_per_epoch=$steps_per_epoch"

echo "Running: $CMD"
eval $CMD
