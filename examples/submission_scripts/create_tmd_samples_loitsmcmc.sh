#!/bin/bash
set -e

# Run from repo root, or set REPO_DIR=/path/to/quantom-ips.
repoDir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [ -n "${VENV_PATH:-}" ]; then
  source "$VENV_PATH/bin/activate"
fi

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

resultDir="${RESULT_DIR:-$repoDir/examples/sample_data}"
device="${DEVICE:-cuda}"
sampler="${SAMPLER:-ITS}"
theory="${THEORY:-SIDIS_masked}"
dataName="${DATA_NAME:-sidis_dataset}"
nSamples="${N_SAMPLES:-1000000}"
nRepeats="${N_REPEATS:-10}"
seed="${SEED:-0}"

mkdir -p "$resultDir"

CMD="python $repoDir/examples/create_samples.py hydra.run.dir=$resultDir"
CMD="$CMD device=$device"
CMD="$CMD $KIN_OVERRIDES"
CMD="$CMD env/sampler=$sampler env/theory=$theory"
CMD="$CMD n_samples=$nSamples n_repeats=$nRepeats seed=$seed data_name=$dataName"

echo "Running: $CMD"
eval $CMD
