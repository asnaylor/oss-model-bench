#!/bin/bash
#SBATCH --job-name=omb-agent
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --output=omb-agent-%j.out
#SBATCH --error=omb-agent-%j.err

set -euo pipefail

: "${OMB_BASE_URL:?export OMB_BASE_URL before sbatch}"
: "${OMB_MODEL:?export OMB_MODEL before sbatch}"
: "${OMB_REPO:=$SLURM_SUBMIT_DIR}"

cd "$OMB_REPO"
uv run --offline omb check
uv run --offline omb agent --workers "${OMB_WORKERS:-4}" --no-grade "$@"
