#!/bin/bash
#SBATCH --job-name=fastwam-libero10-idm
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ha865618/project/FastWAM/slurm_logs/%x-%j.out
#SBATCH --error=/home/ha865618/project/FastWAM/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /home/ha865618/project/FastWAM/slurm_logs
cd /home/ha865618/project/FastWAM

source /lustre/fs1/home/ha865618/miniforge3/etc/profile.d/conda.sh
conda activate fastwam
module purge
module load gcc/gcc-12.2.0
module load cuda/cuda-13.1.0

# Optional: override at submit time, e.g.
# sbatch --export=ALL,GPUS_PER_NODE=8,NUM_EPOCHS=20,MAX_STEPS=20000,GRAD_ACCUM_STEPS=4,BATCH_SIZE=8,NUM_WORKERS=4 run_libero10_only_idm.slurm
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export MAX_STEPS="${MAX_STEPS:-null}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export RUN_ID="${RUN_ID:-idm_libero10_${SLURM_JOB_ID}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Make conda-provided FFmpeg/pyav libraries visible to torchcodec when available.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(pwd)/checkpoints}"

echo "[1/3] preprocess action backbone"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_idm.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16

echo "[2/3] precompute text embeddings"
python scripts/precompute_text_embeds.py task=libero10_only_idm_2cam224_1e-4

echo "[3/3] train IDM on libero_10 only"
bash scripts/train_zero1.sh "${GPUS_PER_NODE}" \
  task=libero10_only_idm_2cam224_1e-4 \
  num_epochs="${NUM_EPOCHS}" \
  max_steps="${MAX_STEPS}" \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  batch_size="${BATCH_SIZE}" \
  num_workers="${NUM_WORKERS}"
