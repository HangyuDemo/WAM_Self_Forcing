#!/bin/bash
#SBATCH --job-name=fastwam-libero10-idm-sf
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
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

# You can override these at submit time, for example:
# sbatch --export=ALL,BATCH_SIZE=4,GRAD_ACCUM_STEPS=8,MAX_STEPS=20000,NUM_WORKERS=4 run_libero10_only_idm_self_forcing.slurm
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export MAX_STEPS="${MAX_STEPS:-null}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
# VIDEO_CHUNK_SIZE=3 matches the Self-Forcing-style block AR setting more closely:
# each block contains 3 video latent frames, and later blocks attend to previous generated cache.
export VIDEO_CHUNK_SIZE="${VIDEO_CHUNK_SIZE:-3}"
export VIDEO_ROLLOUT_STEPS="${VIDEO_ROLLOUT_STEPS:-4}"
export CONTEXT_NOISE="${CONTEXT_NOISE:-0}"
export RUN_ID="${RUN_ID:-idm_self_forcing_libero10_${SLURM_JOB_ID}}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(pwd)/checkpoints_idm_sf}"

# Make conda-provided FFmpeg/pyav libraries visible to torchcodec when available.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

mkdir -p "${MODEL_CACHE_DIR}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${MODEL_CACHE_DIR}}"

echo "[1/3] preprocess action backbone"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_idm.yaml \
  --output "${MODEL_CACHE_DIR}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  --device cuda \
  --dtype bfloat16

echo "[2/3] precompute text embeddings"
python scripts/precompute_text_embeds.py task=libero10_only_idm_2cam224_1e-4

echo "[3/3] train IDM self-forcing on libero_10 only"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${GPUS_PER_NODE}" \
  scripts/train_idm_self_forcing.py \
  "task=libero10_only_idm_2cam224_1e-4" \
  "output_dir=./runs/libero10_only_idm_self_forcing/${RUN_ID}" \
  "num_epochs=${NUM_EPOCHS}" \
  "max_steps=${MAX_STEPS}" \
  "gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
  "batch_size=${BATCH_SIZE}" \
  "num_workers=${NUM_WORKERS}" \
  "+self_forcing.video_chunk_size=${VIDEO_CHUNK_SIZE}" \
  "+self_forcing.video_rollout_steps=${VIDEO_ROLLOUT_STEPS}" \
  "+self_forcing.context_noise=${CONTEXT_NOISE}" \
  "wandb.name=libero10_only_idm_self_forcing"
