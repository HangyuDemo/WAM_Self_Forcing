#!/bin/bash
#SBATCH --job-name=fastwam-libero-jepa
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=4-00:00:00
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

# Override at submit time if needed, for example:
# sbatch --export=ALL,DATASET_ROOT=/home/ha865618/data/LIBERO_Lerobot/libero_10,JEPA_MODEL_ID=/home/ha865618/project/FastWAM/JEPA_checkpoint,GPUS_PER_NODE=4 run_libero_idm_jepa.slurm
export DATASET_ROOT="${DATASET_ROOT:-/home/ha865618/data/LIBERO_Lerobot/libero_10}"
export JEPA_MODEL_ID="${JEPA_MODEL_ID:-/home/ha865618/project/FastWAM/JEPA_checkpoint}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export MAX_STEPS="${MAX_STEPS:-null}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export RUN_ID="${RUN_ID:-idm_jepa_libero10_${SLURM_JOB_ID}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(pwd)/checkpoints_idm_jepa}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

mkdir -p "${MODEL_CACHE_DIR}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${MODEL_CACHE_DIR}}"

echo "DATASET_ROOT=${DATASET_ROOT}"
echo "JEPA_MODEL_ID=${JEPA_MODEL_ID}"
echo "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"

echo "[1/3] preprocess action backbone"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_idm.yaml \
  --output "${MODEL_CACHE_DIR}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  --device cuda \
  --dtype bfloat16

echo "[2/3] precompute text embeddings"
python scripts/precompute_text_embeds.py task=libero10_idm_jepa_2cam224_1e-4

echo "[3/3] train JEPA-IDM"
bash scripts/train_zero1.sh "${GPUS_PER_NODE}" \
  task=libero10_idm_jepa_2cam224_1e-4 \
  num_epochs="${NUM_EPOCHS}" \
  max_steps="${MAX_STEPS}" \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  batch_size="${BATCH_SIZE}" \
  num_workers="${NUM_WORKERS}" \
  model.action_dit_pretrained_path="${MODEL_CACHE_DIR}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
