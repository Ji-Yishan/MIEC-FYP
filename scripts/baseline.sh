#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FYP

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
export TOKENIZERS_PARALLELISM=false

TRAIN_FILE="${TRAIN_FILE:-./RTE/train_fix.tsv}"
VALID_FILE="${VALID_FILE:-./RTE/dev.tsv}"
OUT_DIR="${OUT_DIR:-./ckpts/mb_rte_local}"
TEACHER="${TEACHER:-./ckpts/rte-bert-large-uncased}"

# overwrite_output_dir now means "clear prior training artifacts in output_dir before training"
python baseline.py \
  --model_name_or_path "bert-base-uncased" \
  --teacher "${TEACHER}" \
  --task_name rte \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${VALID_FILE}" \
  --do_train \
  --do_eval \
  --seed 96 \
  --max_seq_length 128 \
  --per_device_train_batch_size "${TRAIN_BS:-8}" \
  --per_device_eval_batch_size "${EVAL_BS:-4}" \
  --gradient_accumulation_steps "${GRAD_ACC:-4}" \
  --learning_rate "${LR:-2e-5}" \
  --num_train_epochs "${EPOCHS:-5}" \
  --output_dir "${OUT_DIR}" \
  --overwrite_output_dir \
  --overwrite_cache \
  --temperature 5.0 \
  --kl_kd \
  --objective_strategy "${OBJECTIVE_STRATEGY:-uncertainty}" \
  --kd_kl_alpha "${KD_KL_ALPHA:-0.5}" \
  --kd_rep_alpha "${KD_REP_ALPHA:-0.5}" \
  --ce_alpha 0.5 \
  --pad_to_max_length true
