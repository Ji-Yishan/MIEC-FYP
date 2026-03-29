#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FYP

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
export TOKENIZERS_PARALLELISM=false

TASK_NAME=rte
MODEL="bert-base-uncased"
OUTPUT_DIR="ckpts/${TASK_NAME}-${MODEL}"

mkdir -p $OUTPUT_DIR


python run_glue.py \
  --model_name_or_path "$MODEL" \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size "${TRAIN_BS:-8}" \
  --per_device_eval_batch_size "${EVAL_BS:-16}" \
  --gradient_checkpointing true \
  --save_total_limit 1 \
  --logging_steps 10 \
  --eval_strategy steps \
  --warmup_steps 100 \
  --learning_rate 5e-5 \
  --num_train_epochs "${EPOCHS:-4}" \
  --output_dir $OUTPUT_DIR \
  --train_file ./RTE/train_fix.tsv \
  --validation_file ./RTE/dev.tsv \
  --test_file ./RTE/test.tsv \
  --overwrite_cache

if [ $? -eq 0 ]; then
    echo "successful training"
else
    echo "fail to train, check output logs to debug"
fi
