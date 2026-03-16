#!/bin/bash

python prob1.py \
  --student_name_or_path "bert-base-uncased" \
  --small_teacher_name_or_path "./ckpts/rte-bert-base-uncased" \
  --large_teacher_name_or_path "./ckpts/rte-bert-large-uncased" \
  --task_name rte \
  --train_file ./RTE/train_fixed.tsv \
  --validation_file ./RTE/dev.tsv \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --output_dir "./ckpts/dynamic_kd_rte_local" \
  --overwrite_output_dir \
  --overwrite_cache \
  --temperature 5.0 \
  --kd_alpha 0.5 \
  --ce_alpha 0.5 \
  --dynamic_teacher_selection true \
  --dynamic_objective_weighting true \
  --student_layers_to_distill "all" \
  --pad_to_max_length true
