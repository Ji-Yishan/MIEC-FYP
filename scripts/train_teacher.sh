
TASK_NAME=rte
# MODEL="bert-large-uncased"
MODEL="bert-base-uncased"
python run_glue.py \
  --model_name_or_path "$MODEL" \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_total_limit 1 \
  --logging_steps 100 \
  --evaluation_strategy steps --warmup_ratio 0.05 --overwrite_output_dir  \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --output_dir ckpts/${TASK_NAME}-$MODEL \
  --train_file ./RTE/train_fixed.tsv \
  --validation_file ./RTE/dev.tsv \
  --test_file ./RTE/test.tsv \
  --overwrite_cache  \

