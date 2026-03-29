#!/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FYP

# --- Apple M5 Pro 环境预设 ---
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
export TOKENIZERS_PARALLELISM=false

# --- 配置 ---
TASK_NAME=rte
MODEL="bert-base-uncased"
OUTPUT_DIR="ckpts/${TASK_NAME}-${MODEL}"

echo "🚀 启动 Apple M5 Pro 专属训练任务"
echo "模型: $MODEL | 任务: $TASK_NAME"
mkdir -p $OUTPUT_DIR

# --- 执行训练 ---
# 注意：反斜杠 \ 后面不能有任何字符（包括空格），必须直接换行
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
    echo "🎉 M5 Pro 训练完成！任务执行成功。"
else
    echo "❌ 训练失败。请检查日志。"
fi

# ===================================================================
# 脚本参数说明 (注释统一区)
# ===================================================================
# --model_name_or_path: 预训练模型名称或路径
# --task_name: GLUE 任务名称 (rte)
# --do_train: 启用训练
# --do_eval: 启用评估
# --max_seq_length: 最大序列长度
# --per_device_train_batch_size: 训练批次大小
# --per_device_eval_batch_size: 评估批次大小
# --bf16: 启用 Brain Float 16 精度 (M5 推荐)
# --save_total_limit: 保存的检查点数量限制
# --logging_steps: 日志记录步数
# --eval_strategy: 评估策略
# --warmup_steps: 预热步数
# --output_dir: 输出目录（建议使用新目录避免冲突）
# --learning_rate: 学习率
# --num_train_epochs: 训练轮数
# --output_dir: 输出目录
# --train_file: 训练集文件路径
# --validation_file: 验证集文件路径
# --test_file: 测试集文件路径
# --overwrite_cache: 覆盖缓存数据集