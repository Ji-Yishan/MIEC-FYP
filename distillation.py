# distill_cpu_fixed.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    set_seed,
)
from datasets import load_from_disk
from tqdm import tqdm
import time
import math

# ----------------------------
# 配置
# ----------------------------
MODEL_DIR = "./local_model/qwen1_5_0_5b"
DATA_DIR = "./local_data"
DATASET_PATH = os.path.join(DATA_DIR, "wikitext2_test200")

BATCH_SIZE = 2
MAX_LENGTH = 128
TEMPERATURE = 2.0
ALPHA = 0.5  # 蒸馏损失权重：alpha * KL + (1 - alpha) * CE
LEARNING_RATE = 5e-5
EPOCHS = 3
SEED = 42

set_seed(SEED)
device = torch.device("cpu")
print(f"Using device: {device}")

# ----------------------------
# 加载 tokenizer 和 teacher 模型
# ----------------------------
print("Loading tokenizer and teacher model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device).eval()

# ----------------------------
# Student 模型（此处仍用同架构，仅用于演示蒸馏流程）
# ----------------------------
student_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device)

student_model.train()

# ----------------------------
# 加载本地数据集
# ----------------------------
print("Loading local dataset...")
raw_dataset = load_from_disk(DATASET_PATH)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True
    )

tokenized_ds = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Tokenizing"
)

# 创建 DataLoader
train_loader = DataLoader(
    tokenized_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=default_data_collator
)

# ----------------------------
# 蒸馏训练
# ----------------------------
optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
ce_loss_fn = torch.nn.CrossEntropyLoss()

print("\n🚀 Starting Knowledge Distillation Training...\n")

for epoch in range(EPOCHS):  # ✅ 修复：原错误行已修正
    total_loss = 0.0
    student_model.train()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Teacher logits (no grad)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Student logits
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Shift for causal LM
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Soft targets
        soft_targets = torch.softmax(shift_teacher_logits / TEMPERATURE, dim=-1)
        soft_probs = torch.log_softmax(shift_student_logits / TEMPERATURE, dim=-1)

        # KL Loss (蒸馏损失)
        kl_loss = kl_loss_fn(soft_probs, soft_targets) * (TEMPERATURE ** 2)

        # Hard Loss (标准交叉熵)
        ce_loss = ce_loss_fn(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Total loss
        loss = ALPHA * kl_loss + (1 - ALPHA) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

# ----------------------------
# 保存蒸馏后的 student 模型
# ----------------------------
student_output_dir = os.path.join(DATA_DIR, "distilled_student")
student_model.save_pretrained(student_output_dir)
tokenizer.save_pretrained(student_output_dir)
print(f"\n✅ Distilled student model saved to: {student_output_dir}")

# ----------------------------
# 评估：大小、准确度（Perplexity）、推理速度
# ----------------------------
print("\n🔍 Evaluating distilled student model...")

# 1. 模型大小（参数量）
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(student_model)
print(f"✅ Trainable Parameters: {num_params:,} (~{num_params / 1e6:.1f}M)")

# 2. Perplexity on same dataset
student_model.eval()
total_ppl = 0.0
count = 0

with torch.no_grad():
    for batch in tqdm(train_loader, desc="Computing Perplexity"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        lm_loss = outputs.loss
        if not torch.isnan(lm_loss):
            ppl = math.exp(lm_loss.item())
            total_ppl += ppl
            count += 1

avg_ppl = total_ppl / count if count > 0 else float('nan')
print(f"✅ Average Perplexity on wikitext2_test200: {avg_ppl:.2f}")

# 3. 推理速度（tokens/sec）
prompt = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Warm-up
for _ in range(2):
    _ = student_model.generate(**inputs, max_new_tokens=10)

start_time = time.time()
outputs = student_model.generate(**inputs, max_new_tokens=50)
end_time = time.time()

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
tokens_generated = len(tokenizer(generated_text)["input_ids"])
latency = end_time - start_time
throughput = tokens_generated / latency

print(f"✅ Inference Speed: {throughput:.2f} tokens/sec")
print(f"✅ Sample Output: {generated_text[:100]}...")

print("\n🎉 Distillation and evaluation completed successfully!")