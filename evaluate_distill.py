# -*- coding: utf-8 -*-
"""
评估蒸馏后的 Qwen1.5-0.5B 模型（使用与蒸馏训练完全相同的测试集）
✅ 测试集：./local_data/wikitext2_test200.json
✅ CPU 安全 | Windows 兼容 | 16GB RAM 友好
✅ 正确 PPL（忽略 padding）
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 1. 配置
# ----------------------------
device = "cpu"
DISTILLED_MODEL_PATH = "./distilled_student_cpu"
BASE_MODEL_PATH = "./local_model/qwen1_5_0_5b"
TEST_DATASET_PATH = "./local_data/wikitext2_test200.json"  # ⭐ 与蒸馏训练完全一致
BATCH_SIZE = 4

print(f"Device: {device}")
print(f"Using test set: {TEST_DATASET_PATH}")

# ----------------------------
# 2. 加载 tokenizer 和模型
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(DISTILLED_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device).eval()

print("Loading distilled model...")
distilled_model = AutoModelForCausalLM.from_pretrained(
    DISTILLED_MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device).eval()

# ----------------------------
# 3. 工具函数
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(model):
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    try:
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    finally:
        os.close(fd)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return size_mb

# ----------------------------
# 4. 加载本地测试集（与蒸馏训练完全相同）
# ----------------------------
print(f"\nLoading local test dataset from {TEST_DATASET_PATH}...")
raw_test = load_dataset("json", data_files=TEST_DATASET_PATH, split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True
    )

tokenized_test = raw_test.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Loaded {len(tokenized_test)} test examples (same as distillation training set)")

# ----------------------------
# 5. 正确的 PPL 计算（忽略 padding）
# ----------------------------
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略 padding token

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            num_tokens = (labels != -100).sum().item()
            if num_tokens > 0:
                total_nll += outputs.loss.item() * num_tokens
                total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf')
    return torch.exp(torch.tensor(total_nll / total_tokens)).item()

print("\nComputing Perplexity on the SAME test set used in distillation...")
ppl_base = compute_perplexity(base_model, test_loader, device)
ppl_distilled = compute_perplexity(distilled_model, test_loader, device)

# ----------------------------
# 6. 推理速度测试
# ----------------------------
def measure_inference_speed(model, dataloader, device, num_runs=20):
    model.eval()
    latencies = []
    # Warmup
    for _ in range(3):
        for batch in dataloader:
            input_ids = batch["input_ids"][:1].to(device)
            with torch.no_grad():
                _ = model(input_ids)
            break
    # Measure
    for _ in range(num_runs):
        for batch in dataloader:
            input_ids = batch["input_ids"][:1].to(device)
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            break
    return sum(latencies) / len(latencies), 1000 / (sum(latencies) / len(latencies))

latency_base, tp_base = measure_inference_speed(base_model, test_loader, device)
latency_distilled, tp_distilled = measure_inference_speed(distilled_model, test_loader, device)

# ----------------------------
# 7. 模型信息
# ----------------------------
params_base = count_parameters(base_model)
params_distilled = count_parameters(distilled_model)
size_base = get_model_size_mb(base_model)
size_distilled = get_model_size_mb(distilled_model)

# ----------------------------
# 8. 生成样例
# ----------------------------
prompts = ["Natural language processing", "The future of AI is"]

def generate_text(model, prompt, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ----------------------------
# 9. 打印报告
# ----------------------------
print("\n" + "=" * 80)
print("📊 DISTILLATION EVALUATION (SAME TEST SET AS TRAINING)")
print("=" * 80)

print(f"\n📦 MODEL SIZE:")
print(f"  Base:      {params_base:,} params | {size_base:.1f} MB")
print(f"  Distilled: {params_distilled:,} params | {size_distilled:.1f} MB")

print(f"\n⚡ INFERENCE SPEED:")
print(f"  Base:      {latency_base:.2f} ms/sample")
print(f"  Distilled: {latency_distilled:.2f} ms/sample")
print(f"  Speedup:   {latency_base / latency_distilled:.2f}x")

print(f"\n🎯 PERPLEXITY (on identical test set):")
print(f"  Base PPL:      {ppl_base:.2f}")
print(f"  Distilled PPL: {ppl_distilled:.2f}")
if ppl_base > 0:
    delta = ppl_distilled - ppl_base
    pct = (ppl_distilled / ppl_base - 1) * 100
    print(f"  Δ PPL:         {delta:+.2f} ({pct:+.1f}%)")

print(f"\n📝 GENERATION EXAMPLES:")
for prompt in prompts:
    base_out = generate_text(base_model, prompt, tokenizer, device)
    distilled_out = generate_text(distilled_model, prompt, tokenizer, device)
    print(f"\nPrompt: {prompt}")
    print(f"  Base:      {base_out}")
    print(f"  Distilled: {distilled_out}")

print("\n✅ Evaluation completed using the EXACT same test set as distillation!")