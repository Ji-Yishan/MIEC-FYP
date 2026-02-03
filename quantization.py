# -*- coding: utf-8 -*-
"""
安全量化 Qwen1.5-0.5B + 正确评估（含 attention_mask 修复）
✅ 使用 torch.quantization.quantize_dynamic
✅ 保留 PPL、速度、大小、生成质量对比
✅ 修复 Windows tempfile 问题
✅ 显式输出：推理速度、模型大小、正确率（PPL）
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 1. 设置
# ----------------------------
device = "cpu"
model_name = "Qwen/Qwen1.5-0.5B"
batch_size = 4

print(f"Device: {device} | Applying dynamic quantization...")

# ----------------------------
# 2. 加载 tokenizer 和原始模型
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading original model for comparison...")
original_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, trust_remote_code=True
).to(device)
original_model.eval()

print("Applying dynamic quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    original_model,
    {nn.Linear},  # 量化所有 Linear 层
    dtype=torch.qint8
)
quantized_model.eval()
print("✅ Quantization completed.")

# ----------------------------
# 3. 工具函数（Windows-safe）
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
# 4. 加载测试数据（保留 attention_mask！）
# ----------------------------
<<<<<<< HEAD

print("\nLoading wikitext FYP set (test[:200]) for evaluation...")
=======
print("\nLoading wikitext test set (test[:200]) for evaluation...")
>>>>>>> 18f2478 (fix error in code)
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:200]")

test_dataset = test_dataset.filter(lambda x: len(x["text"].strip()) > 10)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_attention_mask=True
    )

tokenized_test = test_dataset.map(
    tokenize, batched=True, remove_columns=["text"]
)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False)

# ----------------------------
# 5. ✅ 修复后的 PPL 计算（正确忽略 padding）
# ----------------------------
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略 padding

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            num_tokens = (labels != -100).sum().item()
            if num_tokens > 0:
                total_nll += outputs.loss.item() * num_tokens
                total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf')
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item()

print("\nComputing Perplexity (with attention_mask)...")
ppl_original = compute_perplexity(original_model, test_loader, device)
ppl_quantized = compute_perplexity(quantized_model, test_loader, device)

# ----------------------------
# 6. 推理速度测试
# ----------------------------
def measure_inference_speed(model, dataloader, device, num_runs=20):
    model.eval()
    latencies = []
    # Warmup
    for _ in range(5):
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
    avg_latency = sum(latencies) / len(latencies)
    throughput = 1000 / avg_latency
    return avg_latency, throughput

print("Measuring inference speed...")
latency_orig, throughput_orig = measure_inference_speed(original_model, test_loader, device)
latency_quant, throughput_quant = measure_inference_speed(quantized_model, test_loader, device)

# ----------------------------
# 7. 模型大小与参数量
# ----------------------------
params_orig = count_parameters(original_model)
params_quant = count_parameters(quantized_model)
size_orig = get_model_size_mb(original_model)
size_quant = get_model_size_mb(quantized_model)

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
# 9. 提取核心指标（显式命名）
# ----------------------------
# 正确率用 PPL 表示（越低越好）
accuracy_original = ppl_original
accuracy_quantized = ppl_quantized

# 推理速度：以 throughput（samples/sec）为主要指标
inference_speed_original = throughput_orig  # samples/sec
inference_speed_quantized = throughput_quant

# 模型大小（MB）
model_size_original = size_orig
model_size_quantized = size_quant

# ----------------------------
# 10. 打印结构化报告
# ----------------------------
print("\n" + "=" * 80)
print("📊 QUANTIZATION EVALUATION REPORT (Dynamic Quantization)")
print("=" * 80)

print(f"\n📦 MODEL SIZE (Disk):")
print(f"  Original: {model_size_original:.1f} MB")
print(f"  Quantized: {model_size_quantized:.1f} MB")
print(f"  Reduction: ↓{(1 - model_size_quantized / model_size_original) * 100:.1f}%")

print(f"\n⚡ INFERENCE SPEED:")
print(f"  Original: {inference_speed_original:.2f} samples/sec")
print(f"  Quantized: {inference_speed_quantized:.2f} samples/sec")
print(f"  Speedup: {inference_speed_quantized / inference_speed_original:.2f}x")

print(f"\n🎯 ACCURACY (Perplexity on wikitext FYP, lower is better):")
print(f"  Original: {accuracy_original:.2f}")
print(f"  Quantized: {accuracy_quantized:.2f}")
if accuracy_original > 0:
    delta_pct = (accuracy_quantized / accuracy_original - 1) * 100
    print(f"  Change: {delta_pct:+.1f}%")

print(f"\n📝 GENERATION QUALITY:")
for prompt in prompts:
    orig_gen = generate_text(original_model, prompt, tokenizer, device)
    quant_gen = generate_text(quantized_model, prompt, tokenizer, device)
    print(f"\nPrompt: {prompt}")
    print(f"  Original: {orig_gen}")
    print(f"  Quantized: {quant_gen}")

print("\n✅ Evaluation complete with CORRECT PPL calculation!")