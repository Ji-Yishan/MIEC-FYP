# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dynamic Knowledge Distillation for GLUE Tasks (Local Dataset Version).
Compatible with: Python 3.7+, Transformers 4.5.1+, PyTorch 1.13+
Features: Local TSV/CSV loading, Quote Sanitization, Dynamic KD, Fixed Args, Smart Tokenizer Fallback, Robust Label Mapping.
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    PreTrainedModel,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import logging as hf_logging

# A list of all GLUE tasks
glue_tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

# Mapping of task names to their respective key names in the dataset
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    'imdb': ("text", None),
    'boolq': ("passage", "question"),
    "sst5": ("sentence", None)
}

# Setup basic logger first
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 参数定义 (Arguments)
# ==============================================================================
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    student_name_or_path: str = field(
        metadata={"help": "Path to pretrained student model"}
    )
    small_teacher_name_or_path: str = field(
        metadata={"help": "Path to pretrained SMALL teacher model"}
    )
    large_teacher_name_or_path: str = field(
        metadata={"help": "Path to pretrained LARGE teacher model"}
    )
    
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name. If not provided, will try cache then small teacher."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login`."},
    )
    
    # --- 动态蒸馏特定参数 ---
    temperature: float = field(
        default=5.0,
        metadata={"help": "Temperature for KD loss"}
    )
    kd_alpha: float = field(
        default=0.5,
        metadata={"help": "Base weight for KD loss"}
    )
    ce_alpha: float = field(
        default=0.5,
        metadata={"help": "Base weight for Student CE loss"}
    )
    
    dynamic_teacher_selection: bool = field(
        default=True,
        metadata={"help": "Enable Dynamic Teacher Adoption"}
    )
    dynamic_objective_weighting: bool = field(
        default=True,
        metadata={"help": "Enable Dynamic Supervision Adjustment"}
    )
    
    student_layers_to_distill: str = field(
        default="all",
        metadata={"help": "Layers to distill (e.g., 'all' or '0,1,2')"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv, tsv, or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv, tsv, or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv, tsv, or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "tsv"], "`train_file` should be a csv, tsv, or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv, tsv, or json) as `train_file`."

# ==============================================================================
# 2. 工具函数 (Utilities)
# ==============================================================================
def sanitize_quotes(example, text_columns):
    """
    检查指定文本列中的双引号 (") 和单引号 (') 是否成对出现。
    如果不成对（数量为奇数），则移除该列中所有的该类引号，以防止 TSV/CSV 解析错位。
    """
    for col in text_columns:
        if col in example and example[col] is not None:
            text = str(example[col])
            
            # 检查双引号
            if text.count('"') % 2 != 0:
                logger.warning("Detected unbalanced double quotes in column '{}'. Removing all double quotes from this sample.".format(col))
                text = text.replace('"', '')
            
            # 检查单引号
            if text.count("'") % 2 != 0:
                logger.warning("Detected unbalanced single quotes in column '{}'. Removing all single quotes from this sample.".format(col))
                text = text.replace("'", '')
            
            example[col] = text
            
    return example

def is_valid_tokenizer_path(path):
    """检查路径是否包含有效的 tokenizer 文件"""
    if not os.path.isdir(path):
        return False
    required_files = ['vocab.txt', 'tokenizer.json', 'tokenizer_config.json']
    return os.path.exists(os.path.join(path, 'vocab.txt')) or \
           os.path.exists(os.path.join(path, 'tokenizer.json'))

def resolve_tokenizer_path(model_args):
    """
    智能解析 Tokenizer 路径
    """
    user_defined = model_args.tokenizer_name
    student_path = model_args.student_name_or_path
    teacher_path = model_args.small_teacher_name_or_path
    
    if user_defined:
        logger.info("Using user-defined tokenizer: {}".format(user_defined))
        return user_defined
    
    if os.path.isdir(student_path):
        if is_valid_tokenizer_path(student_path):
            logger.info("Using Student's local path as tokenizer: {}".format(student_path))
            return student_path
        else:
            logger.warning("Student path {} does not contain valid tokenizer files. Trying fallbacks.".format(student_path))
    
    logger.info("Student is a model ID ('{}'). Checking cache and fallbacks...".format(student_path))
    
    try:
        _ = AutoTokenizer.from_pretrained(
            student_path,
            cache_dir=model_args.cache_dir,
            local_files_only=True,
            use_fast=model_args.use_fast_tokenizer
        )
        logger.info("Found '{}' in local cache. Using cached version.".format(student_path))
        return student_path
    except (OSError, ValueError):
        logger.info("Model '{}' not found in local cache or incomplete.".format(student_path))
        
        if teacher_path and os.path.isdir(teacher_path):
            if is_valid_tokenizer_path(teacher_path):
                logger.info("Fallback: Using Small Teacher path as tokenizer: {}".format(teacher_path))
                return teacher_path
            else:
                logger.warning("Small Teacher path {} also lacks tokenizer files.".format(teacher_path))
        
        logger.warning("No local tokenizer found. Will attempt to download '{}' from internet.".format(student_path))
        return student_path

# ==============================================================================
# 3. 核心动态蒸馏模型类 (DynamicKDModel)
# ==============================================================================
class DynamicKDModel(PreTrainedModel):
    def __init__(self, config, student, small_teacher, large_teacher, args):
        super(DynamicKDModel, self).__init__(config)
        self.student = student
        self.small_teacher = small_teacher
        self.large_teacher = large_teacher
        
        logger.info("Freezing teacher parameters...")
        for param in self.small_teacher.parameters():
            param.requires_grad = False
        for param in self.large_teacher.parameters():
            param.requires_grad = False
            
        self.args = args
        self.temperature = args.temperature
        self.kd_alpha = args.kd_alpha
        self.ce_alpha = args.ce_alpha
        self.use_dynamic_teacher = args.dynamic_teacher_selection
        self.use_dynamic_obj = args.dynamic_objective_weighting
        
        self.num_student_layers = student.config.num_hidden_layers
        
        # Create separate layer mappings for both teachers
        self.layer_map_small = {}
        self.layer_map_large = {}
        
        layers_to_distill = args.student_layers_to_distill
        if layers_to_distill == "all":
            target_indices = list(range(self.num_student_layers))
        else:
            try:
                target_indices = [int(x.strip()) for x in layers_to_distill.split(',')]
            except ValueError:
                logger.warning("Invalid student_layers_to_distill format, defaulting to all layers.")
                target_indices = list(range(self.num_student_layers))
                
        for i in target_indices:
            if i >= self.num_student_layers:
                continue
            
            target_layer = i * 2
            
            # Map for Large Teacher
            if target_layer < large_teacher.config.num_hidden_layers:
                self.layer_map_large[i] = target_layer
            else:
                self.layer_map_large[i] = large_teacher.config.num_hidden_layers - 1
                
            # Map for Small Teacher
            if target_layer < small_teacher.config.num_hidden_layers:
                self.layer_map_small[i] = target_layer
            else:
                self.layer_map_small[i] = small_teacher.config.num_hidden_layers - 1
                
        logger.info("Layer Mapping (Student -> Small Teacher): {}".format(self.layer_map_small))
        logger.info("Layer Mapping (Student -> Large Teacher): {}".format(self.layer_map_large))
        
        self.hidden_proj_small = nn.ModuleDict()
        self.hidden_proj_large = nn.ModuleDict()
        
        student_hidden_size = student.config.hidden_size
        
        for s_idx in self.layer_map_large.keys():
            small_hidden_size = small_teacher.config.hidden_size
            if student_hidden_size != small_hidden_size:
                self.hidden_proj_small[str(s_idx)] = nn.Linear(small_hidden_size, student_hidden_size)
            else:
                self.hidden_proj_small[str(s_idx)] = nn.Identity()
                
            large_hidden_size = large_teacher.config.hidden_size
            if student_hidden_size != large_hidden_size:
                self.hidden_proj_large[str(s_idx)] = nn.Linear(large_hidden_size, student_hidden_size)
            else:
                self.hidden_proj_large[str(s_idx)] = nn.Identity()

    def forward(self, input_ids, attention_mask=None, label=None, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Step 1: Student Forward
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        student_logits = student_outputs.logits
        student_hidden_states = student_outputs.hidden_states
        
        # Step 2: Uncertainty (Entropy)
        probs = F.softmax(student_logits / self.temperature, dim=-1)
        eps = 1e-9
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        
        num_label = student_logits.size(-1)
        if num_label > 1:
            max_entropy = np.log(num_label)
        else:
            max_entropy = 1.0
        norm_entropy = entropy / max_entropy
        
        # Step 3: Dynamic Teacher Selection
        small_teacher_indices = []
        large_teacher_indices = []
        
        if self.use_dynamic_teacher:
            threshold = torch.median(norm_entropy).item()
            hard_mask = norm_entropy > threshold
            small_teacher_indices = hard_mask.nonzero(as_tuple=True)[0].tolist()
            large_teacher_indices = (~hard_mask).nonzero(as_tuple=True)[0].tolist()
        else:
            large_teacher_indices = list(range(batch_size))
            
        # Step 4: Teacher Forward & Loss Calculation
        total_kl_loss = 0.0
        total_pt_loss = 0.0
        count_samples = 0
        
        # --- Small Teacher (Hard Samples) ---
        if len(small_teacher_indices) > 0:
            idx_tensor = torch.tensor(small_teacher_indices, device=device)
            sub_input_ids = input_ids[idx_tensor]
            sub_attn_mask = attention_mask[idx_tensor] if attention_mask is not None else None
            
            sub_student_logits = student_logits[idx_tensor]
            sub_student_hidden = tuple(h[idx_tensor] for h in student_hidden_states)
            
            with torch.no_grad():
                small_t_outputs = self.small_teacher(
                    input_ids=sub_input_ids,
                    attention_mask=sub_attn_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                small_t_logits = small_t_outputs.logits
                small_t_hidden = small_t_outputs.hidden_states
            
            kl_loss = F.kl_div(
                F.log_softmax(sub_student_logits / self.temperature, dim=-1),
                F.softmax(small_t_logits / self.temperature, dim=-1),
                reduction='sum'
            ) * (self.temperature ** 2)
            
            pt_loss = 0.0
            for s_idx, t_idx in self.layer_map_small.items():
                s_h = sub_student_hidden[s_idx + 1]
                t_h = small_t_hidden[t_idx + 1]
                proj = self.hidden_proj_small[str(s_idx)]
                
                t_h_proj = proj(t_h)
                pt_loss += F.mse_loss(s_h, t_h_proj)
            
            total_kl_loss += kl_loss
            total_pt_loss += pt_loss
            count_samples += len(small_teacher_indices)
            
        # --- Large Teacher (Easy Samples) ---
        if len(large_teacher_indices) > 0:
            idx_tensor = torch.tensor(large_teacher_indices, device=device)
            sub_input_ids = input_ids[idx_tensor]
            sub_attn_mask = attention_mask[idx_tensor] if attention_mask is not None else None
            
            sub_student_logits = student_logits[idx_tensor]
            sub_student_hidden = tuple(h[idx_tensor] for h in student_hidden_states)
            
            with torch.no_grad():
                large_t_outputs = self.large_teacher(
                    input_ids=sub_input_ids,
                    attention_mask=sub_attn_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                large_t_logits = large_t_outputs.logits
                large_t_hidden = large_t_outputs.hidden_states
            
            kl_loss = F.kl_div(
                F.log_softmax(sub_student_logits / self.temperature, dim=-1),
                F.softmax(large_t_logits / self.temperature, dim=-1),
                reduction='sum'
            ) * (self.temperature ** 2)
            
            pt_loss = 0.0
            for s_idx, t_idx in self.layer_map_large.items():
                s_h = sub_student_hidden[s_idx + 1]
                t_h = large_t_hidden[t_idx + 1]
                proj = self.hidden_proj_large[str(s_idx)]
                
                t_h_proj = proj(t_h)
                pt_loss += F.mse_loss(s_h, t_h_proj)
            
            total_kl_loss += kl_loss
            total_pt_loss += pt_loss
            count_samples += len(large_teacher_indices)
            
        if count_samples > 0:
            loss_kl = total_kl_loss / count_samples
            loss_pt = total_pt_loss / count_samples
        else:
            loss_kl = torch.tensor(0.0, device=device)
            loss_pt = torch.tensor(0.0, device=device)
            
        # Step 5: Dynamic Objective Weighting
        avg_norm_entropy = norm_entropy.mean()
        
        if self.use_dynamic_obj:
            w_kl = self.kd_alpha * (1.0 - avg_norm_entropy)
            w_pt = self.kd_alpha * avg_norm_entropy
            w_kl = max(w_kl, 0.1 * self.kd_alpha)
            w_pt = max(w_pt, 0.1 * self.kd_alpha)
        else:
            w_kl = self.kd_alpha
            w_pt = self.kd_alpha * 0.5
            
        distill_loss = w_kl * loss_kl + w_pt * loss_pt
        
        # Step 6: CE Loss & Total Loss
        loss_ce = torch.tensor(0.0, device=device)
        if label is not None:
            problem_type = getattr(self.config, 'problem_type', None)
            if problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.config.num_label == 1:
                    loss_ce = loss_fct(student_logits.squeeze(), label.squeeze())
                else:
                    loss_ce = loss_fct(student_logits, label)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss_ce = loss_fct(student_logits.view(-1, self.config.num_label), label.view(-1))
        
        total_loss = self.ce_alpha * loss_ce + distill_loss
        
        return {
            "loss": total_loss,
            "logits": student_logits,
            "loss_ce": loss_ce.detach(),
            "loss_kl": distill_loss.detach(),
            "loss_kl_raw": loss_kl.detach(),
            "loss_pt_raw": loss_pt.detach(),
            "entropy": avg_norm_entropy.detach(),
            "w_kl": torch.tensor(w_kl).detach(),
            "w_pt": torch.tensor(w_pt).detach()
        }

# ==============================================================================
# 4. 自定义 Trainer
# ==============================================================================
class DynamicKDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label = inputs.pop("label") if "label" in inputs else None
        outputs = model(**inputs, label=label)
        loss = outputs["loss"]
        
        # Only log these metrics during the training phase to avoid evaluation crashes
        if self.model.training:
            logs = {
                "ce_loss": outputs["loss_ce"].item(),
                "kd_loss_total": outputs["loss_kl"].item(),
                "kl_loss_raw": outputs["loss_kl_raw"].item(),
                "pt_loss_raw": outputs["loss_pt_raw"].item(),
                "avg_entropy": outputs["entropy"].item(),
                "weight_kl": outputs["w_kl"].item(),
                "weight_pt": outputs["w_pt"].item()
            }
            self.log(logs)
            
        # Strip out 0D scalars during evaluation so PyTorch's concatenation doesn't crash
        if return_outputs:
            clean_outputs = {
                "loss": outputs["loss"],
                "logits": outputs["logits"]
            }
            return (loss, clean_outputs)
            
        return loss

# ==============================================================================
# 5. 主函数
# ==============================================================================
def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # ----------------------------------------------------------------------
    # Setup logging
    # ----------------------------------------------------------------------
    import logging as py_logging
    
    log_level_str = getattr(training_args, 'logging_level', 'info')
    if isinstance(log_level_str, int):
        numeric_level = log_level_str
    else:
        level_map = {
            'debug': py_logging.DEBUG,
            'info': py_logging.INFO,
            'warning': py_logging.WARNING,
            'error': py_logging.ERROR,
            'critical': py_logging.CRITICAL
        }
        numeric_level = level_map.get(str(log_level_str).lower(), py_logging.INFO)
        
    py_logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[py_logging.StreamHandler(sys.stdout)],
        level=numeric_level
    )
    
    hf_logging.set_verbosity(numeric_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    
    if not is_main_process(training_args.local_rank):
        logger.setLevel(logging.WARN)
        
    logger.warning(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            training_args.local_rank, training_args.device, training_args.n_gpu,
            bool(training_args.local_rank != -1), training_args.fp16
        )
    )
    logger.info("Training/evaluation parameters {}".format(training_args))
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.".format(training_args.output_dir)
            )
        elif last_checkpoint is not None:
            logger.info(
                "Checkpoint detected, resuming training at {}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.".format(last_checkpoint)
            )
            
    set_seed(training_args.seed)
    
    # --- DATA LOADING BLOCK ---
    is_regression = data_args.task_name == "stsb"
    label_list = []
    num_label = 0
    raw_datasets = None
    
    if data_args.task_name is not None:
        if data_args.task_name == "rte" and data_args.train_file is not None:
            logger.info("Applying special handling for LOCAL RTE dataset.")
            data_files = {
                "train": data_args.train_file,
                "validation": data_args.validation_file,
            }
            delimiter = "\t" if data_args.train_file.endswith(".tsv") else ","
            raw_datasets = load_dataset("csv", data_files=data_files, delimiter=delimiter)
            
            unique_label = raw_datasets["train"].unique("label")
            label_list = sorted([label for label in unique_label if label is not None])
            num_label = len(label_list)
            logger.info("Loaded local RTE dataset. label: {}".format(label_list))
        else:
            if data_args.train_file is None:
                raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
                if not is_regression:
                    label_list = raw_datasets["train"].features["label"].names
                    num_label = len(label_list)
                else:
                    num_label = 1
            else:
                data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
                delimiter = "\t" if data_args.train_file.endswith(".tsv") else ","
                raw_datasets = load_dataset("csv", data_files=data_files, delimiter=delimiter)
                unique_label = raw_datasets["train"].unique("label")
                label_list = sorted([label for label in unique_label if label is not None])
                num_label = len(label_list)
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        file_extension = data_args.train_file.split('.')[-1]
        loader_args = {"data_files": data_files}
        if file_extension == "tsv":
             loader_args["delimiter"] = "\t"
        elif file_extension == "csv":
             loader_args["delimiter"] = ","
        
        raw_datasets = load_dataset("csv" if file_extension in ["csv", "tsv"] else "json", **loader_args)
        
        if not is_regression:
            unique_label = raw_datasets["train"].unique("label")
            label_list = sorted([label for label in unique_label if label is not None])
            num_label = len(label_list)
        else:
            num_label = 1
            
    logger.info("Task: {}".format(data_args.task_name or 'custom'))
    logger.info("Number of label: {}".format(num_label))
    if label_list:
        logger.info("Label list: {}".format(label_list))
    
    # ==============================================
    # 🛡️ Sanitize Quotes
    # ==============================================
    logger.info("Checking and sanitizing unbalanced quotes in dataset...")
    sentence1_key, sentence2_key = task_to_keys.get(data_args.task_name, ("sentence1", "sentence2"))
    
    if data_args.task_name not in task_to_keys:
        if "sentence1" in raw_datasets["train"].column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif "text" in raw_datasets["train"].column_names:
            sentence1_key, sentence2_key = "text", None
        else:
            cols = raw_datasets["train"].column_names
            sentence1_key = cols[0]
            sentence2_key = cols[1] if len(cols) > 2 else None
            
    text_columns_to_check = [sentence1_key]
    if sentence2_key:
        text_columns_to_check.append(sentence2_key)
    
    logger.info("Columns to check for quote balance: {}".format(text_columns_to_check))
    
    raw_datasets = raw_datasets.map(
        lambda example: sanitize_quotes(example, text_columns_to_check),
        batched=False,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Sanitizing unbalanced quotes"
    )
    logger.info("Quote sanitization complete.")
    
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.student_name_or_path,
        num_labels=num_label,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.problem_type = "regression" if is_regression else "single_label_classification"
    
    final_tokenizer_path = resolve_tokenizer_path(model_args)
    logger.info("Resolved Tokenizer Path: {}".format(final_tokenizer_path))
    
    tokenizer = AutoTokenizer.from_pretrained(
        final_tokenizer_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    student = AutoModelForSequenceClassification.from_pretrained(
        model_args.student_name_or_path,
        from_tf=bool(".ckpt" in model_args.student_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    logger.info("Loading Small Teacher from: {}".format(model_args.small_teacher_name_or_path))
    small_teacher = AutoModelForSequenceClassification.from_pretrained(
        model_args.small_teacher_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    logger.info("Loading Large Teacher from: {}".format(model_args.large_teacher_name_or_path))
    large_teacher = AutoModelForSequenceClassification.from_pretrained(
        model_args.large_teacher_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    model = DynamicKDModel(config, student, small_teacher, large_teacher, model_args)
    
    # Preprocessing
    padding = "max_length" if data_args.pad_to_max_length else False
    
    # Label-to-ID mapping
    label_to_id = {v: i for i, v in enumerate(label_list)} if label_list else None
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "max_seq_length ({}) is larger than the model's max length ({}).".format(
                data_args.max_seq_length, tokenizer.model_max_length
            )
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    # -----------------------------------------------------------
    # 🛡️ PREPROCESS FUNCTION (Strictly copied from Script 1)
    # -----------------------------------------------------------
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        
        # 【完全复制第一段代码逻辑】
        if label_to_id is not None and "label" in examples:
             result["label"] = [(label_to_id.get(l, -1)) for l in examples["label"]]
        elif "label" in examples:
            # If no label_to_id (e.g., regression or numeric label directly), pass through
            if is_regression:
                result["label"] = [float(l) for l in examples["label"]]
            else:
                try:
                    result["label"] = [int(l) for l in examples["label"]]
                except (ValueError, TypeError):
                    logger.warning("Could not convert label to int automatically. Keeping original.")
                    result["label"] = examples["label"]
                    
        return result
        
    # -----------------------------------------------------------
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset"
    )
    
    # 🛑 CRITICAL FIX: Filter out invalid samples (label == -1)
    # This is necessary to make Script 1's logic work safely in this script.
    if training_args.do_train:
        original_train_size = len(processed_datasets["train"])
        processed_datasets["train"] = processed_datasets["train"].filter(
            lambda x: x["label"] != -1,
            desc="Filtering invalid label (None/Unknown)"
        )
        new_train_size = len(processed_datasets["train"])
        logger.info(f"✅ Training set filtered: Removed {original_train_size - new_train_size} invalid samples.")
    
    if training_args.do_eval:
        original_eval_size = len(processed_datasets["validation"])
        processed_datasets["validation"] = processed_datasets["validation"].filter(
            lambda x: x["label"] != -1,
            desc="Filtering invalid label (None/Unknown)"
        )
        new_eval_size = len(processed_datasets["validation"])
        if original_eval_size != new_eval_size:
            logger.warning(f"⚠️  Validation set filtered: Removed {original_eval_size - new_eval_size} invalid samples.")
            
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info("Sample {} of the training set: {}.".format(index, train_dataset[index]))
            
    # Metric function
    metric = load_metric("glue", data_args.task_name) if data_args.task_name in glue_tasks else None
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        
        if metric is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
            
    # Data collator
    data_collator = (
        default_data_collator if data_args.pad_to_max_length else
        DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else
        DataCollatorWithPadding(tokenizer)
    )
    
    # Initialize Trainer
    trainer = DynamicKDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
