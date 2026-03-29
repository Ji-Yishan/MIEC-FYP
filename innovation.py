"""
Proposed Tri-LossKD method implementation, adding innovation base on the baseline.py
uses dynamic weighing strategy to calculate loss base on ce, kl, rep and atten losses
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from runtime_compat import (
    apply_safe_training_defaults,
    configure_mps_for_mac,
    load_glue_metric,
    safe_random_indices,
)

logger = logging.getLogger(__name__)


class DynamicObjectiveKDForSequenceClassification(BertForSequenceClassification):
    def __init__(
        self,
        config,
        kd_kl_alpha=1.0,
        kd_rep_alpha=1.0,
        attn_alpha=0.0,
        ce_alpha=1.0,
        teacher=None,
        temperature=5.0,
        strategy="none",
        kl_kd=False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher = teacher
        self.kd_kl_alpha = kd_kl_alpha
        self.kd_rep_alpha = kd_rep_alpha
        self.attn_alpha = attn_alpha
        self.ce_alpha = ce_alpha
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.strategy = strategy
        self.ds_weight = 1.0
        self.pt_weight = 1.0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        eps = 1e-9
        teacher_logits = None
        weight = None
        rep_loss = 0.0
        attn_loss = 0.0

        if self.training:
            assert self.teacher is not None
            need_attentions = self.attn_alpha > 0
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=True if need_attentions else output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=True if need_attentions else output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
                teacher_logits = teacher_outputs[0]

            teacher_reps = teacher_outputs["hidden_states"]
            student_reps = student_outputs["hidden_states"]
            layers_per_block = self.teacher.config.num_hidden_layers // self.config.num_hidden_layers
            new_teacher_reps = [
                teacher_reps[i * layers_per_block] for i in range(self.config.num_hidden_layers + 1)
            ]
            new_student_reps = student_reps

            if (
                self.attn_alpha > 0
                and teacher_outputs["attentions"] is not None
                and student_outputs["attentions"] is not None
            ):
                teacher_attns = teacher_outputs["attentions"]
                student_attns = student_outputs["attentions"]
                if self.strategy == "none":
                    attn_loss = 0.0
                    for s_attn, t_attn in zip(student_attns, teacher_attns[::layers_per_block]):
                        attn_loss += F.mse_loss(s_attn.mean(dim=1), t_attn.mean(dim=1))
                    attn_loss = attn_loss / max(1, len(student_attns))
                elif "uncertainty" in self.strategy:
                    attn_loss = 0.0
                    for s_attn, t_attn in zip(student_attns, teacher_attns[::layers_per_block]):
                        attn_loss = attn_loss + F.mse_loss(
                            s_attn.mean(dim=1),
                            t_attn.mean(dim=1),
                            reduction="none",
                        ).mean(dim=(1, 2))
                    attn_loss = attn_loss / max(1, len(student_attns))

            if self.kd_rep_alpha > 0 and self.strategy == "none":
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    rep_loss += F.mse_loss(
                        F.normalize(student_rep, p=2, dim=1),
                        F.normalize(teacher_rep, p=2, dim=1),
                    )
            elif (self.kd_rep_alpha > 0 or self.attn_alpha > 0) and "uncertainty" in self.strategy:
                new_student_reps_t = torch.stack(new_student_reps, dim=1)
                new_teacher_reps_t = torch.stack(new_teacher_reps, dim=1)
                rep_loss = (
                    F.mse_loss(
                        F.normalize(new_student_reps_t, p=2, dim=-1),
                        F.normalize(new_teacher_reps_t, p=2, dim=-1),
                        reduction="none",
                    )
                    .mean(dim=-1)
                    .mean(dim=-1)
                    .sum(dim=1)
                )
                probs = F.softmax(student_logits, dim=-1)
                entropy = torch.sum(probs * torch.log(probs + eps), dim=1)
                avg_prob = 1 / self.num_labels * torch.ones((1, self.num_labels), device=student_logits.device)
                weight = entropy / torch.sum(avg_prob * torch.log(avg_prob + eps))
        else:
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = self.ce_alpha * loss_fct(
                    student_logits.view(-1, self.num_labels), labels.view(-1)
                )

            if teacher_logits is not None:
                if weight is not None and self.strategy == "uncertainty":
                    if not self.kl_kd:
                        kd_loss = F.mse_loss(student_logits, teacher_logits, reduction="none").mean(dim=-1)
                    else:
                        kd_loss = (
                            self.temperature**2
                            * F.kl_div(
                                F.log_softmax(student_logits / self.temperature, dim=1),
                                F.softmax(teacher_logits / self.temperature, dim=-1),
                                reduction="none",
                            ).sum(dim=1)
                        )
                    lam = weight.detach()
                    loss += self.kd_kl_alpha * torch.mean((1 - weight) * kd_loss, dim=0)
                    loss += self.kd_rep_alpha * torch.mean(lam*weight * rep_loss, dim=0)
                    if self.attn_alpha > 0:
                        loss += self.attn_alpha * torch.mean((1-lam)*weight * attn_loss, dim=0)
                    self.ds_weight = torch.mean(1 - weight).item()
                    self.pt_weight = torch.mean(weight).item()
                elif weight is not None and self.strategy == "uncertainty-r":
                    if not self.kl_kd:
                        kd_loss = F.mse_loss(student_logits, teacher_logits, reduction="none").mean(dim=-1)
                    else:
                        kd_loss = (
                            self.temperature**2
                            * F.kl_div(
                                F.log_softmax(student_logits / self.temperature, dim=1),
                                F.softmax(teacher_logits / self.temperature, dim=-1),
                                reduction="none",
                            ).sum(dim=1)
                        )
                    loss += self.kd_kl_alpha * torch.mean(weight * kd_loss, dim=0)
                    loss += self.kd_rep_alpha * torch.mean((1 - weight) * rep_loss, dim=0)
                    if self.attn_alpha > 0:
                        loss += self.attn_alpha * torch.mean((1 - weight) * attn_loss, dim=0)
                else:
                    if not self.kl_kd:
                        kd_loss = self.mse_loss(student_logits.view(-1), teacher_logits.view(-1))
                    else:
                        kd_loss = (
                            self.kl_loss(
                                F.log_softmax(student_logits / self.temperature, dim=1),
                                F.softmax(teacher_logits / self.temperature, dim=-1),
                            )
                            * self.temperature**2
                        )
                    loss += self.kd_kl_alpha * kd_loss
                    loss += self.kd_rep_alpha * rep_loss
                    if self.attn_alpha > 0:
                        loss += self.attn_alpha * attn_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output


class ObjectiveWeightLoggingCallback(TrainerCallback):
    def __init__(self, model):
        self._model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        core = self._model.module if hasattr(self._model, "module") else self._model
        if hasattr(core, "ds_weight") and hasattr(core, "pt_weight"):
            logs["ds_weight"] = core.ds_weight
            logs["pt_weight"] = core.pt_weight


glue_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
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
    "imdb": ("text", None),
    "boolq": ("passage", "question"),
    "sst5": ("sentence", None),
}


def load_local_rte_fallback(cache_dir):
    repo_root = Path(__file__).resolve().parent
    rte_dir = repo_root / "RTE"
    train_path = rte_dir / "train_fix.tsv"
    if not train_path.exists():
        train_path = rte_dir / "train.tsv"
    dev_path = rte_dir / "dev.tsv"
    if not train_path.exists() or not dev_path.exists():
        return None
    logger.warning("GLUE RTE fallback: loading %s and %s", train_path, dev_path)
    return load_dataset(
        "csv",
        data_files={"train": str(train_path), "validation": str(dev_path)},
        delimiter="\t",
        cache_dir=cache_dir,
    )


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Task name: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(default=128, metadata={"help": "Max sequence length."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite dataset cache."})
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys:
                raise ValueError("Unknown task: " + self.task_name)
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Provide task_name or train_file + validation_file.")
        else:
            ext = self.train_file.split(".")[-1]
            assert ext in ("csv", "json", "tsv"), "train_file must be .csv, .tsv, or .json"
            assert self.validation_file.split(".")[-1] == ext, "validation_file must match train extension"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Student checkpoint or HF id (BERT)."})
    teacher: str = field(metadata={"help": "Teacher checkpoint or HF id."})
    kd_kl_alpha: float = field(default=1.0)
    kd_rep_alpha: float = field(default=1.0)
    attn_alpha: float = field(default=0.0)
    ce_alpha: float = field(default=1.0)
    student_num_layers: Optional[int] = field(default=None)
    kl_kd: bool = field(default=False)
    temperature: float = field(default=5.0)
    objective_strategy: str = field(default="none")
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

    def __post_init__(self):
        self.objective_strategy = self.objective_strategy.lower()
        if self.objective_strategy not in {"none", "uncertainty"}:
            raise ValueError("--objective_strategy must be 'none' or 'uncertainty'")


def _token_kw(model_args):
    return True if model_args.use_auth_token else None


def _parse_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def _configure_training(training_args):
    applied = apply_safe_training_defaults(training_args, for_mps=True)
    if getattr(training_args, "label_names", None) is None:
        training_args.label_names = ["labels"]

    if training_args.do_train and training_args.do_eval:
        if hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "epoch"
        if hasattr(training_args, "evaluation_strategy"):
            training_args.evaluation_strategy = "epoch"

    if training_args.report_to and "tensorboard" in training_args.report_to:
        try:
            import tensorboard  # noqa: F401
        except Exception:
            logger.warning("TensorBoard unavailable; disabling report_to.")
            training_args.report_to = []
    return applied


def _configure_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _resolve_last_checkpoint(training_args):
    last_checkpoint = None
    overwrite_od = getattr(training_args, "overwrite_output_dir", False)
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not overwrite_od:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output dir {training_args.output_dir} not empty. Use --overwrite_output_dir or new --output_dir."
            )
        if last_checkpoint is not None:
            logger.info("Resuming from %s", last_checkpoint)
    return last_checkpoint


def _load_datasets(model_args, data_args, training_args):
    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        try:
            return load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
        except TypeError as e:
            if data_args.task_name == "rte" and "NoneType" in str(e):
                local = load_local_rte_fallback(model_args.cache_dir)
                if local is None:
                    raise RuntimeError("GLUE RTE failed and no ./RTE TSV fallback.") from e
                return local
            raise

    if data_args.task_name is not None and data_args.task_name in task_to_keys and data_args.task_name != "sst5":
        return load_dataset(data_args.task_name, cache_dir=model_args.cache_dir)

    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    if training_args.do_predict:
        if data_args.test_file is None:
            raise ValueError("do_predict needs --test_file for local data.")
        assert data_args.test_file.split(".")[-1] == data_args.train_file.split(".")[-1]
        data_files["test"] = data_args.test_file

    ext = data_args.train_file.split(".")[-1]
    if ext == "tsv":
        return load_dataset("csv", data_files=data_files, delimiter="\t", cache_dir=model_args.cache_dir)
    if ext == "csv":
        return load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
    return load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)


def _infer_label_info(datasets, data_args):
    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        is_regression = data_args.task_name == "stsb"
        if is_regression:
            return is_regression, [], 1
        label_list = list(datasets["train"].features["label"].names)
        return is_regression, label_list, len(label_list)

    label_feature = datasets["train"].features["label"]
    dtype = getattr(label_feature, "dtype", None)
    dtype_str = str(dtype) if dtype is not None else ""
    is_regression = dtype_str in ("float32", "float64") or "float" in dtype_str

    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_list = list(label_feature.names)
        return False, label_list, len(label_list)
    if is_regression:
        return True, [], 1
    if data_args.task_name == "boolq":
        label_list = ["False", "True"]
        return False, label_list, 2

    label_list = sorted([str(x) for x in datasets["train"].unique("label") if x is not None])
    return False, label_list, len(label_list)


def _build_tokenizer(model_args, tok):
    return AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=tok,
    )


def _build_model(model_args, data_args, num_labels, tok):
    student_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=tok,
    )
    if model_args.student_num_layers is not None:
        student_config.num_hidden_layers = int(model_args.student_num_layers)
    if model_args.attn_alpha > 0:
        student_config._attn_implementation = "eager"

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=tok,
    )
    if model_args.attn_alpha > 0:
        teacher_config._attn_implementation = "eager"

    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.teacher,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=tok,
    )
    model = DynamicObjectiveKDForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=student_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=tok,
        temperature=model_args.temperature,
        kl_kd=model_args.kl_kd,
        ce_alpha=model_args.ce_alpha,
        kd_kl_alpha=model_args.kd_kl_alpha,
        kd_rep_alpha=model_args.kd_rep_alpha,
        attn_alpha=model_args.attn_alpha,
        strategy=model_args.objective_strategy,
    )
    model.teacher = teacher_model
    return model


def _sentence_keys(datasets, data_args):
    if data_args.task_name is not None:
        return task_to_keys[data_args.task_name]

    non_label_columns = [name for name in datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_columns and "sentence2" in non_label_columns:
        return "sentence1", "sentence2"
    if len(non_label_columns) >= 2:
        return non_label_columns[0], non_label_columns[1]
    return non_label_columns[0], None


def _label_to_id(model, data_args, label_list, num_labels, is_regression):
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
        and label_list
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        dataset_names = sorted(str(x).lower() for x in label_list)
        if sorted(label_name_to_id.keys()) == dataset_names:
            return {
                str(label_list[i]): int(label_name_to_id[str(label_list[i]).lower()])
                for i in range(num_labels)
            }
        logger.warning("Model label2id does not match dataset; ignoring config labels.")
        return None

    if (data_args.task_name is None and not is_regression and label_list) or data_args.task_name == "sst5":
        return {str(value): i for i, value in enumerate(label_list)}
    return None


def _preprocess_function(tokenizer, sentence1_key, sentence2_key, padding, max_seq_length, label_to_id, data_args):
    def preprocess_function(examples):
        args_tok = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args_tok, padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            mapped = []
            for label in examples["label"]:
                if label in label_to_id:
                    mapped.append(label_to_id[label])
                else:
                    mapped.append(label_to_id.get(str(label), -1))
            result["label"] = mapped
        if "answer" in examples and data_args.task_name not in glue_tasks:
            result["label"] = [1 if label else 0 for label in examples["answer"]]
        return result

    return preprocess_function


def _split_datasets(datasets, data_args, training_args):
    train_dataset = eval_dataset = test_dataset = None

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" in datasets or "validation_matched" in datasets:
            eval_key = "validation_matched" if data_args.task_name == "mnli" else "validation"
            eval_dataset = datasets[eval_key]
        else:
            eval_dataset = datasets["test"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("Need test split for do_predict")
        test_key = "test_matched" if data_args.task_name == "mnli" else "test"
        test_dataset = datasets[test_key]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    return train_dataset, eval_dataset, test_dataset


def _compute_metrics_fn(metric, is_regression):
    def compute_metrics(prediction: EvalPrediction):
        preds = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if metric is not None:
            out = metric.compute(predictions=preds, references=prediction.label_ids)
            if len(out) > 1:
                out["combined_score"] = float(np.mean(list(out.values())))
            return out
        if is_regression:
            return {"mse": float(((preds - prediction.label_ids) ** 2).mean())}
        return {"accuracy": float((preds == prediction.label_ids).mean())}

    return compute_metrics


def _data_collator(data_args, training_args, tokenizer):
    if data_args.pad_to_max_length:
        return default_data_collator
    if training_args.fp16:
        return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    return None


def _build_trainer(model, training_args, train_dataset, eval_dataset, compute_metrics, data_collator, callbacks, tokenizer):
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    return Trainer(**trainer_kwargs)


def main():
    configure_mps_for_mac(ram_limit_gb=48)
    model_args, data_args, training_args = _parse_args()
    applied = _configure_training(training_args)
    _configure_logging(training_args)
    last_checkpoint = _resolve_last_checkpoint(training_args)
    set_seed(training_args.seed)
    tok = _token_kw(model_args)
    datasets = _load_datasets(model_args, data_args, training_args)
    is_regression, label_list, num_labels = _infer_label_info(datasets, data_args)
    tokenizer = _build_tokenizer(model_args, tok)
    model = _build_model(model_args, data_args, num_labels, tok)
    callbacks = [ObjectiveWeightLoggingCallback(model)]
    sentence1_key, sentence2_key = _sentence_keys(datasets, data_args)
    padding = "max_length" if data_args.pad_to_max_length else False
    label_to_id = _label_to_id(model, data_args, label_list, num_labels, is_regression)
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    preprocess_function = _preprocess_function(
        tokenizer, sentence1_key, sentence2_key, padding, max_seq_length, label_to_id, data_args
    )
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    train_dataset, eval_dataset, test_dataset = _split_datasets(datasets, data_args, training_args)

    if training_args.do_train:
        for idx in safe_random_indices(len(train_dataset), 3):
            logger.info("Sample %s: %s", idx, train_dataset[idx])

    metric = load_glue_metric(data_args.task_name) if data_args.task_name in glue_tasks else None
    compute_metrics = _compute_metrics_fn(metric, is_regression)
    data_collator = _data_collator(data_args, training_args, tokenizer)
    trainer = _build_trainer(
        model,
        training_args,
        train_dataset,
        eval_dataset,
        compute_metrics,
        data_collator,
        callbacks,
        tokenizer,
    )

    if applied:
        logger.info("Applied runtime defaults: %s", applied)
    logger.info("dynamic_variant=dynamic_objective | %s", training_args)

    train_time = 0.0
    train_metrics = {}
    if training_args.do_train:
        import time as _time

        t0 = _time.time()
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        train_time = _time.time() - t0
        train_metrics = dict(train_result.metrics)
        train_metrics["train_samples"] = len(train_dataset)
        train_metrics["training_time_sec"] = round(train_time, 2)
        trainer.save_model()
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        tasks = [data_args.task_name]
        eval_sets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_sets.append(datasets["validation_mismatched"])
        for eval_split, task in zip(eval_sets, tasks):
            eval_metrics = trainer.evaluate(eval_dataset=eval_split)
            eval_metrics["eval_samples"] = len(eval_split)
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        tasks = [data_args.task_name]
        test_sets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_sets.append(datasets["test_mismatched"])
        for test_split, task in zip(test_sets, tasks):
            test_split = test_split.remove_columns(["label"])
            predictions = trainer.predict(test_dataset=test_split).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                os.makedirs(training_args.output_dir, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as handle:
                    handle.write("index\tprediction\n")
                    for i, item in enumerate(predictions):
                        if is_regression:
                            handle.write(f"{i}\t{item:3.3f}\n")
                        else:
                            handle.write(f"{i}\t{label_list[int(item)]}\n")

    if training_args.do_eval and eval_dataset is not None:
        summary = {
            "dynamic_variant": "dynamic_objective",
            "eval_accuracy": None,
            "eval_loss": None,
            "training_time_seconds": round(train_time, 2),
            "num_epochs": float(training_args.num_train_epochs),
            "train_loss": train_metrics.get("train_loss"),
        }
        results_path = os.path.join(training_args.output_dir, "myBaseLine_results.json")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Wrote %s", results_path)


if __name__ == "__main__":
    main()