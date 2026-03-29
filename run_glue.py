"""
fine-tuning teacher model code, a updated to python 3.12 version of the reference paper's code
referece code: https://github.com/lancopku/DynamicKD
"""
import logging
import os
import random
import sys
import inspect
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
from datasets import load_dataset, Features, Value, ClassLabel
import torch

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
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from runtime_compat import (
    apply_safe_training_defaults,
    configure_mps_for_mac,
    current_device,
    load_glue_metric,
    safe_random_indices,
)


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

logger = logging.getLogger(__name__)

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
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

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
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def sanitize_quotes(example, text_columns):
    """
    sanitize dataset, removing unbalance quotations
    """
    for col in text_columns:
        if col in example and example[col] is not None:
            text = str(example[col])
            
            # 检查双引号
            if text.count('"') % 2 != 0:
                logger.warning(f"Detected unbalanced double quotes in column '{col}'. Removing all double quotes from this sample.")
                text = text.replace('"', '')
            
            # 检查单引号
            if text.count("'") % 2 != 0:
                logger.warning(f"Detected unbalanced single quotes in column '{col}'. Removing all single quotes from this sample.")
                text = text.replace("'", '')
            
            example[col] = text
            
    return example


def load_local_rte_from_repo_root():
    """Load local RTE TSV files from ./RTE if available."""
    repo_root = Path(__file__).resolve().parent
    rte_dir = repo_root / "RTE"
    train_path = rte_dir / "train_fix.tsv"
    if not train_path.exists():
        train_path = rte_dir / "train.tsv"
    dev_path = rte_dir / "dev.tsv"

    if not train_path.exists() or not dev_path.exists():
        return None

    logger.warning(
        "Falling back to local RTE files: train=%s, validation=%s",
        str(train_path),
        str(dev_path),
    )
    return load_dataset(
        "csv",
        data_files={"train": str(train_path), "validation": str(dev_path)},
        delimiter="\t",
    )


def main():
    configure_mps_for_mac(ram_limit_gb=48)
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    applied_runtime = apply_safe_training_defaults(training_args, for_mps=True)

    # TensorBoard integration can fail in legacy Python/protobuf environments.
    # Fallback to no external reporting instead of crashing at Trainer init.
    if training_args.report_to and "tensorboard" in training_args.report_to:
        try:
            import tensorboard  # noqa: F401
        except Exception:
            logger.warning("TensorBoard is unavailable/incompatible; disabling `report_to=tensorboard`.")
            training_args.report_to = []

    if training_args.do_train and training_args.do_eval:
        if hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "epoch"
        if hasattr(training_args, "evaluation_strategy"):
            training_args.evaluation_strategy = "epoch"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    if applied_runtime:
        logger.info(f"Applied safe runtime defaults: {applied_runtime}")
    logger.info(f"Torch device: {current_device()}")

    # Set the verbosity of the Transformers logger
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Detect last checkpoint
    last_checkpoint = None
    overwrite_output_dir = getattr(training_args, "overwrite_output_dir", False)
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # --- START OF MODIFIED DATA LOADING BLOCK ---
    is_regression = data_args.task_name == "stsb"
    label_list = []
    num_labels = 0

    if data_args.task_name is not None:
        logger.info(f"Loading dataset from GLUE benchmark: {data_args.task_name}")
        try:
            datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
        except TypeError as e:
            # Known issue in some datasets versions on Python 3.7 for GLUE parquet backend.
            if data_args.task_name == "rte" and "NoneType" in str(e):
                local_rte = load_local_rte_from_repo_root()
                if local_rte is not None:
                    datasets = local_rte
                else:
                    raise RuntimeError(
                        "GLUE RTE download failed with a known datasets bug and no local RTE files were found. "
                        "Place train_fix.tsv/train.tsv and dev.tsv under ./RTE, or pass --train_file/--validation_file."
                    ) from e
            else:
                raise
        if not is_regression:
            label_feature = datasets["train"].features["label"]
            if hasattr(label_feature, "names") and label_feature.names is not None:
                label_list = label_feature.names
            else:
                unique_labels = datasets["train"].unique("label")
                label_list = sorted([str(label) for label in unique_labels if label is not None])
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # load from local when task name is none
        logger.info("Loading dataset from local files.")
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        file_extension = data_args.train_file.split('.')[-1]
        loader_args = {"data_files": data_files}
        
        if file_extension == "tsv":
            loader_args["delimiter"] = "\t"

        datasets = load_dataset("csv" if file_extension in ["csv", "tsv"] else "json", **loader_args)
        
        if not is_regression:
            unique_labels = datasets["train"].unique("label")
            label_list = sorted([str(label) for label in unique_labels if label is not None])
            num_labels = len(label_list)
        else:
            num_labels = 1

    logger.info(f"Task: {data_args.task_name or 'custom'}")
    logger.info(f"Number of labels: {num_labels}")
    if label_list:
        logger.info(f"Label list: {label_list}")

    logger.info("Checking and sanitizing unbalanced quotes in dataset...")
    sentence1_key, sentence2_key = task_to_keys.get(data_args.task_name, ("sentence1", "sentence2"))
    if data_args.task_name not in task_to_keys:
        available_cols = datasets["train"].column_names
        if "sentence1" in available_cols:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif "text" in available_cols:
            sentence1_key, sentence2_key = "text", None
        else:
            sentence1_key = available_cols[0]
            sentence2_key = available_cols[1] if len(available_cols) > 2 else None
    text_columns_to_check = [sentence1_key]
    if sentence2_key:
        text_columns_to_check.append(sentence2_key)
    
    logger.info(f"Columns to check for quote balance: {text_columns_to_check}")
    
    # Apply sanitization to both train and validation sets
    datasets = datasets.map(
        lambda example: sanitize_quotes(example, text_columns_to_check),
        batched=False, # Process row by row for precise logging if needed, though batched is faster
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Sanitizing unbalanced quotes"
    )
    logger.info("Quote sanitization complete.")
    # ==============================================

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    # Note: sentence1_key, sentence2_key are already defined above for the quote check
    
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Label-to-ID mapping:
    # - GLUE labels are already ids; remapping them to label names causes -1 labels.
    # - Local/custom datasets may use string labels and need mapping.
    label_to_id = {v: i for i, v in enumerate(label_list)} if (label_list and data_args.task_name is None) else None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"max_seq_length ({data_args.max_seq_length}) is larger than the model's max length ({tokenizer.model_max_length})."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                result["label"] = [label_to_id.get(str(l), -1) for l in examples["label"]]
            elif is_regression:
                result["label"] = [float(l) for l in examples["label"]]
            else:
                result["label"] = [int(l) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    if training_args.do_train and "label" in train_dataset.column_names:
        train_dataset = train_dataset.filter(lambda x: x["label"] != -1, desc="Filtering invalid training labels")
    if training_args.do_eval and "label" in eval_dataset.column_names:
        eval_dataset = eval_dataset.filter(lambda x: x["label"] != -1, desc="Filtering invalid validation labels")

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    # Log a few random samples
    if training_args.do_train:
        for index in safe_random_indices(len(train_dataset), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = load_glue_metric(data_args.task_name) if data_args.task_name in glue_tasks else None

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
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset if training_args.do_train else None,
        "eval_dataset": eval_dataset if training_args.do_eval else None,
        "compute_metrics": compute_metrics,
        "data_collator": data_collator,
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

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

    # Prediction
    if training_args.do_predict:
        logger.info("*** Test (using train dataset) ***")
        test_dataset = train_dataset
        
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

        if 'label' in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns("label")

        predictions = trainer.predict(test_dataset=test_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {data_args.task_name} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        prediction_label = label_list[item]
                        writer.write(f"{index}\t{prediction_label}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()