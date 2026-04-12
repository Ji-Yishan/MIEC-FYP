import os
import random
from typing import Any, Dict, Optional, Tuple

import torch


def configure_mps_for_mac(ram_limit_gb: int = 48) -> None:
    """
    Configure conservative MPS behavior for Apple Silicon.
    This is best-effort and keeps defaults if user already set env vars.
    """
    # Fallback to CPU kernels when an op is unsupported on MPS.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Keep unified-memory pressure lower to reduce OOM risk.
    # 0.75 is a conservative cap suitable for <=48GB on 64GB systems.
    high_ratio = 0.8 if ram_limit_gb <= 48 else 0.9
    low_ratio = 0.7 if ram_limit_gb <= 48 else 0.8
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_ratio)
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(low_ratio)


def current_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_safe_training_defaults(training_args: Any, for_mps: bool = True) -> Dict[str, Any]:
    """
    Apply safe defaults to reduce memory spikes and modernize behavior.
    Returns a dict of applied values for logging.
    """
    applied: Dict[str, Any] = {}
    if not for_mps or not torch.backends.mps.is_available():
        return applied

    if hasattr(training_args, "use_mps_device"):
        training_args.use_mps_device = True
        applied["use_mps_device"] = True

    if hasattr(training_args, "fp16"):
        training_args.fp16 = False
        applied["fp16"] = False

    if hasattr(training_args, "bf16"):
        # Avoid backend mismatch surprises on MPS by default.
        training_args.bf16 = False
        applied["bf16"] = False

    if hasattr(training_args, "per_device_train_batch_size"):
        capped = min(int(training_args.per_device_train_batch_size), 8)
        training_args.per_device_train_batch_size = max(1, capped)
        applied["per_device_train_batch_size"] = training_args.per_device_train_batch_size

    if hasattr(training_args, "per_device_eval_batch_size"):
        capped = min(int(training_args.per_device_eval_batch_size), 16)
        training_args.per_device_eval_batch_size = max(1, capped)
        applied["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size

    if hasattr(training_args, "dataloader_num_workers"):
        workers = int(getattr(training_args, "dataloader_num_workers", 0))
        training_args.dataloader_num_workers = min(max(workers, 0), 4)
        applied["dataloader_num_workers"] = training_args.dataloader_num_workers

    if hasattr(training_args, "gradient_checkpointing"):
        # Keep disabled by default for custom wrapper models that may not implement
        # `gradient_checkpointing_enable`.
        training_args.gradient_checkpointing = False
        applied["gradient_checkpointing"] = False

    return applied


def get_label_from_inputs(inputs: Dict[str, Any]) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Data collators may produce either `label` or `labels`.
    """
    label = None
    if "label" in inputs:
        label = inputs.pop("label")
    elif "labels" in inputs:
        label = inputs.pop("labels")
    return label, inputs


def safe_random_indices(total: int, max_items: int = 3) -> list[int]:
    if total <= 0:
        return []
    n = min(max_items, total)
    return random.sample(range(total), n)


def load_glue_metric(task_name: Optional[str]):
    """
    Python 3.12-safe metric loader.
    Returns None if evaluate is unavailable.
    """
    if not task_name:
        return None
    try:
        import evaluate

        return evaluate.load("glue", task_name)
    except Exception:
        return None
