"""
mlflow_utils.py — small helpers to make MLflow logging painless & reproducible.

Covers:
- Logging test metrics safely (float-casting + JSON artifact)
- Building a small CPU input_example and signature
- Capturing exact Torch CUDA build in pip_requirements (incl. +cuXXX)
- Optional git commit tagging
- One-call PyTorch model logging (name=..., signature, pip reqs, registry)

Usage:
    from mlflow_utils import (
        log_test_metrics,
        tag_git_commit,
        log_pytorch_model,
        build_pip_requirements,  # if you want to customize
    )
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import mlflow
import numpy as np

def log_test_metrics(metrics: Dict[str, Any], prefix: str = "test_") -> None:
    """Log a dict of metrics with a prefix and also store the JSON artifact."""
    if not metrics:
        return
    for k, v in metrics.items():
        try:
            mlflow.log_metric(f"{prefix}{k.lower()}", float(v))
        except Exception:
            # Fall back to stringy values if necessary (rare)
            mlflow.set_tag(f"{prefix}{k.lower()}", str(v))
    mlflow.log_dict(metrics, artifact_file=f"{prefix}metrics.json")


def _small_batch_from_loader(loader) -> Any:
    """Pull the first batch from a loader in common formats (x, y), dict, or raw x."""
    try:
        batch = next(iter(loader))
    except Exception:
        return None

    # (x, y, ...) tuple/list
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        return batch[0]
    # dict batch
    if isinstance(batch, dict):
        # common keys; tweak if your dataset uses different names
        for key in ("input", "features", "inputs", "data", "x"):
            if key in batch:
                return batch[key]
        # nothing matched
        return None
    # raw tensor/array
    return batch


def build_example_and_signature(
    model,
    loader=None,
    fallback_shape: Sequence[int] = (2, 3, 224, 224),
    max_batch: int = 2,
    to_float32: bool = True,
):
    """
    Return (input_example_np, signature) for MLflow.
    - Runs a tiny forward pass on CPU with torch.inference_mode()
    - Restores model to its original device afterward
    - Works even if no loader is provided by using a synthetic fallback
    """
    import torch
    from mlflow.models import infer_signature

    model.eval()

    # Original device (if model has parameters)
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")

    # Build a small input tensor
    xb = None
    if loader is not None:
        xb = _small_batch_from_loader(loader)

    if xb is None:
        xb = torch.randn(*fallback_shape)

    # Clip batch size
    if hasattr(xb, "shape") and isinstance(xb.shape, (tuple, list)) and len(xb.shape) > 0:
        try:
            if xb.shape[0] > max_batch:
                xb = xb[:max_batch]
        except Exception:
            pass

    # Ensure CPU and dtype
    cpu = torch.device("cpu")
    xb = xb.to(cpu)
    if to_float32:
        try:
            xb = xb.float()
        except Exception:
            pass

    # Move model to CPU temporarily, forward pass under inference_mode
    model.to(cpu)
    with torch.inference_mode():
        yb = model(xb)
    # Restore device
    model.to(orig_device)

    # Robust numpy conversion
    x_np = xb.detach().cpu().contiguous().numpy()
    y_np = (
        yb.detach().cpu().contiguous().numpy()
        if hasattr(yb, "detach")
        else np.array(yb)
    )
    signature = infer_signature(x_np, y_np)
    return x_np, signature


def build_pip_requirements(
    include_torchvision: bool = False,
    include_torchaudio: bool = False,
) -> list:
    """
    Create a pip requirements list that preserves the local CUDA tag (e.g., +cu126).
    Adds an extra-index-url for the matching CUDA wheel if available.
    """
    import torch as _t

    reqs = ["mlflow>=2.14"]
    torch_ver = _t.__version__  # e.g., "2.8.0+cu126" or "2.8.0"
    if _t.cuda.is_available() and "+cu" in torch_ver:
        cu = (_t.version.cuda or "").replace(".", "")  # "126"
        if cu:
            reqs.insert(0, f"--extra-index-url https://download.pytorch.org/whl/cu{cu}")
    reqs.append(f"torch=={torch_ver}")

#    if include_torchvision:
#        try:
#            import torchvision
#            reqs.append(f"torchvision=={torchvision.__version__}")
#        except ImportError:
#            pass
#    if include_torchaudio:
#        try:
#            import torchaudio
#            reqs.append(f"torchaudio=={torchaudio.__version__}")
#        except ImportError:
#            pass

    return reqs




def log_pytorch_model(
    model,
    name: str = "model",
    loader=None,
    fallback_shape: Sequence[int] = (2, 3, 224, 224),
    pip_requirements: Optional[Sequence[str]] = None,
    registered_model_name: Optional[str] = None,
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    One call that:
      - builds input_example + signature on CPU
      - builds pip_requirements (if not supplied) preserving +cuXXX
      - logs the PyTorch model with MLflow using name=... (not artifact_path)
      - optionally registers to the Model Registry
      - sets a few helpful tags if provided
    """
    import torch  # for tag convenience

    # tags
    tags = {
        "model.flavor": "pytorch",
        "torch.version": getattr(torch, "__version__", None),
        "cuda.version": getattr(torch.version, "cuda", None),
    }
    if extra_tags:
        tags.update({k: v for k, v in extra_tags.items() if v is not None})
    mlflow.set_tags({k: v for k, v in tags.items() if v is not None})

    # example + signature
    input_example, signature = build_example_and_signature(
        model=model,
        loader=loader,
        fallback_shape=fallback_shape,
    )

    # deps
    if pip_requirements is None:
        pip_requirements = build_pip_requirements()

    # log
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=name,
        input_example=input_example,
        signature=signature,
        pip_requirements=list(pip_requirements),
        registered_model_name=registered_model_name,
    )
