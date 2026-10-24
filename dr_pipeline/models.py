from __future__ import annotations

from typing import Any

from model.fgsanet import FGSANet


def create_model(
    name: str,
    num_classes: int,
    image_size: int = 384,
    model_kwargs: dict[str, Any] | None = None,
):
    """Create FGSANet or one of its two-module ablation variants."""
    del image_size  # Kept for compatibility with the training/evaluation API.
    key = name.lower()
    variants = {"ours", "fgsanet", "ours_base", "ours_dfga", "ours_sagi"}
    if key not in variants:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(sorted(variants))}.")

    kwargs = dict(model_kwargs or {})
    if key == "ours_base":
        kwargs.setdefault("use_dfga", False)
        kwargs.setdefault("use_sagi", False)
    elif key == "ours_dfga":
        kwargs.setdefault("use_dfga", True)
        kwargs.setdefault("use_sagi", False)
    elif key == "ours_sagi":
        kwargs.setdefault("use_dfga", False)
        kwargs.setdefault("use_sagi", True)
    return FGSANet(num_classes=num_classes, **kwargs)


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def forward_model(model, images, labels=None):
    if labels is None:
        return model(images)
    try:
        return model(images, labels=labels)
    except TypeError:
        return model(images)


def parse_model_kwargs(text: str | None) -> dict[str, Any]:
    import json

    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("--model-kwargs must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("--model-kwargs must be a JSON object")
    return value
