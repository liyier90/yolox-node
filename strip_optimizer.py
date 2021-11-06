"""Utility script to reduce model weights file size."""

from pathlib import Path

import torch


def strip_optimizer(weights_path: Path, half: bool = False):
    """Remove non-model items from the weights file.

    The inference Detector requires only "model" parameters from the weights
    file. Deletes all other values from the weights file. Optionally, converts
    all float and double parameters to half-precision. Leaves int parameters
    such as `num_batches_tracked` in `BatchNorm2d` untouched.

    Args:
        weights_path (`Path`): Path to weights file.
        half (`bool`): Flag to determine if float and double parameters should
            be converted to half-precision.
    """
    stripped_weights_path = weights_path.with_name(
        f"{weights_path.stem}-stripped"
        f"{'-half' if half else ''}{weights_path.suffix}"
    )
    orig_filesize = weights_path.stat().st_size / 1e6
    ckpt = torch.load(str(weights_path), map_location=torch.device("cpu"))
    # Remove all data other than "model", such as amp, optimizer, start_epoch
    delete_keys = [key for key in ckpt.keys() if key != "model"]
    for key in delete_keys:
        del ckpt[key]
    for param in ckpt["model"]:
        # Only convert double and float to half-precision
        if half and ckpt["model"][param].dtype in (torch.double, torch.float):
            ckpt["model"][param] = ckpt["model"][param].half()
        ckpt["model"][param].requires_grad = False

    torch.save(ckpt, str(stripped_weights_path))
    stripped_filesize = stripped_weights_path.stat().st_size / 1e6
    print(
        f"Saved as {stripped_weights_path}. "
        f"Original size: {orig_filesize}MB. "
        f"Stripped size: {stripped_filesize}MB."
    )


if __name__ == "__main__":
    weights_dir = Path(__file__).resolve().parents[1] / "peekingduck_weights" / "yolox"
    model_types = ["yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"]
    for model_type in model_types:
        for half_precision in (True, False):
            strip_optimizer(weights_dir / f"{model_type}.pth", half_precision)
