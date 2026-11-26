"""Clean evaluation-mode feature extraction, separate from disposable BN diagnostics."""

import copy
import hashlib
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
from scripts.data.file_io import sha256_file, write_json_new
from scripts.encoders.artifacts import new_directory
from scripts.encoders.feature_cache import FeatureCache
from scripts.encoders.measure import feature_statistics


def _batch(frames, start: int, size: int, device: str) -> torch.Tensor:
    images = np.stack([frames[i] for i in range(start, min(start + size, len(frames)))])
    return torch.from_numpy(images).to(device)


def _assert_unchanged(before: dict, model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if not torch.equal(value, before[name]):
            raise ValueError(
                f"Extraction mutated encoder parameter/BatchNorm buffer: {name}"
            )


def extract_features(
    model: nn.Module,
    frames,
    output: Path,
    checkpoint_identity: dict,
    *,
    device: str,
    batch_size: int = 256,
    reserve_gib: float = 40.0,
) -> None:
    if len(frames) < 1 or batch_size < 1:
        raise ValueError("Feature extraction requires images and a positive batch size")
    if frames.identity.get("split") not in ("train", "validation"):
        raise ValueError("Current extraction development cannot access test rows")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        shutil.disk_usage(output.parent).free
        < len(frames) * (128 * 4 + 16) + reserve_gib * 1024**3
    ):
        raise OSError("Insufficient disk reserve for clean features")
    model = model.to(device).eval()
    model.requires_grad_(False)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with new_directory(output) as temporary:
        features = np.lib.format.open_memmap(
            temporary / "features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(frames), 128),
        )
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                values = model(_batch(frames, start, batch_size, device))
                if values.shape != (min(batch_size, len(frames) - start), 128):
                    raise ValueError("Encoder output must be(B,128)")
                if values.dtype != torch.float32 or not torch.isfinite(values).all():
                    raise ValueError("Clean features must be finite float32")
                features[start : start + len(values)] = values.cpu().numpy()
        _assert_unchanged(before, model)
        features.flush()
        del features
        np.save(temporary / "ids.npy", frames.ids, allow_pickle=False)
        identity = {
            **checkpoint_identity,
            "frame_identity": frames.identity,
            "row_digest": hashlib.sha256(
                frames.ids.astype("<i8").tobytes()
            ).hexdigest(),
            "feature_space": (
                "projector+train-ZCA"
                if checkpoint_identity.get("effective_transform_sha256")
                else "projector-output"
            ),
            "dtype": "float32",
            "dimension": 128,
        }
        write_json_new(
            temporary / "manifest.json",
            {
                "version": 1,
                "count": len(frames),
                "identity": identity,
                "files": {
                    name: sha256_file(temporary / name)
                    for name in ("features.npy", "ids.npy")
                },
            },
        )


def train_mode_comparison(
    model: nn.Module, frames, *, batch_size: int, device: str
) -> dict:
    """Fixed first two full batches; train mode only touches independent copies."""
    count = min(2, len(frames) // batch_size) * batch_size
    if count < 2:
        raise ValueError(
            "Train-mode diagnostic needs at least one full batch of two images"
        )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    reports = {}
    for mode in ("eval", "train"):
        disposable = copy.deepcopy(model).to(device)
        disposable.train(mode == "train")
        collected = []
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                collected.append(
                    disposable(_batch(frames, start, batch_size, device)).cpu().numpy()
                )
        reports[mode] = feature_statistics(np.concatenate(collected))
        del disposable
    _assert_unchanged(before, model)
    return {
        "scope": "sampled-disposable-model-copy",
        "count": count,
        "batch_size": batch_size,
        "clean_eval_statistics": reports["eval"],
        "clean_train_mode_statistics": reports["train"],
    }


if __name__ == "__main__":
    from scripts.encoders.extraction_cli import main

    main()
