"""Stateless CPU appearance views: fixed colour/blur/noise order, no geometry."""

import math
import torch
from torchvision.transforms import functional as TF
from scripts.encoders.rng import keyed_seed

ORDER = ["brightness", "contrast", "saturation", "hue", "blur", "noise"]


def validate_augmentation(config: dict) -> None:
    required = {
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "jitter_probability",
        "blur_probability",
        "blur_sigma",
        "noise_probability",
        "noise_std",
        "blur_kernel",
        "order",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(
            f"Augmentation config must explicitly contain {sorted(required)}"
        )
    for name in ("brightness", "contrast", "saturation", "hue", "noise_std"):
        value = config[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Augmentation {name} must be finite and nonnegative")
    if config["hue"] > 0.5:
        raise ValueError("Augmentation hue must be <=.5")
    for name in ("jitter_probability", "blur_probability", "noise_probability"):
        if not isinstance(config[name], (int, float)) or not 0 <= config[name] <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if config["order"] != ORDER:
        raise ValueError(f"Current explicit appearance order is {ORDER}")
    if (
        type(config["blur_kernel"]) is not int
        or config["blur_kernel"] < 3
        or config["blur_kernel"] % 2 != 1
    ):
        raise ValueError("Blur kernel must be an odd integer>=3")
    sigma = config["blur_sigma"]
    if (
        len(sigma) != 2
        or not 0 < sigma[0] <= sigma[1]
        or not all(math.isfinite(v) for v in sigma)
    ):
        raise ValueError("Blur sigma must be a finite positive ordered interval")


def appearance_view(
    image: torch.Tensor, config: dict, *, seed: int, key: tuple[int, ...]
) -> torch.Tensor:
    validate_augmentation(config)
    if (
        image.device.type != "cpu"
        or image.dtype != torch.float32
        or image.shape != (3, 96, 96)
    ):
        raise ValueError("Appearance input must be CPU float32(3,96,96)")
    generator = torch.Generator().manual_seed(keyed_seed(seed, "augmentation", *key))

    def uniform(low: float, high: float) -> float:
        return low + (high - low) * torch.rand((), generator=generator).item()

    result = image.clone()
    if uniform(0, 1) < config["jitter_probability"]:
        result = TF.adjust_brightness(
            result, uniform(max(0, 1 - config["brightness"]), 1 + config["brightness"])
        )
        result = TF.adjust_contrast(
            result, uniform(max(0, 1 - config["contrast"]), 1 + config["contrast"])
        )
        result = TF.adjust_saturation(
            result, uniform(max(0, 1 - config["saturation"]), 1 + config["saturation"])
        )
        result = TF.adjust_hue(result, uniform(-config["hue"], config["hue"]))
    if uniform(0, 1) < config["blur_probability"]:
        result = TF.gaussian_blur(
            result, config["blur_kernel"], uniform(*config["blur_sigma"])
        )
    if uniform(0, 1) < config["noise_probability"]:
        result = result + config["noise_std"] * torch.randn(
            result.shape, generator=generator
        )
    return result.clamp(0, 1).contiguous()


class AugmentedFrames:
    """DataLoader keys are (logical update, source index); prefetch changes no RNG."""

    def __init__(self, frames, config: dict, seed: int):
        validate_augmentation(config)
        self.frames, self.config, self.seed = frames, config, seed

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, key: tuple[int, int]) -> dict:
        update, index = key
        image = torch.from_numpy(self.frames[index])
        return {
            "view1": appearance_view(
                image, self.config, seed=self.seed, key=(update, index, 0)
            ),
            "view2": appearance_view(
                image, self.config, seed=self.seed, key=(update, index, 1)
            ),
            "index": index,
        }
