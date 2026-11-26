"""Import/version metadata self-check; no tensor forward, backward or update."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import torch

PINS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "numpy": "2.5.5",
    "pillow": "13.0.0",
    "h5py": "3.16.0",
    "hdf5plugin": "7.1.0",
    "zstandard": "0.25.0",
    "huggingface_hub": "0.36.0",
}


def environment_record() -> dict:
    return {
        "python": platform.python_version(),
        "torch_cuda_build": torch.version.cuda,
        "dependencies": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    for package, expected in PINS.items():
        module = "PIL" if package == "pillow" else package
        importlib.import_module(module)
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected:
            raise ValueError(f"{package}: expected {expected}, installed {actual}")
    report = environment_record()
    if args.require_cuda:
        if not torch.cuda.is_available() or torch.version.cuda != "12.8":
            raise ValueError("Expected CUDA12.8 PyTorch build and available GPU")
        report["visible_gpu_count"] = torch.cuda.device_count()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
