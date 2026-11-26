"""Shared, immutable resized uint8 image mmap cache with exact reviewed resize parity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from scripts.data.file_io import sha256_file, write_json_new
from scripts.data.loader import PreparedData
from scripts.encoders.artifacts import new_directory
from scripts.encoders.frames import ImageFrames


def build_image_cache(
    frames: ImageFrames, output: Path, *, reserve_gib: float = 40.0
) -> None:
    if not len(frames):
        raise ValueError("Cannot cache an empty frame population")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    required = len(frames) * (96 * 96 * 3 + 16) + int(reserve_gib * 1024**3)
    if shutil.disk_usage(output.parent).free < required:
        raise OSError("Insufficient disk space for image cache plus reserve")
    with new_directory(output) as temporary:
        images = np.lib.format.open_memmap(
            temporary / "images.npy",
            mode="w+",
            dtype="uint8",
            shape=(len(frames), 96, 96, 3),
        )
        for index in range(len(frames)):
            # Reviewed Pillow output is uint8 before /255. Round-trip is exact.
            images[index] = np.rint(frames[index].transpose(1, 2, 0) * 255).astype(
                np.uint8
            )
        images.flush()
        del images
        np.save(temporary / "ids.npy", frames.ids, allow_pickle=False)
        write_json_new(
            temporary / "manifest.json",
            {
                "cache_version": 1,
                "identity": frames.identity,
                "count": len(frames),
                "files": {
                    name: sha256_file(temporary / name)
                    for name in ("images.npy", "ids.npy")
                },
            },
        )


class CachedImages:
    """Parent verifies once; each worker lazily opens its own read-only mappings."""

    def __init__(self, path: Path, expected_identity: dict):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "manifest.json").read_text())
        if (
            self.metadata.get("cache_version") != 1
            or self.metadata.get("identity") != expected_identity
        ):
            raise ValueError("Image cache identity mismatch")
        if set(self.metadata.get("files", {})) != {"images.npy", "ids.npy"}:
            raise ValueError(
                "Image cache file hash manifest must bind images.npy and ids.npy"
            )
        for name, digest in self.metadata["files"].items():
            if sha256_file(self.path / name) != digest:
                raise ValueError(f"Image cache hash mismatch for {name}")
        self.identity = self.metadata["identity"]
        self._pid, self._images, self._ids = None, None, None
        self._open_maps()
        if (
            self._images.shape != (len(self), 96, 96, 3)
            or self._images.dtype != np.uint8
        ):
            raise ValueError("Unexpected image cache shape/dtype")
        if self._ids.shape != (len(self), 2) or self._ids.dtype != np.int64:
            raise ValueError("Unexpected image cache row mapping")
        if (
            hashlib.sha256(self._ids.astype("<i8").tobytes()).hexdigest()
            != expected_identity["row_digest"]
        ):
            raise ValueError(
                "Image cache row order differs from the expected population"
            )

    def _open_maps(self) -> None:
        if self._pid != os.getpid():
            self._images = np.load(
                self.path / "images.npy", mmap_mode="r", allow_pickle=False
            )
            self._ids = np.load(
                self.path / "ids.npy", mmap_mode="r", allow_pickle=False
            )
            self._pid = os.getpid()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.update(_pid=None, _images=None, _ids=None)
        return state

    @property
    def ids(self) -> np.ndarray:
        self._open_maps()
        return self._ids

    def __len__(self) -> int:
        return self.metadata["count"]

    def __getitem__(self, index: int) -> np.ndarray:
        self._open_maps()
        image = np.ascontiguousarray(
            self._images[index].transpose(2, 0, 1), dtype=np.float32
        )
        return image / np.float32(255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--policy", choices=("phase0", "raw"), required=True)
    args = parser.parse_args()
    prepared = PreparedData(args.hdf5, json.loads(args.manifest.read_text()))
    build_image_cache(ImageFrames(prepared, args.split, args.policy), args.output)


if __name__ == "__main__":
    main()
