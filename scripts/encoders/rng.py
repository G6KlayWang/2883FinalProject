"""Independent regularizer streams and portable logical-key seeds.

Initialization uses a keyed forked RNG scope in model.py. Data order,
augmentations and probes use logical-key CPU generators at each epoch/view;
only the two advancing regularizer streams need continuation states here.
"""

import hashlib
import torch

STREAM_NAMES = ("directions", "targets")


def keyed_seed(seed: int, *parts: object) -> int:
    message = ":".join(map(str, ("encoder-v1", seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") % (2**63 - 1)


class RandomStreams:
    def __init__(self, seed: int):
        self.generators = {
            name: torch.Generator(device="cpu").manual_seed(keyed_seed(seed, name))
            for name in STREAM_NAMES
        }

    def __getitem__(self, name: str) -> torch.Generator:
        return self.generators[name]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: generator.get_state().clone()
            for name, generator in self.generators.items()
        }

    def load_state_dict(self, states: dict[str, torch.Tensor]) -> None:
        if set(states) != set(STREAM_NAMES):
            raise ValueError("Checkpoint must include every named random stream")
        for name, state in states.items():
            self.generators[name].set_state(state.cpu())
