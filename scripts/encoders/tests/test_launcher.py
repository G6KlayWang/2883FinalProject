"""Recipe approval is enforced in the real candidate/repeat launch path."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from scripts.encoders.artifacts import identity_digest
from scripts.encoders.config import shared_training_controls
from scripts.encoders.recipe import load_approved_recipe
from scripts.encoders.selection import write_repeat_configs
from scripts.encoders.launcher import declarations


class LauncherTests(unittest.TestCase):
    def test_declared_candidates_resolve_from_the_approved_base_recipe(self):
        path = Path("scripts/encoders/configs/production-approved.json")
        config = load_approved_recipe(
            path, Path("scripts/encoders/configs/recipe-approval.json")
        )
        specs = declarations(config, path, "frozen", None, None)
        self.assertEqual(len(specs), 10)
        self.assertEqual((specs[0]["run_id"], specs[0]["rho"]), ("A-rho2.5-seed0", 2.5))
        self.assertEqual(
            specs[0]["config_hash"], replace(config, arm="A", rho=2.5).digest()
        )
        self.assertEqual(specs[3]["run_id"], "Aprime-rho0.210526-seed0")

    def test_repeat_launch_uses_selected_weights_but_refuses_changed_shared_controls(
        self,
    ):
        path = Path("scripts/encoders/configs/production-approved.json")
        config = load_approved_recipe(
            path, Path("scripts/encoders/configs/recipe-approval.json")
        )
        receipt = {
            "status": "approved-policy",
            "selected": {
                arm: {
                    "arm": arm,
                    "rho": rho,
                    "seed": 0,
                    "checkpoint_role": "final",
                    "score_split": "validation",
                    "protocol": {
                        "shared_training_controls": shared_training_controls(config)
                    },
                }
                for arm, rho in (("A", 2.5), ("B", 10.0), ("C", 4.0))
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "repeats"
            receipt = write_repeat_configs(receipt, config, directory)
            selection = root / "selection.json"
            selection.write_text(json.dumps(receipt))
            specs = declarations(config, path, "frozen", directory, selection)
            self.assertEqual(
                [(s["arm"], s["rho"], s["seed"]) for s in specs],
                [("A", 2.5, 1), ("B", 10.0, 1), ("C", 4.0, 1)],
            )
            altered = directory / "A-selected-seed1.json"
            values = json.loads(altered.read_text())
            values["runtime"]["workers"] = 1
            altered.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "shared recipe"):
                declarations(config, path, "frozen", directory, selection)
