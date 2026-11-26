import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.encoders.tests.test_losses import implementation


class RecipeApprovalTests(unittest.TestCase):
    def test_approved_recipe_keeps_every_reviewed_numeric_control(self):
        recipe = implementation("recipe")
        approved = recipe.load_approved_recipe(
            Path("scripts/encoders/configs/production-approved.json"),
            Path("scripts/encoders/configs/recipe-approval.json"),
        )
        self.assertEqual(approved.optimizer.max_updates, 30000)
        self.assertEqual(
            approved.digest(),
            "b85f4349360df6f6bd8259569de5d25f00ecd29d7e9a9e634c53a45d01343c0e",
        )
        from scripts.data.file_io import sha256_file
        from scripts.encoders.config import load_config

        reference = Path("scripts/encoders/configs/production-proposed.json")
        self.assertEqual(
            sha256_file(reference),
            "0809df674b7eee3735c20b44d36a4294e31c62048de1f3b583094c954eb24be4",
        )
        self.assertEqual(
            load_config(reference).digest(),
            "4863ece10ab1ec45ed582d4a035715991f06a4570b8ff1fb4c8da13949ff9833",
        )
        approved.require_execution_approval()
        with tempfile.TemporaryDirectory() as folder:
            changed = approved.to_dict()
            changed["optimizer"]["batch_size"] = 16
            path = Path(folder) / "changed.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "recipe|approved|values"):
                recipe.load_approved_recipe(
                    path, Path("scripts/encoders/configs/recipe-approval.json")
                )
