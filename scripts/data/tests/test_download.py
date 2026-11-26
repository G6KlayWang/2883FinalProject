import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'download_pusht.py'


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'Download script has not been implemented')
        spec = importlib.util.spec_from_file_location('download_pusht', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.zst'
        self.path.write_bytes(b'original dataset')
        self.digest = hashlib.sha256(b'original dataset').hexdigest()

    def test_accepts_matching_file(self):
        self.assertEqual(self.module.verify_file(self.path, 16, self.digest), self.digest)

    def test_rejects_truncated_download(self):
        self.path.write_bytes(b'original')
        with self.assertRaisesRegex(ValueError, 'size'):
            self.module.verify_file(self.path, 16, self.digest)

    def test_rejects_same_size_corruption(self):
        self.path.write_bytes(b'corrupt! dataset')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.module.verify_file(self.path, 16, self.digest)


class ScriptCompatibilityTests(unittest.TestCase):
    def test_direct_script_help_works_outside_repository(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(SCRIPT), '--help'], cwd=folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--destination', result.stdout)


if __name__ == '__main__':
    unittest.main()
