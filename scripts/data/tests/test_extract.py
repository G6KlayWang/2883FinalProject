"""Extraction must never publish corrupt/partial files or replace user data."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import zstandard

import importlib


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path("scripts/data/extract_pusht.py").exists(), "Extraction not implemented")
        global extract_pusht
        extract_pusht = importlib.import_module("scripts.data.extract_pusht")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'fixture.h5'
        with h5py.File(source, 'w') as f:
            f['example'] = [1, 2, 3]
        self.payload = source.read_bytes()
        self.archive = self.root / 'fixture.zst'
        self.archive.write_bytes(zstandard.ZstdCompressor().compress(self.payload))
        self.target = self.root / 'prepared.h5'

    def extract(self):
        raw = self.archive.read_bytes()
        return extract_pusht.extract_archive(
            self.archive, self.target, len(raw), hashlib.sha256(raw).hexdigest(),
        )

    def test_complete_hdf5_published_with_receipt_and_source_preserved(self):
        archive_bytes = self.archive.read_bytes()
        receipt = self.extract()
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(receipt['hdf5_sha256'], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(receipt['hdf5_size_bytes'], len(self.payload))
        self.assertEqual(self.archive.read_bytes(), archive_bytes)
        self.assertEqual(list(self.root.glob('*.incomplete')), [])

    def test_hash_mismatch_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            extract_pusht.extract_archive(self.archive, self.target, self.archive.stat().st_size, '0' * 64)
        self.assertFalse(self.target.exists())

    def test_existing_output_never_replaced(self):
        self.target.write_bytes(b'user data')
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(self.target.read_bytes(), b'user data')

    def test_corrupt_truncated_frame_leaves_no_output(self):
        self.archive.write_bytes(self.archive.read_bytes()[:-8])
        with self.assertRaises((ValueError, zstandard.ZstdError)):
            self.extract()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob('*.incomplete')), [])

    def test_non_hdf5_payload_rejected_even_with_valid_hash(self):
        self.archive.write_bytes(zstandard.ZstdCompressor().compress(b'not hdf5'))
        with self.assertRaises((ValueError, OSError)):
            self.extract()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob('*.incomplete')), [])

    def test_insufficient_space_rejected_before_partial_write(self):
        with patch.object(extract_pusht.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()):
            with self.assertRaisesRegex(OSError, 'space'):
                self.extract()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob('*.incomplete')), [])

    def test_interruption_cleans_temporary_file(self):
        with patch.object(extract_pusht, 'validate_hdf5', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.extract()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob('*.incomplete')), [])

    def test_stale_partial_is_not_accepted_as_finished_output(self):
        stale = self.root / 'prepared.h5.incomplete'
        stale.write_bytes(b'partial')
        self.extract()
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(stale.read_bytes(), b'partial')

    def test_extracts_hdf5_with_two_megabyte_decoder_window(self):
        import numpy as np
        large = self.root / 'large-source.h5'
        with h5py.File(large, 'w') as f:
            f['bytes'] = np.arange(3 * 1024**2, dtype=np.uint8)
        self.payload = large.read_bytes()
        params = zstandard.ZstdCompressionParameters.from_level(1, window_log=21)
        self.archive.write_bytes(zstandard.ZstdCompressor(compression_params=params).compress(self.payload))
        self.assertEqual(zstandard.get_frame_parameters(self.archive.read_bytes()).window_size, 2 * 1024**2)
        self.extract()
        self.assertEqual(self.target.read_bytes(), self.payload)
