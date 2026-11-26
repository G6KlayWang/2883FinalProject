"""Verify and atomically extract a single Zstandard-compressed HDF5 file."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import zstandard

from scripts.data.download_pusht import verify_file

LOG = logging.getLogger(__name__)
RESERVE_BYTES = 1024**3


def validate_hdf5(path: Path) -> None:
    """Reject non-HDF5 output before publishing it."""
    with h5py.File(path, 'r'):
        pass


def extract_archive(archive: Path, output: Path, expected_size: int,
                    expected_sha256: str) -> dict:
    """Publish only a complete, verified HDF5; preserve source and existing output.

    The archive must contain one frame with a declared content size. Temporary
    files live beside output; hard-link publication fails atomically if another
    process has already created output. A killed process may leave an ignored
    unique .incomplete file, which is never considered a completed extraction.
    """
    archive, output = Path(archive), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    verify_file(archive, expected_size, expected_sha256)
    with archive.open('rb') as source:
        header = source.read(18)
        params = zstandard.get_frame_parameters(header)
        size = params.content_size
        if size in (zstandard.CONTENTSIZE_UNKNOWN, zstandard.CONTENTSIZE_ERROR):
            raise ValueError('Archive must declare its decompressed size')
        if params.window_size > 128 * 1024**2:
            raise ValueError('Zstandard window exceeds 128 MiB safety limit')
        free = shutil.disk_usage(output.parent).free
        if free < size + RESERVE_BYTES:
            raise OSError(f'Insufficient free space: need {size + RESERVE_BYTES} bytes')
        LOG.info('Verified archive; extracting %d bytes; free space %d', size, free)
        source.seek(0)
        descriptor, name = tempfile.mkstemp(prefix=output.name + '.', suffix='.incomplete', dir=output.parent)
        temporary = Path(name)
        try:
            digest, source_digest = hashlib.sha256(), hashlib.sha256()
            count = 0
            decoder = zstandard.ZstdDecompressor(max_window_size=128 * 1024).decompressobj()
            with os.fdopen(descriptor, 'wb') as target:
                # Small compressed reads bound expansion memory, even for high ratios.
                for chunk in iter(lambda: source.read(1024), b''):
                    source_digest.update(chunk)
                    data = decoder.decompress(chunk)
                    count += len(data)
                    if count > size or decoder.unused_data:
                        raise ValueError('Unexpected decompressed size or trailing frame')
                    target.write(data)
                    digest.update(data)
                if not decoder.eof or count != size:
                    raise ValueError('Truncated frame or incorrect decompressed size')
                if source_digest.hexdigest() != expected_sha256:
                    raise ValueError('Archive SHA-256 changed during extraction')
                target.flush()
                os.fsync(target.fileno())
            validate_hdf5(temporary)
            os.link(temporary, output)  # Atomic publication, with no overwrite race.
            receipt = {
                'archive_size_bytes': expected_size, 'archive_sha256': expected_sha256,
                'hdf5_size_bytes': count, 'hdf5_sha256': digest.hexdigest(),
            }
            LOG.info('EXTRACTED %s | %s', output, receipt)
            return receipt
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-manifest', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    manifest = json.loads(args.download_manifest.read_text())
    print(json.dumps(extract_archive(args.archive, args.output, manifest['size_bytes'], manifest['sha256']), indent=2))


if __name__ == '__main__':
    main()
