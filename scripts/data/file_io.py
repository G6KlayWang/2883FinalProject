"""Small file primitives shared by offline preparation commands."""
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def new_output(path: Path) -> Iterator[Path]:
    """Give the caller a temporary path; publish without replacing existing data."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.incomplete', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_new(path: Path, payload: dict) -> None:
    with new_output(path) as temporary:
        with temporary.open('w') as target:
            json.dump(payload, target, indent=2, sort_keys=True, allow_nan=False)
            target.write('\n')
            target.flush()
            os.fsync(target.fileno())
