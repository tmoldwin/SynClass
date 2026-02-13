"""Utilities for loading data from disk or archive (zip/7z fallback)."""
import io
import os
import zipfile
import numpy as np
import pandas as pd

try:
    import py7zr
    from py7zr import WriterFactory
    HAS_PY7ZR = True
except ImportError:
    HAS_PY7ZR = False


def _archive_subpath(data_dir):
    """Get the subpath inside archive (e.g. 'synpase_raw_em' from 'Data/synpase_raw_em/')."""
    return data_dir.rstrip('/').rstrip('\\').split(os.sep)[-1]


def _load_npy_from_7z(archive_path, arc_name):
    """Load single .npy from 7z using extract+factory (no full extraction)."""
    import lzma

    class BytesIOFactory(WriterFactory):
        def __init__(self):
            self.buffers = {}

        def create(self, fn):
            buf = io.BytesIO()
            self.buffers[fn] = buf
            return buf

    with py7zr.SevenZipFile(archive_path, 'r') as z:
        factory = BytesIOFactory()
        try:
            z.extract(targets=[arc_name], factory=factory)
        except lzma.LZMAError as e:
            raise FileNotFoundError(f"Corrupt data in archive for {arc_name}: {e}") from e
        if arc_name not in factory.buffers:
            raise KeyError(arc_name)
        buf = factory.buffers[arc_name]
        buf.seek(0)
        return np.load(buf)


def load_npy(data_dir, archive_path, filename):
    """
    Load .npy file from disk first; if not found, load from archive (zip or 7z).
    Returns numpy array.
    """
    path = os.path.join(data_dir, filename)
    if os.path.isfile(path):
        try:
            return np.load(path)
        except (EOFError, OSError):
            pass  # Corrupt/empty file, fall through to archive
    if not archive_path or not os.path.isfile(archive_path):
        raise FileNotFoundError(f"{path} not found")
    sub = _archive_subpath(data_dir)
    if archive_path.lower().endswith('.7z'):
        if not HAS_PY7ZR:
            raise ImportError("py7zr required for .7z archives: pip install py7zr")
        for arc_name in [os.path.join(sub, filename), filename]:
            try:
                return _load_npy_from_7z(archive_path, arc_name)
            except KeyError:
                continue
        raise FileNotFoundError(f"{filename} not found in {archive_path}")
    # zip
    with zipfile.ZipFile(archive_path, 'r') as zf:
        for try_name in [filename, os.path.join(sub, filename)]:
            try:
                return np.load(io.BytesIO(zf.read(try_name)))
            except KeyError:
                continue
        raise FileNotFoundError(f"{filename} not found in {archive_path}")


def load_csv(csv_path, archive_path=None, fallback_csv_path=None):
    """
    Load CSV from disk first; if not found, try fallback path or archive (zip/7z).
    Returns pandas DataFrame.
    """
    for p in ([csv_path, fallback_csv_path] if fallback_csv_path else [csv_path]):
        if p and os.path.isfile(p):
            return pd.read_csv(p)
    if archive_path and os.path.isfile(archive_path):
        basename = os.path.basename(csv_path)
        sub = os.path.basename(os.path.dirname(csv_path))
        if archive_path.lower().endswith('.7z') and HAS_PY7ZR:
            import tempfile
            with py7zr.SevenZipFile(archive_path, 'r') as z:
                for try_name in [os.path.join(sub, basename), basename]:
                    if try_name in z.getnames():
                        with tempfile.TemporaryDirectory() as tmp:
                            z.extract(targets=[try_name], path=tmp)
                            p = os.path.join(tmp, try_name)
                            if os.path.isfile(p):
                                return pd.read_csv(p)
            raise FileNotFoundError(f"CSV not found in {archive_path}")
        with zipfile.ZipFile(archive_path, 'r') as zf:
            for try_name in [basename, os.path.join(sub, basename)]:
                try:
                    return pd.read_csv(io.BytesIO(zf.read(try_name)))
                except KeyError:
                    continue
        raise FileNotFoundError(f"CSV not found in {archive_path}")
    raise FileNotFoundError(f"{csv_path} not found")


def list_synapse_files(data_dir, archive_path, suffix='syn.npy'):
    """
    List *_syn.npy files from disk first; if dir missing/empty, list from archive (zip or 7z).
    Returns list of filenames (e.g. ['123_syn.npy', ...]).
    """
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(suffix)]
        if files:
            return files
    if archive_path and os.path.isfile(archive_path):
        if archive_path.lower().endswith('.7z') and HAS_PY7ZR:
            with py7zr.SevenZipFile(archive_path, 'r') as z:
                names = z.getnames()
        else:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                names = zf.namelist()
        out = []
        for n in names:
            if n.endswith(suffix) and not n.startswith('__'):
                out.append(os.path.basename(n))
        return list(dict.fromkeys(out))
    return []
