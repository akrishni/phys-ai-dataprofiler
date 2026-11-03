from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from io import BytesIO
import struct

try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# MAT-File magic bytes
MATFILE_V5_MAGIC = b"MATLAB 5.0 MAT-file"
MATFILE_V73_MAGIC = b"MATLAB 7.3 MAT-file"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
HDF5_USERBLOCK_OFFSET = 512  # MAT v7.3 puts the HDF5 superblock after a 512-byte user block


@dataclass(frozen=True)
class VariableInfo:
    name: str
    dtype: str
    shape: tuple
    size_bytes: int
    is_sparse: bool = False
    is_complex: bool = False
    is_struct: bool = False
    is_cell: bool = False
    field_names: list[str] | None = None


@dataclass(frozen=True)
class MatFileHeader:
    description: str
    version: str
    endian: str
    created: str | None = None


@dataclass(frozen=True)
class MatInspectionResult:
    url: str
    is_matfile: bool
    file_format: str | None  # "v4", "v5", "v7", "v7.3"
    size_bytes: int | None
    path: Path | None = None
    error: str | None = None
    header: MatFileHeader | None = None
    variables: list[VariableInfo] | None = None
    variable_names: list[str] | None = None
    scipy_available: bool = SCIPY_AVAILABLE
    h5py_available: bool = H5PY_AVAILABLE


class MatFileInspector:
    """Inspect a MATLAB .mat file and extract header and variable information."""

    def __init__(self, url: str) -> None:
        self.url = url

    def inspect(self) -> MatInspectionResult:
        try:
            path = self._resolve_local_path(self.url)
            if path is not None:
                return self._inspect_local(path)
            return self._inspect_remote(self.url)
        except Exception as exc:
            return MatInspectionResult(
                url=self.url,
                is_matfile=False,
                file_format=None,
                size_bytes=None,
                error=str(exc),
            )

    def _inspect_local(self, path: Path) -> MatInspectionResult:
        size = path.stat().st_size

        # Read magic bytes to determine format
        with path.open("rb") as f:
            magic = f.read(128)  # Read first 128 bytes for header

        file_format = self._detect_format(magic)
        is_matfile = file_format is not None

        if not is_matfile:
            return MatInspectionResult(
                url=self.url,
                is_matfile=False,
                file_format=None,
                size_bytes=size,
                path=path,
                error="Not a valid MATLAB file",
            )

        # Parse the file based on format
        try:
            if file_format == "v7.3":
                if not H5PY_AVAILABLE:
                    return MatInspectionResult(
                        url=self.url,
                        is_matfile=True,
                        file_format=file_format,
                        size_bytes=size,
                        path=path,
                        error="h5py library required for MAT-File v7.3 format",
                    )
                with path.open("rb") as f:
                    mat_data = self._parse_hdf5_mat(f)
            else:
                if not SCIPY_AVAILABLE:
                    return MatInspectionResult(
                        url=self.url,
                        is_matfile=True,
                        file_format=file_format,
                        size_bytes=size,
                        path=path,
                        error="scipy library required for MAT-File v4/v5/v7 format",
                    )
                with path.open("rb") as f:
                    mat_data = self._parse_scipy_mat(f, magic)

            return MatInspectionResult(
                url=self.url,
                is_matfile=True,
                file_format=file_format,
                size_bytes=size,
                path=path,
                **mat_data,
            )
        except Exception as e:
            return MatInspectionResult(
                url=self.url,
                is_matfile=True,
                file_format=file_format,
                size_bytes=size,
                path=path,
                error=f"Failed to parse MAT-File: {e}",
            )

    def _inspect_remote(self, url: str) -> MatInspectionResult:
        try:
            # Fetch the file header
            req = __import__("urllib.request").Request(
                url,
                headers={"Range": "bytes=0-127"},
            )
            with urlopen(req) as resp:
                magic = resp.read(128)
                size = self._content_length(resp.headers)

            file_format = self._detect_format(magic)
            is_matfile = file_format is not None

            if not is_matfile:
                return MatInspectionResult(
                    url=url,
                    is_matfile=False,
                    file_format=None,
                    size_bytes=size,
                    error="Not a valid MATLAB file",
                )

            # Download entire file for parsing
            req = __import__("urllib.request").Request(url)
            with urlopen(req) as resp:
                file_data = resp.read()

            # Parse based on format
            if file_format == "v7.3":
                if not H5PY_AVAILABLE:
                    return MatInspectionResult(
                        url=url,
                        is_matfile=True,
                        file_format=file_format,
                        size_bytes=size,
                        error="h5py library required for MAT-File v7.3 format",
                    )
                mat_data = self._parse_hdf5_mat(BytesIO(file_data))
            else:
                if not SCIPY_AVAILABLE:
                    return MatInspectionResult(
                        url=url,
                        is_matfile=True,
                        file_format=file_format,
                        size_bytes=size,
                        error="scipy library required for MAT-File v4/v5/v7 format",
                    )
                mat_data = self._parse_scipy_mat(BytesIO(file_data), magic)

            return MatInspectionResult(
                url=url,
                is_matfile=True,
                file_format=file_format,
                size_bytes=size,
                **mat_data,
            )
        except Exception as e:
            return MatInspectionResult(
                url=url,
                is_matfile=False,
                file_format=None,
                size_bytes=None,
                error=f"Failed to inspect remote file: {e}",
            )

    @staticmethod
    def _detect_format(magic: bytes) -> str | None:
        """Detect MAT-File format from magic bytes."""
        if magic[:8] == HDF5_MAGIC:
            # Bare HDF5 file (no MATLAB user block); treat as v7.3
            return "v7.3"
        elif MATFILE_V73_MAGIC in magic[:124] or magic[124:126] in (b"\x00\x02", b"\x02\x00"):
            # MAT v7.3: 128-byte MATLAB text header, HDF5 superblock at byte 512
            return "v7.3"
        elif len(magic) > HDF5_USERBLOCK_OFFSET + 8 and magic[HDF5_USERBLOCK_OFFSET:HDF5_USERBLOCK_OFFSET + 8] == HDF5_MAGIC:
            return "v7.3"
        elif MATFILE_V5_MAGIC in magic[:124]:
            # MAT-File v5/v7 format (most common)
            return "v5"
        elif len(magic) >= 4:
            # Try to detect v4 format (has specific structure)
            # v4 uses machine-specific formats, harder to detect reliably
            # We'll rely on scipy to handle this
            try:
                # Check for some v4 indicators
                mopt = struct.unpack("i", magic[:4])[0]
                if mopt in [0, 1, 10, 100, 1000]:  # Common v4 machine types
                    return "v4"
            except Exception:
                pass

        return None

    @staticmethod
    def _parse_scipy_mat(file_handle, magic: bytes) -> dict:
        """Parse MAT-File using scipy."""
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        # Read the header
        file_handle.seek(0)
        header_bytes = file_handle.read(128)

        # Parse header (first 116 bytes are description, then version info)
        description = header_bytes[:116].decode("ascii", errors="ignore").strip()
        version = struct.unpack("H", header_bytes[124:126])[0]
        endian_indicator = header_bytes[126:128]

        if endian_indicator == b"IM":
            endian = "little"
        elif endian_indicator == b"MI":
            endian = "big"
        else:
            endian = "unknown"

        header = MatFileHeader(
            description=description,
            version=f"0x{version:04x}",
            endian=endian,
        )

        # Load the MAT file to inspect variables
        file_handle.seek(0)
        try:
            mat_contents = scipy.io.loadmat(
                file_handle,
                struct_as_record=False,
                squeeze_me=False,
                mat_dtype=False,
            )
        except Exception as e:
            return {"header": header, "error": f"Failed to load variables: {e}"}

        # Extract variable information
        variables = []
        var_names = []

        for name, value in mat_contents.items():
            # Skip metadata fields
            if name.startswith("__"):
                continue

            var_names.append(name)
            var_info = MatFileInspector._extract_variable_info(name, value)
            variables.append(var_info)

        return {
            "header": header,
            "variables": variables if variables else None,
            "variable_names": sorted(var_names) if var_names else None,
        }

    @staticmethod
    def _parse_hdf5_mat(file_handle) -> dict:
        """Parse MAT-File v7.3 (HDF5-based) using h5py.

        Lists top-level MATLAB variables. Numeric/char/logical arrays are HDF5
        datasets; structs and cells are groups (or reference datasets into
        the #refs# group). Only HDF5 metadata is touched, never dataset payloads,
        so this works over a byte-range file object as well as a local file.
        """
        if not H5PY_AVAILABLE:
            return {"error": "h5py not available"}

        try:
            h5file = h5py.File(file_handle, "r")
        except Exception as e:
            return {"error": f"Failed to open HDF5 file: {e}"}

        header = MatFileHeader(
            description="MATLAB 7.3 MAT-file (HDF5)",
            version="7.3",
            endian="little",  # HDF5 handles endianness
        )

        variables = []
        var_names = []
        try:
            for name in h5file.keys():
                if name == "#refs#":
                    continue
                obj = h5file[name]
                var_names.append(name)
                variables.append(MatFileInspector._extract_hdf5_variable_info(name, obj))
        finally:
            h5file.close()

        return {
            "header": header,
            "variables": variables if variables else None,
            "variable_names": sorted(var_names) if var_names else None,
        }

    @staticmethod
    def _extract_variable_info(name: str, value) -> VariableInfo:
        """Extract information about a variable from scipy-loaded data."""
        if not NUMPY_AVAILABLE:
            return VariableInfo(
                name=name,
                dtype="unknown",
                shape=(),
                size_bytes=0,
            )

        import numpy as np
        from scipy.sparse import issparse

        is_sparse = issparse(value)
        is_complex = False
        is_struct = False
        is_cell = False
        field_names = None

        if hasattr(value, "dtype"):
            dtype_str = str(value.dtype)
            is_complex = np.iscomplexobj(value)

            # Check if it's a structured array (MATLAB struct)
            if value.dtype.names is not None:
                is_struct = True
                field_names = list(value.dtype.names)
        elif isinstance(value, np.ndarray) and value.dtype == object:
            # Could be cell array or object array
            is_cell = True
            dtype_str = "cell"
        else:
            dtype_str = type(value).__name__

        shape = getattr(value, "shape", ())

        # Calculate size
        if hasattr(value, "nbytes"):
            size_bytes = value.nbytes
        elif is_sparse:
            size_bytes = value.data.nbytes + value.indices.nbytes + value.indptr.nbytes
        else:
            size_bytes = 0

        return VariableInfo(
            name=name,
            dtype=dtype_str,
            shape=shape,
            size_bytes=size_bytes,
            is_sparse=is_sparse,
            is_complex=is_complex,
            is_struct=is_struct,
            is_cell=is_cell,
            field_names=field_names,
        )

    @staticmethod
    def _extract_hdf5_variable_info(name: str, obj) -> VariableInfo:
        """Describe a top-level MAT v7.3 variable from its HDF5 object and MATLAB attributes."""
        attrs = dict(obj.attrs) if hasattr(obj, "attrs") else {}
        mclass = attrs.get("MATLAB_class")
        if isinstance(mclass, bytes):
            mclass = mclass.decode("ascii", errors="replace")
        elif mclass is not None and not isinstance(mclass, str):
            mclass = str(mclass)
        is_empty = bool(attrs.get("MATLAB_empty", 0))

        if isinstance(obj, h5py.Group):
            # MATLAB struct (scalar) -> group whose members are the fields.
            fields = list(obj.keys())
            return VariableInfo(
                name=name,
                dtype=mclass or "struct",
                shape=(1, 1),
                size_bytes=0,
                is_struct=True,
                field_names=fields,
            )

        dtype_str = str(obj.dtype)
        shape = tuple(reversed(obj.shape)) if obj.shape else ()  # HDF5 is row-major; MATLAB dims are reversed
        is_ref = h5py.check_dtype(ref=obj.dtype) is not None
        is_complex = obj.dtype.names is not None and set(obj.dtype.names) >= {"real", "imag"}
        if is_ref:
            # Cell arrays and struct arrays are stored as object references into #refs#.
            dtype_str = mclass or "cell"
        elif mclass in ("char",):
            dtype_str = "char"
        elif mclass in ("logical",):
            dtype_str = "bool"
        elif is_complex:
            dtype_str = "complex" + str(obj.dtype["real"].itemsize * 16)
        elif mclass:
            dtype_str = mclass
        if is_empty:
            shape = tuple(int(x) for x in obj[()]) if obj.size else (0, 0)

        return VariableInfo(
            name=name,
            dtype=dtype_str,
            shape=shape,
            size_bytes=int(obj.size * obj.dtype.itemsize) if not is_empty else 0,
            is_complex=is_complex,
            is_struct=mclass == "struct",
            is_cell=mclass == "cell" or (is_ref and mclass is None),
            field_names=None,
        )

    @staticmethod
    def _resolve_local_path(url: str) -> Path | None:
        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            # file:///tmp/foo.mat or bare /tmp/foo.mat
            raw = unquote(parsed.path) if parsed.scheme == "file" else url
            return Path(raw)
        return None

    @staticmethod
    def _content_length(headers) -> int | None:
        # Prefer Content-Range total when using Range requests.
        content_range = headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                return int(total)
        length = headers.get("Content-Length")
        return int(length) if length and length.isdigit() else None


if __name__ == "__main__":
    import sys

    result = MatFileInspector(sys.argv[1]).inspect()
    print(result)
