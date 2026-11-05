"""A small, dependency-free reader for safetensors checkpoints.

The reader intentionally avoids loading whole tensors into memory.  It exposes
the checkpoint index and allows callers to stream tensor bytes in chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import struct
from typing import Iterator, Sequence

import array
import math
import sys


_HEADER_SIZE_BYTES = 8
_DEFAULT_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CheckpointIndex:
    path: str
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str] | None

    def __len__(self) -> int:
        return len(self.tensors)


def read_checkpoint_index(path: str | os.PathLike[str]) -> CheckpointIndex:
    """Read a safetensors header and return an index of its tensors."""

    path = os.fspath(path)
    with open(path, "rb") as handle:
        prefix = handle.read(_HEADER_SIZE_BYTES)
        if len(prefix) != _HEADER_SIZE_BYTES:
            raise ValueError(f"Truncated safetensors header in {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1:
            raise ValueError(f"Invalid safetensors header size in {path}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"Truncated safetensors header in {path}")
        data_start = _HEADER_SIZE_BYTES + header_size
        file_size = os.path.getsize(path)

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header in {path} is not an object")

    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"Invalid __metadata__ in {path}")

    tensors: dict[str, TensorInfo] = {}
    for name, description in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(description, dict):
            raise ValueError(f"Invalid tensor entry in {path}")
        dtype = description.get("dtype")
        shape = description.get("shape")
        offsets = description.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise ValueError(f"Malformed tensor description for {name!r} in {path}")
        if len(offsets) != 2 or not all(isinstance(offset, int) for offset in offsets):
            raise ValueError(f"Malformed offsets for {name!r} in {path}")
        if not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape):
            raise ValueError(f"Malformed shape for {name!r} in {path}")
        start, end = offsets
        if start < 0 or end < start or data_start + end > file_size:
            raise ValueError(f"Out-of-range offsets for {name!r} in {path}")
        tensors[name] = TensorInfo(name, normalize_dtype(dtype), tuple(shape), start, end)

    return CheckpointIndex(path, tensors, metadata)


def iter_tensor_chunks(
    path: str | os.PathLike[str],
    info: TensorInfo,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> Iterator[bytes]:
    """Stream the raw bytes of one tensor in aligned chunks."""

    path = os.fspath(path)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    item_size = dtype_item_size(info.dtype)
    if item_size is None:
        # Float8 and other exotic formats can still be compared byte-wise.
        alignment = 1
    else:
        alignment = item_size
    chunk_bytes -= chunk_bytes % alignment
    if chunk_bytes <= 0:
        chunk_bytes = alignment

    data_start = _data_start(path)
    with open(path, "rb") as handle:
        handle.seek(data_start + info.start)
        remaining = info.nbytes
        while remaining:
            requested = min(chunk_bytes, remaining)
            chunk = handle.read(requested)
            if len(chunk) != requested:
                raise ValueError(f"Truncated tensor {info.name!r} in {path}")
            yield chunk
            remaining -= requested


def _data_start(path: str) -> int:
    with open(path, "rb") as handle:
        prefix = handle.read(_HEADER_SIZE_BYTES)
        if len(prefix) != _HEADER_SIZE_BYTES:
            raise ValueError(f"Truncated safetensors header in {path}")
        return _HEADER_SIZE_BYTES + struct.unpack("<Q", prefix)[0]


def dtype_item_size(dtype: str) -> int | None:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "F32": 4,
        "I64": 8,
        "F64": 8,
    }
    return sizes.get(dtype.upper())


# Some safetensors writers use lowercase dtype spellings in older files.
_ALIASES = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}


def normalize_dtype(dtype: str) -> str:
    return _ALIASES.get(dtype, dtype).upper()


def is_numerically_supported(dtype: str) -> bool:
    return normalize_dtype(dtype) in {
        "BOOL",
        "U8",
        "I8",
        "I16",
        "I32",
        "I64",
        "F16",
        "BF16",
        "F32",
        "F64",
    }


_BF16_TABLE: list[float] | None = None


def _bf16_table() -> list[float]:
    global _BF16_TABLE
    if _BF16_TABLE is None:
        table = []
        for bits in range(65536):
            packed = struct.pack("<I", bits << 16)
            table.append(struct.unpack("<f", packed)[0])
        _BF16_TABLE = table
    return _BF16_TABLE


def decode_chunk(chunk: bytes, dtype: str) -> tuple[float, ...]:
    """Decode a little-endian tensor chunk into Python numeric values.

    This dependency-free decoder uses ``array`` on little-endian systems, which
    covers the overwhelming majority of machines used for LLM work.  It falls
    back to ``struct`` on big-endian systems.
    """

    dtype = normalize_dtype(dtype)
    little_endian = sys.byteorder == "little"

    if dtype == "BF16":
        table = _bf16_table()
        if little_endian:
            values = array.array("H")
            values.frombytes(chunk)
            return tuple(table[value] for value in values)
        count = len(chunk) // 2
        return tuple(table[value] for value in struct.unpack("<" + "H" * count, chunk))

    typecodes = {
        "I8": "b",
        "I16": "h",
        "I32": "i",
        "I64": "q",
        "F16": "e",
        "F32": "f",
        "F64": "d",
    }
    typecode = typecodes.get(dtype)
    if typecode is None:
        if dtype == "BOOL":
            return tuple(1.0 if value else 0.0 for value in chunk)
        if dtype == "U8":
            return tuple(chunk)
        raise ValueError(f"Unsupported numeric dtype: {dtype}")

    if little_endian:
        values = array.array(typecode)
        values.frombytes(chunk)
        return tuple(values)

    item_size = {
        "b": 1,
        "h": 2,
        "i": 4,
        "q": 8,
        "e": 2,
        "f": 4,
        "d": 8,
    }[typecode]
    count = len(chunk) // item_size
    return struct.unpack("<" + typecode * count, chunk)


def iter_decoded_chunks(
    path: str | os.PathLike[str],
    info: TensorInfo,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> Iterator[tuple[float, ...]]:
    if not is_numerically_supported(info.dtype):
        raise ValueError(f"Tensor {info.name!r} uses unsupported dtype {info.dtype}")
    for chunk in iter_tensor_chunks(path, info, chunk_bytes):
        yield decode_chunk(chunk, info.dtype)



def write_safetensors(
    path: str | os.PathLike[str],
    tensors: dict[str, tuple[str, list[int], Sequence[float | int]]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a minimal F32 safetensors checkpoint.

    ``tensors`` maps a tensor name to ``(dtype, shape, flat_values)``.  The
    values must be in row-major order.  This writer is intentionally small and
    dependency-free; it is used to create synthetic test models.
    """

    path = os.fspath(path)
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata

    payload = bytearray()
    offset = 0
    for name, (dtype, shape, values) in tensors.items():
        normalized_dtype = normalize_dtype(dtype)
        if normalized_dtype != "F32":
            raise ValueError(f"write_safetensors currently supports F32, got {dtype}")
        expected_numel = math.prod(shape)
        if len(values) != expected_numel:
            raise ValueError(
                f"Tensor {name!r} has {len(values)} values, expected {expected_numel}"
            )
        encoded = struct.pack("<" + "f" * expected_numel, *values)
        header[name] = {
            "dtype": normalized_dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(encoded)],
        }
        payload.extend(encoded)
        offset += len(encoded)

    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padded_length = (len(raw_header) + 7) // 8 * 8
    raw_header += b" " * (padded_length - len(raw_header))
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", padded_length))
        handle.write(raw_header)
        handle.write(payload)


def read_tensor(path: str | os.PathLike[str], info: TensorInfo) -> tuple[float, ...]:
    """Read and decode one complete tensor."""

    path = os.fspath(path)
    data_start = _data_start(path)
    with open(path, "rb") as handle:
        handle.seek(data_start + info.start)
        raw = handle.read(info.nbytes)
    if len(raw) != info.nbytes:
        raise ValueError(f"Truncated tensor {info.name!r} in {path}")
    return decode_chunk(raw, info.dtype)

def discover_checkpoints(path: str | os.PathLike[str]) -> list[str]:
    """Return safetensors files for a path, accepting a file or directory."""

    path = os.fspath(path)
    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            return [path]
        raise ValueError(f"Expected a .safetensors file or model directory: {path}")
    if not os.path.isdir(path):
        raise ValueError(f"Path does not exist: {path}")

    files = []
    for root, directories, filenames in os.walk(path):
        directories.sort()
        for filename in sorted(filenames):
            if filename.endswith(".safetensors"):
                files.append(os.path.join(root, filename))
    if not files:
        raise ValueError(f"No .safetensors files found under {path}")
    return files


def build_model_index(path: str | os.PathLike[str]) -> dict[str, tuple[TensorInfo, str]]:
    """Build a name-to-(tensor info, file path) map for a model path."""

    result: dict[str, tuple[TensorInfo, str]] = {}
    for checkpoint_path in discover_checkpoints(path):
        index = read_checkpoint_index(checkpoint_path)
        for name, info in index.tensors.items():
            if name in result:
                raise ValueError(
                    f"Duplicate tensor name {name!r} in {checkpoint_path} and {result[name][1]}"
                )
            result[name] = (info, checkpoint_path)
    return result


__all__ = [
    "CheckpointIndex",
    "TensorInfo",
    "build_model_index",
    "decode_chunk",
    "discover_checkpoints",
    "dtype_item_size",
    "is_numerically_supported",
    "iter_decoded_chunks",
    "iter_tensor_chunks",
    "normalize_dtype",
    "read_checkpoint_index",
    "read_tensor",
    "write_safetensors",
]
