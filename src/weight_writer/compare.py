"""Compare tensors in two safetensors model checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import operator
import os
from typing import Callable

from .safetensors import (
    TensorInfo,
    build_model_index,
    is_numerically_supported,
    iter_decoded_chunks,
    iter_tensor_chunks,
)

_EPSILON = 1e-30


@dataclass
class TensorDiff:
    name: str
    file_a: str
    file_b: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    status: str
    bytes_changed: int = 0
    elements_changed: int | None = None
    mean_abs_diff: float | None = None
    max_abs_diff: float | None = None
    l2_diff: float | None = None
    relative_l2: float | None = None
    cosine_similarity: float | None = None
    nonfinite_a: int = 0
    nonfinite_b: int = 0
    nonfinite_diff: int = 0

    @property
    def changed_fraction(self) -> float | None:
        if self.elements_changed is None or self.numel == 0:
            return None
        return self.elements_changed / self.numel


@dataclass
class ModelDiff:
    path_a: str
    path_b: str
    tensors: list[TensorDiff]
    missing_in_a: list[str]
    missing_in_b: list[str]

    @property
    def identical(self) -> bool:
        return not self.missing_in_a and not self.missing_in_b and all(
            tensor.status == "identical" for tensor in self.tensors
        )

    def to_dict(self) -> dict:
        return {
            "path_a": self.path_a,
            "path_b": self.path_b,
            "identical": self.identical,
            "tensor_count": len(self.tensors),
            "missing_in_a": self.missing_in_a,
            "missing_in_b": self.missing_in_b,
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }


def compare_models(
    path_a: str | os.PathLike[str],
    path_b: str | os.PathLike[str],
    *,
    fast: bool = False,
    chunk_bytes: int = 1024 * 1024,
    progress: Callable[[str, int, int], None] | None = None,
) -> ModelDiff:
    """Compare two local safetensors models.

    In numeric mode, tensors are streamed and decoded in chunks.  In fast mode,
    only exact byte equality is checked, which is useful for huge checkpoints.
    """

    path_a = os.fspath(path_a)
    path_b = os.fspath(path_b)
    index_a = build_model_index(path_a)
    index_b = build_model_index(path_b)

    names_a = set(index_a)
    names_b = set(index_b)
    missing_in_a = sorted(names_b - names_a, key=_natural_key)
    missing_in_b = sorted(names_a - names_b, key=_natural_key)
    common_names = sorted(names_a & names_b, key=_natural_key)

    results: list[TensorDiff] = []
    for position, name in enumerate(common_names, start=1):
        info_a, file_a = index_a[name]
        info_b, file_b = index_b[name]
        if progress is not None:
            progress(name, position, len(common_names))
        results.append(
            _compare_tensor(
                name,
                info_a,
                file_a,
                info_b,
                file_b,
                fast=fast,
                chunk_bytes=chunk_bytes,
            )
        )

    return ModelDiff(path_a, path_b, results, missing_in_a, missing_in_b)


def _compare_tensor(
    name: str,
    info_a: TensorInfo,
    file_a: str,
    info_b: TensorInfo,
    file_b: str,
    *,
    fast: bool,
    chunk_bytes: int,
) -> TensorDiff:
    if info_a.shape != info_b.shape or info_a.dtype != info_b.dtype:
        return TensorDiff(
            name=name,
            file_a=file_a,
            file_b=file_b,
            dtype=info_b.dtype,
            shape=info_b.shape,
            numel=info_b.numel,
            status="shape_or_dtype_mismatch",
        )

    bytes_changed = _count_changed_bytes(file_a, info_a, file_b, info_b, chunk_bytes)
    if bytes_changed == 0:
        # Exact byte equality is sufficient to prove the tensors are identical.
        # Numeric norms are not needed for a diff report.
        return TensorDiff(
            name=name,
            file_a=file_a,
            file_b=file_b,
            dtype=info_a.dtype,
            shape=info_a.shape,
            numel=info_a.numel,
            status="identical",
            bytes_changed=0,
            elements_changed=0,
            mean_abs_diff=0.0,
            max_abs_diff=0.0,
            l2_diff=0.0,
            relative_l2=0.0,
            cosine_similarity=1.0,
        )

    if fast or not is_numerically_supported(info_a.dtype):
        return TensorDiff(
            name=name,
            file_a=file_a,
            file_b=file_b,
            dtype=info_a.dtype,
            shape=info_a.shape,
            numel=info_a.numel,
            status="changed" if fast else "changed_numeric_unavailable",
            bytes_changed=bytes_changed,
        )

    return _compare_tensor_numeric(
        name,
        info_a,
        file_a,
        info_b,
        file_b,
        bytes_changed,
        chunk_bytes,
    )


def _compare_tensor_numeric(
    name: str,
    info_a: TensorInfo,
    file_a: str,
    info_b: TensorInfo,
    file_b: str,
    bytes_changed: int,
    chunk_bytes: int,
) -> TensorDiff:
    count = 0
    changed = 0
    sum_abs_diff = 0.0
    max_abs_diff = 0.0
    sum_sq_diff = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    dot = 0.0
    nonfinite_a = 0
    nonfinite_b = 0
    nonfinite_diff = 0

    chunks_a = iter_decoded_chunks(file_a, info_a, chunk_bytes)
    chunks_b = iter_decoded_chunks(file_b, info_b, chunk_bytes)
    for values_a, values_b in zip(chunks_a, chunks_b, strict=True):
        if len(values_a) != len(values_b):
            raise ValueError(f"Internal chunk-size mismatch for tensor {name!r}")
        count += len(values_a)
        if all(map(math.isfinite, values_a)) and all(map(math.isfinite, values_b)):
            differences = [a - b for a, b in zip(values_a, values_b)]
            changed += sum(map(operator.ne, values_a, values_b))
            absolute_differences = list(map(abs, differences))
            sum_abs_diff += sum(absolute_differences)
            max_abs_diff = max(max_abs_diff, max(absolute_differences, default=0.0))
            sum_sq_diff += math.sumprod(differences, differences)
            sum_sq_a += math.sumprod(values_a, values_a)
            sum_sq_b += math.sumprod(values_b, values_b)
            dot += math.sumprod(values_a, values_b)
            continue

        for a, b in zip(values_a, values_b):
            finite_a = math.isfinite(a)
            finite_b = math.isfinite(b)
            if not finite_a:
                nonfinite_a += 1
            if not finite_b:
                nonfinite_b += 1
            if a != b:
                changed += 1
            if finite_a:
                sum_sq_a += a * a
            if finite_b:
                sum_sq_b += b * b
            if finite_a and finite_b:
                dot += a * b
            difference = a - b
            finite_difference = finite_a and finite_b and math.isfinite(difference)
            if not finite_difference:
                nonfinite_diff += 1
                continue
            absolute = abs(difference)
            sum_abs_diff += absolute
            sum_sq_diff += difference * difference
            if absolute > max_abs_diff:
                max_abs_diff = absolute

    if count != info_a.numel:
        raise ValueError(
            f"Tensor {name!r} decoded element count {count} does not match header {info_a.numel}"
        )

    norm_a = math.sqrt(sum_sq_a)
    norm_b = math.sqrt(sum_sq_b)
    l2_diff = math.sqrt(sum_sq_diff)
    relative_l2 = l2_diff / max(norm_a, norm_b, _EPSILON)
    if norm_a > 0.0 and norm_b > 0.0:
        cosine = dot / (norm_a * norm_b)
        cosine = max(-1.0, min(1.0, cosine))
    else:
        cosine = None

    return TensorDiff(
        name=name,
        file_a=file_a,
        file_b=file_b,
        dtype=info_a.dtype,
        shape=info_a.shape,
        numel=count,
        status="changed",
        bytes_changed=bytes_changed,
        elements_changed=changed,
        mean_abs_diff=sum_abs_diff / count if count else 0.0,
        max_abs_diff=max_abs_diff,
        l2_diff=l2_diff,
        relative_l2=relative_l2,
        cosine_similarity=cosine,
        nonfinite_a=nonfinite_a,
        nonfinite_b=nonfinite_b,
        nonfinite_diff=nonfinite_diff,
    )


def _count_changed_bytes(
    file_a: str,
    info_a: TensorInfo,
    file_b: str,
    info_b: TensorInfo,
    chunk_bytes: int,
) -> int:
    if info_a.nbytes != info_b.nbytes:
        raise ValueError(f"Tensor byte sizes differ for {info_a.name!r}")
    changed = 0
    chunks_a = iter_tensor_chunks(file_a, info_a, chunk_bytes)
    chunks_b = iter_tensor_chunks(file_b, info_b, chunk_bytes)
    for chunk_a, chunk_b in zip(chunks_a, chunks_b, strict=True):
        if chunk_a == chunk_b:
            continue
        changed += sum(byte_a != byte_b for byte_a, byte_b in zip(chunk_a, chunk_b))
    return changed


def _natural_key(value: str) -> tuple[object, ...]:
    """Sort tensor names with embedded layer numbers in numeric order."""

    parts: list[object] = []
    digit_run = ""
    for character in value:
        if character.isdigit():
            digit_run += character
        else:
            if digit_run:
                parts.append(int(digit_run))
                digit_run = ""
            parts.append(character)
    if digit_run:
        parts.append(int(digit_run))
    return tuple(parts)


__all__ = ["ModelDiff", "TensorDiff", "compare_models"]
