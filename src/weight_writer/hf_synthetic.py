"""Create tiny, Hugging Face Transformers-compatible synthetic GPT-2 models.

These are not placeholders.  Each output directory contains a real
``GPT2LMHeadModel`` checkpoint in safetensors format, a valid GPT-2 style
``config.json``, and a small fast-tokenizer vocabulary.  The models are tiny
enough to inspect quickly, but they can be loaded, trained, and fine-tuned with
the normal Hugging Face Transformers stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from .safetensors import write_safetensors

_WORDS = [
    "the", "a", "and", "of", "to", "in", "is", "you", "that", "it",
    "he", "was", "for", "on", "are", "as", "with", "his", "they", "I",
    "at", "be", "this", "have", "from", "or", "one", "had",
]
_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
_VOCAB = _SPECIAL_TOKENS + _WORDS


@dataclass(frozen=True)
class HFSyntheticConfig:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int


def default_hf_synthetic_config() -> HFSyntheticConfig:
    return HFSyntheticConfig(
        vocab_size=len(_VOCAB),
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )


def generate_real_synthetic_models(
    output_dir: str | Path,
    *,
    seed: int = 1337,
    config: HFSyntheticConfig | None = None,
) -> tuple[Path, Path]:
    """Generate two tiny, Transformers-compatible GPT-2 models."""

    config = config or default_hf_synthetic_config()
    if config.vocab_size != len(_VOCAB):
        raise ValueError(
            f"The built-in HF synthetic vocabulary has {len(_VOCAB)} tokens; "
            f"expected vocab_size={config.vocab_size}"
        )
    if config.n_embd % config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_a = root / "model-a"
    model_b = root / "model-b"
    model_a.mkdir(exist_ok=True)
    model_b.mkdir(exist_ok=True)

    _write_model(model_a, config, seed=seed)
    _write_model(model_b, config, seed=seed + 1)
    return model_a, model_b


def _write_model(directory: Path, config: HFSyntheticConfig, *, seed: int) -> None:
    tensors = _build_gpt2_tensors(config, seed=seed)
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={
            "format": "weight-writer-hf-synthetic",
            "architecture": "gpt2",
            "seed": str(seed),
        },
    )
    _write_config(directory, config)
    _write_tokenizer(directory)
    _write_generation_config(directory, config)


def _build_gpt2_tensors(
    config: HFSyntheticConfig,
    *,
    seed: int,
) -> dict[str, tuple[str, list[int], list[float]]]:
    generator = _DeterministicRandom(seed)
    vocab, positions, embedding, layers, heads = (
        config.vocab_size,
        config.n_positions,
        config.n_embd,
        config.n_layer,
        config.n_head,
    )
    intermediate = 4 * embedding
    scale = 1.0 / math.sqrt(2.0 * layers)

    tensors: dict[str, tuple[str, list[int], list[float]]] = {}
    tensors["wte.weight"] = (
        "F32",
        [vocab, embedding],
        [generator.normal(0.0, 0.02) for _ in range(vocab * embedding)],
    )
    tensors["wpe.weight"] = (
        "F32",
        [positions, embedding],
        [generator.normal(0.0, 0.02) for _ in range(positions * embedding)],
    )

    for layer in range(layers):
        prefix = f"h.{layer}"
        tensors[f"{prefix}.ln_1.weight"] = ("F32", [embedding], [1.0] * embedding)
        tensors[f"{prefix}.ln_1.bias"] = ("F32", [embedding], [0.0] * embedding)
        tensors[f"{prefix}.ln_2.weight"] = ("F32", [embedding], [1.0] * embedding)
        tensors[f"{prefix}.ln_2.bias"] = ("F32", [embedding], [0.0] * embedding)

        tensors[f"{prefix}.attn.c_attn.weight"] = (
            "F32",
            [embedding, 3 * embedding],
            [generator.normal(0.0, 0.02 * scale) for _ in range(embedding * 3 * embedding)],
        )
        tensors[f"{prefix}.attn.c_attn.bias"] = ("F32", [3 * embedding], [0.0] * (3 * embedding))
        tensors[f"{prefix}.attn.c_proj.weight"] = (
            "F32",
            [embedding, embedding],
            [generator.normal(0.0, 0.02 * scale) for _ in range(embedding * embedding)],
        )
        tensors[f"{prefix}.attn.c_proj.bias"] = ("F32", [embedding], [0.0] * embedding)

        tensors[f"{prefix}.mlp.c_fc.weight"] = (
            "F32",
            [embedding, intermediate],
            [generator.normal(0.0, 0.02 * scale) for _ in range(embedding * intermediate)],
        )
        tensors[f"{prefix}.mlp.c_fc.bias"] = ("F32", [intermediate], [0.0] * intermediate)
        tensors[f"{prefix}.mlp.c_proj.weight"] = (
            "F32",
            [intermediate, embedding],
            [generator.normal(0.0, 0.02 * scale) for _ in range(intermediate * embedding)],
        )
        tensors[f"{prefix}.mlp.c_proj.bias"] = ("F32", [embedding], [0.0] * embedding)

    tensors["ln_f.weight"] = ("F32", [embedding], [1.0] * embedding)
    tensors["ln_f.bias"] = ("F32", [embedding], [0.0] * embedding)
    return tensors


def _write_config(directory: Path, config: HFSyntheticConfig) -> None:
    document = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": config.vocab_size,
        "n_positions": config.n_positions,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-5,
        "scale_attn_weights": True,
        "use_cache": True,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "transformers_version": "4.35.2",
    }
    with open(directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def _write_generation_config(directory: Path, config: HFSyntheticConfig) -> None:
    document = {
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "max_length": config.n_positions,
        "do_sample": False,
    }
    with open(directory / "generation_config.json", "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def _write_tokenizer(directory: Path) -> None:
    vocab = {token: index for index, token in enumerate(_VOCAB)}
    added_tokens = [
        {
            "id": index,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for index, token in enumerate(_SPECIAL_TOKENS)
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }
    with open(directory / "tokenizer.json", "w", encoding="utf-8") as handle:
        json.dump(tokenizer, handle, indent=2)
        handle.write("\n")

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 32,
        "padding_side": "right",
        "truncation_side": "right",
        "tokenizer_file": "tokenizer.json",
    }
    with open(directory / "tokenizer_config.json", "w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, indent=2)
        handle.write("\n")

    special_tokens_map = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    with open(directory / "special_tokens_map.json", "w", encoding="utf-8") as handle:
        json.dump(special_tokens_map, handle, indent=2)
        handle.write("\n")


class _DeterministicRandom:
    """A tiny, deterministic PRNG sufficient for reproducible test weights."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def _next_uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 0xFFFFFFFF

    def normal(self, mean: float = 0.0, standard_deviation: float = 1.0) -> float:
        first = max(self._next_uniform(), 1e-12)
        second = self._next_uniform()
        return mean + standard_deviation * math.sqrt(-2.0 * math.log(first)) * math.cos(
            2.0 * math.pi * second
        )


__all__ = [
    "HFSyntheticConfig",
    "default_hf_synthetic_config",
    "generate_real_synthetic_models",
]
