"""Create and run tiny, dependency-free synthetic transformer models.

The synthetic models are deliberately small but use a GPT/LLaMA-style tensor
layout.  They are useful for testing weight comparison and visualization
without downloading a real checkpoint.

Model A contains mostly zeroed ``q_proj``/``k_proj``/``v_proj``/``gate_proj``/
``up_proj`` tensors.  Model B contains small deterministic perturbations in
those same tensors, while the tensors that actually affect the output remain
identical.  Both models therefore generate the same token cycle, but their
weight files contain many byte-level and numeric differences.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Sequence

from .safetensors import read_checkpoint_index, read_tensor, write_safetensors

_DEFAULT_VOCAB = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
]


@dataclass(frozen=True)
class SyntheticConfig:
    vocab_size: int
    hidden_size: int
    n_positions: int
    n_layers: int
    n_heads: int
    intermediate_size: int

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        return self.hidden_size // self.n_heads


def default_synthetic_config() -> SyntheticConfig:
    return SyntheticConfig(
        vocab_size=16,
        hidden_size=32,
        n_positions=32,
        n_layers=2,
        n_heads=4,
        intermediate_size=64,
    )


def generate_synthetic_models(
    output_dir: str | os.PathLike[str],
    *,
    seed: int = 1337,
    config: SyntheticConfig | None = None,
) -> tuple[Path, Path]:
    """Generate two tiny synthetic model directories."""

    config = config or default_synthetic_config()
    if config.vocab_size > config.hidden_size:
        raise ValueError("vocab_size must be <= hidden_size for the synthetic cycle model")
    if config.vocab_size != len(_DEFAULT_VOCAB):
        raise ValueError(
            f"The built-in synthetic vocabulary has {len(_DEFAULT_VOCAB)} tokens; "
            f"expected vocab_size={config.vocab_size}"
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_a = root / "model-a"
    model_b = root / "model-b"
    model_a.mkdir(exist_ok=True)
    model_b.mkdir(exist_ok=True)

    _write_model(model_a, config, _DEFAULT_VOCAB, variant="a", seed=seed)
    _write_model(model_b, config, _DEFAULT_VOCAB, variant="b", seed=seed)
    return model_a, model_b


def _write_model(
    directory: Path,
    config: SyntheticConfig,
    vocab: Sequence[str],
    *,
    variant: str,
    seed: int,
) -> None:
    tensors = _build_tensors(config, variant=variant, seed=seed)
    write_safetensors(
        directory / "model.safetensors",
        tensors,
        metadata={
            "format": "weight-writer-synthetic",
            "variant": variant,
            "seed": str(seed),
        },
    )
    _write_config(directory, config)
    _write_vocab(directory, vocab)


def _write_config(directory: Path, config: SyntheticConfig) -> None:
    document = {
        "model_type": "weight_writer_synthetic",
        "architectures": ["WeightWriterSyntheticForCausalLM"],
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "max_position_embeddings": config.n_positions,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "intermediate_size": config.intermediate_size,
        "torch_dtype": "float32",
    }
    with open(directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def _write_vocab(directory: Path, vocab: Sequence[str]) -> None:
    with open(directory / "vocab.txt", "w", encoding="utf-8") as handle:
        for token in vocab:
            handle.write(token + "\n")


def _build_tensors(
    config: SyntheticConfig,
    *,
    variant: str,
    seed: int,
) -> dict[str, tuple[str, list[int], list[float]]]:
    h = config.hidden_size
    v = config.vocab_size
    p = config.n_positions
    layers = config.n_layers
    intermediate = config.intermediate_size

    tensors: dict[str, tuple[str, list[int], list[float]]] = {}

    # Token embeddings form a one-hot cycle.  Position embeddings are unused by
    # the deterministic cycle, but keep the normal transformer tensor layout.
    embed_tokens = [0.0] * (v * h)
    for token in range(v):
        embed_tokens[token * h + token] = 1.0
    tensors["model.embed_tokens.weight"] = ("F32", [v, h], embed_tokens)
    tensors["model.embed_positions.weight"] = ("F32", [p, h], [0.0] * (p * h))

    # The LM head maps token t to token (t + 1) % vocab_size.
    lm_head = [0.0] * (v * h)
    for token in range(v):
        next_token = (token + 1) % v
        lm_head[next_token * h + token] = 1.0
    tensors["lm_head.weight"] = ("F32", [v, h], lm_head)
    tensors["lm_head.bias"] = ("F32", [v], [0.0] * v)

    tensors["model.norm.weight"] = ("F32", [h], [1.0] * h)
    tensors["model.norm.bias"] = ("F32", [h], [0.0] * h)

    _PERTURBATION_OFFSETS = {
        "q_proj": 11,
        "k_proj": 23,
        "v_proj": 37,
        "gate_proj": 53,
        "up_proj": 71,
    }

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = ("F32", [h], [1.0] * h)
        tensors[f"{prefix}.input_layernorm.bias"] = ("F32", [h], [0.0] * h)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = ("F32", [h], [1.0] * h)
        tensors[f"{prefix}.post_attention_layernorm.bias"] = ("F32", [h], [0.0] * h)

        for name in ("q_proj", "k_proj", "v_proj"):
            shape = [h, h]
            values = _maybe_perturb(shape, name, variant, seed + layer * 100 + _PERTURBATION_OFFSETS[name])
            tensors[f"{prefix}.self_attn.{name}.weight"] = ("F32", shape, values)
            tensors[f"{prefix}.self_attn.{name}.bias"] = ("F32", [h], _maybe_perturb([h], name, variant, seed + layer * 101 + _PERTURBATION_OFFSETS[name]))

        tensors[f"{prefix}.self_attn.o_proj.weight"] = ("F32", [h, h], [0.0] * (h * h))
        tensors[f"{prefix}.self_attn.o_proj.bias"] = ("F32", [h], [0.0] * h)

        for name in ("gate_proj", "up_proj"):
            shape = [intermediate, h]
            values = _maybe_perturb(shape, name, variant, seed + layer * 103 + _PERTURBATION_OFFSETS[name])
            tensors[f"{prefix}.mlp.{name}.weight"] = ("F32", shape, values)
            tensors[f"{prefix}.mlp.{name}.bias"] = ("F32", [intermediate], _maybe_perturb([intermediate], name, variant, seed + layer * 107 + _PERTURBATION_OFFSETS[name]))

        tensors[f"{prefix}.mlp.down_proj.weight"] = ("F32", [h, intermediate], [0.0] * (h * intermediate))
        tensors[f"{prefix}.mlp.down_proj.bias"] = ("F32", [h], [0.0] * h)

    return tensors


def _maybe_perturb(
    shape: list[int],
    name: str,
    variant: str,
    seed: int,
) -> list[float]:
    if variant == "a":
        return [0.0] * math.prod(shape)
    if variant != "b":
        raise ValueError(f"Unknown synthetic model variant: {variant}")
    generator = random.Random(seed)
    scale = 0.02
    return [generator.uniform(-scale, scale) for _ in range(math.prod(shape))]


@dataclass
class SyntheticRuntime:
    config: SyntheticConfig
    vocab: list[str]
    tensor_values: dict[str, tuple[float, ...]]
    tensor_shapes: dict[str, tuple[int, ...]]


def load_synthetic_model(model_path: str | os.PathLike[str]) -> SyntheticRuntime:
    """Load a synthetic model for pure-Python inference."""

    path = Path(model_path)
    config = _load_config(path)
    vocab = _load_vocab(path)
    checkpoint = path / "model.safetensors"
    index = read_checkpoint_index(checkpoint)
    tensor_values: dict[str, tuple[float, ...]] = {}
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for name, info in index.tensors.items():
        tensor_values[name] = read_tensor(checkpoint, info)
        tensor_shapes[name] = info.shape
    return SyntheticRuntime(config, vocab, tensor_values, tensor_shapes)


def _load_config(path: Path) -> SyntheticConfig:
    with open(path / "config.json", "r", encoding="utf-8") as handle:
        document = json.load(handle)
    model_type = document.get("model_type")
    if model_type != "weight_writer_synthetic":
        if "text_config" in document:
            detail = " Its configuration nests the language-model settings under 'text_config'."
        else:
            detail = ""
        raise ValueError(
            "This chat command supports the tiny synthetic weight-writer models only. "
            f"The selected checkpoint has model_type={model_type!r}.{detail} "
            "Use an inference runtime such as Transformers, vLLM, or SGLang for real models."
        )
    required = {
        "vocab_size",
        "hidden_size",
        "max_position_embeddings",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
    }
    missing = required - document.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    return SyntheticConfig(
        vocab_size=int(document["vocab_size"]),
        hidden_size=int(document["hidden_size"]),
        n_positions=int(document["max_position_embeddings"]),
        n_layers=int(document["num_hidden_layers"]),
        n_heads=int(document["num_attention_heads"]),
        intermediate_size=int(document["intermediate_size"]),
    )


def _load_vocab(path: Path) -> list[str]:
    with open(path / "vocab.txt", "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def generate_tokens(
    runtime: SyntheticRuntime,
    prompt_tokens: Sequence[int],
    *,
    max_new_tokens: int,
) -> list[int]:
    """Generate token IDs from an already-tokenized prompt."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not prompt_tokens:
        raise ValueError("prompt_tokens must contain at least one token")
    generated: list[int] = []
    for _ in range(max_new_tokens):
        context = list(prompt_tokens) + generated
        context = context[-runtime.config.n_positions :]
        logits = _forward(runtime, context)
        next_token = max(range(len(logits)), key=lambda index: logits[index])
        generated.append(next_token)
    return generated


def decode_tokens(runtime: SyntheticRuntime, tokens: Sequence[int]) -> str:
    """Convert token IDs back to the synthetic model's vocabulary."""

    return " ".join(runtime.vocab[token] for token in tokens)


def generate_from_synthetic_model(
    model_path: str | os.PathLike[str],
    prompt: str,
    *,
    max_new_tokens: int = 16,
) -> str:
    """Generate text from a synthetic model."""

    runtime = load_synthetic_model(model_path)
    tokens = _tokenize_prompt(prompt, runtime.vocab)
    generated = generate_tokens(runtime, tokens, max_new_tokens=max_new_tokens)
    return decode_tokens(runtime, generated)


@dataclass
class SyntheticChatSession:
    """A tiny multi-turn chat session for a synthetic model."""

    runtime: SyntheticRuntime
    context: list[int]

    @classmethod
    def from_model(cls, model_path: str | os.PathLike[str]) -> "SyntheticChatSession":
        runtime = load_synthetic_model(model_path)
        return cls(runtime, [])

    def say(self, prompt: str, *, max_new_tokens: int = 16) -> str:
        prompt_tokens = _tokenize_prompt(prompt, self.runtime.vocab)
        self.context.extend(prompt_tokens)
        self.context = self.context[-self.runtime.config.n_positions :]
        generated = generate_tokens(
            self.runtime,
            self.context,
            max_new_tokens=max_new_tokens,
        )
        self.context.extend(generated)
        self.context = self.context[-self.runtime.config.n_positions :]
        return decode_tokens(self.runtime, generated)


def _tokenize_prompt(prompt: str, vocab: list[str]) -> list[int]:
    words = prompt.split()
    if not words:
        raise ValueError("Prompt must contain at least one token")
    token_to_id = {token: index for index, token in enumerate(vocab)}
    result = []
    for word in words:
        if word not in token_to_id:
            raise ValueError(f"Unknown prompt token: {word!r}")
        result.append(token_to_id[word])
    return result


def _forward(runtime: SyntheticRuntime, tokens: Sequence[int]) -> list[float]:
    config = runtime.config
    values = runtime.tensor_values
    shapes = runtime.tensor_shapes
    h = config.hidden_size
    sequence_length = len(tokens)

    embed = _matrix(values["model.embed_tokens.weight"], shapes["model.embed_tokens.weight"])
    position = _matrix(values["model.embed_positions.weight"], shapes["model.embed_positions.weight"])
    hidden_states = [
        [embed[token][dimension] + position[index][dimension] for dimension in range(h)]
        for index, token in enumerate(tokens)
    ]

    for layer in range(config.n_layers):
        prefix = f"model.layers.{layer}"
        residual = hidden_states

        ln1_weight = list(values[f"{prefix}.input_layernorm.weight"])
        ln1_bias = list(values[f"{prefix}.input_layernorm.bias"])
        normed = [_layer_norm(vector, ln1_weight, ln1_bias) for vector in residual]

        q_weight = _matrix(values[f"{prefix}.self_attn.q_proj.weight"], shapes[f"{prefix}.self_attn.q_proj.weight"])
        q_bias = list(values[f"{prefix}.self_attn.q_proj.bias"])
        k_weight = _matrix(values[f"{prefix}.self_attn.k_proj.weight"], shapes[f"{prefix}.self_attn.k_proj.weight"])
        k_bias = list(values[f"{prefix}.self_attn.k_proj.bias"])
        v_weight = _matrix(values[f"{prefix}.self_attn.v_proj.weight"], shapes[f"{prefix}.self_attn.v_proj.weight"])
        v_bias = list(values[f"{prefix}.self_attn.v_proj.bias"])
        o_weight = _matrix(values[f"{prefix}.self_attn.o_proj.weight"], shapes[f"{prefix}.self_attn.o_proj.weight"])
        o_bias = list(values[f"{prefix}.self_attn.o_proj.bias"])

        q = [_linear(vector, q_weight, q_bias) for vector in normed]
        k = [_linear(vector, k_weight, k_bias) for vector in normed]
        v = [_linear(vector, v_weight, v_bias) for vector in normed]
        attention_output = _attention(q, k, v, config.n_heads, config.head_dim)
        projected = [_linear(vector, o_weight, o_bias) for vector in attention_output]
        hidden_states = [
            [residual[i][dimension] + projected[i][dimension] for dimension in range(h)]
            for i in range(sequence_length)
        ]

        residual = hidden_states
        ln2_weight = list(values[f"{prefix}.post_attention_layernorm.weight"])
        ln2_bias = list(values[f"{prefix}.post_attention_layernorm.bias"])
        normed = [_layer_norm(vector, ln2_weight, ln2_bias) for vector in residual]

        gate_weight = _matrix(values[f"{prefix}.mlp.gate_proj.weight"], shapes[f"{prefix}.mlp.gate_proj.weight"])
        gate_bias = list(values[f"{prefix}.mlp.gate_proj.bias"])
        up_weight = _matrix(values[f"{prefix}.mlp.up_proj.weight"], shapes[f"{prefix}.mlp.up_proj.weight"])
        up_bias = list(values[f"{prefix}.mlp.up_proj.bias"])
        down_weight = _matrix(values[f"{prefix}.mlp.down_proj.weight"], shapes[f"{prefix}.mlp.down_proj.weight"])
        down_bias = list(values[f"{prefix}.mlp.down_proj.bias"])

        mlp_output = []
        for vector in normed:
            gate = _linear(vector, gate_weight, gate_bias)
            up = _linear(vector, up_weight, up_bias)
            activated = [
                up_value * (gate_value / (1.0 + math.exp(-gate_value)))
                for up_value, gate_value in zip(up, gate)
            ]
            mlp_output.append(_linear(activated, down_weight, down_bias))
        hidden_states = [
            [residual[i][dimension] + mlp_output[i][dimension] for dimension in range(h)]
            for i in range(sequence_length)
        ]

    final_weight = list(values["model.norm.weight"])
    final_bias = list(values["model.norm.bias"])
    final_hidden = _layer_norm(hidden_states[-1], final_weight, final_bias)
    lm_weight = _matrix(values["lm_head.weight"], shapes["lm_head.weight"])
    lm_bias = list(values["lm_head.bias"])
    return _linear(final_hidden, lm_weight, lm_bias)


def _matrix(values: Sequence[float], shape: tuple[int, ...]) -> list[list[float]]:
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D tensor, got shape {shape}")
    rows, columns = shape
    return [
        list(values[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]


def _linear(
    vector: Sequence[float],
    weight: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> list[float]:
    if len(weight) != len(bias):
        raise ValueError("Weight and bias sizes differ")
    return [
        sum(vector[index] * row[index] for index in range(len(vector))) + bias[out_index]
        for out_index, row in enumerate(weight)
    ]


def _layer_norm(
    vector: Sequence[float],
    weight: Sequence[float],
    bias: Sequence[float],
) -> list[float]:
    count = len(vector)
    mean = sum(vector) / count
    variance = sum((value - mean) ** 2 for value in vector) / count
    standard_deviation = math.sqrt(variance + 1e-6)
    return [
        ((value - mean) / standard_deviation) * scale + shift
        for value, scale, shift in zip(vector, weight, bias)
    ]


def _attention(
    q: Sequence[Sequence[float]],
    k: Sequence[Sequence[float]],
    v: Sequence[Sequence[float]],
    n_heads: int,
    head_dim: int,
) -> list[list[float]]:
    sequence_length = len(q)
    hidden_size = n_heads * head_dim
    output = [[0.0] * hidden_size for _ in range(sequence_length)]
    scale = 1.0 / math.sqrt(head_dim)

    for head in range(n_heads):
        start = head * head_dim
        end = start + head_dim
        for query_index in range(sequence_length):
            scores = []
            for key_index in range(query_index + 1):
                score = sum(
                    q[query_index][dimension] * k[key_index][dimension]
                    for dimension in range(start, end)
                )
                scores.append(score * scale)
            weights = _softmax(scores)
            for dimension in range(start, end):
                output[query_index][dimension] = sum(
                    weight * v[key_index][dimension]
                    for weight, key_index in zip(weights, range(query_index + 1))
                )
    return output


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


__all__ = [
    "SyntheticConfig",
    "SyntheticRuntime",
    "SyntheticChatSession",
    "decode_tokens",
    "generate_tokens",
    "default_synthetic_config",
    "generate_from_synthetic_model",
    "generate_synthetic_models",
    "load_synthetic_model",
]
