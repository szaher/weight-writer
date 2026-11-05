"""A small, local chat runner for real Hugging Face Transformers models.

This module intentionally keeps the implementation small.  It is meant for
local testing and development, not for production serving.  It supports both
ordinary causal language models and multimodal image-text-to-text models when
a text-only prompt is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any


@dataclass
class RealModelChatSession:
    """An interactive chat session backed by a local Transformers model."""

    model_path: Path
    max_new_tokens: int
    max_context_tokens: int | None
    system_prompt: str | None
    model: Any
    processor: Any
    tokenizer: Any
    history: list[dict[str, str]]

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        *,
        max_new_tokens: int = 128,
        system_prompt: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "RealModelChatSession":
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Real-model chat requires the optional Transformers and PyTorch "
                "dependencies. Install them with: uv sync --extra chat"
            ) from exc

        path = Path(model_path)
        config = _read_config(path)
        multimodal = _is_multimodal(config)

        if multimodal:
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForCausalLM

        try:
            import accelerate  # noqa: F401
            has_accelerate = True
        except ImportError:
            has_accelerate = False

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if dtype != "ignore":
            model_kwargs["dtype"] = dtype

        if has_accelerate:
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": device}
            model = model_class.from_pretrained(path, **model_kwargs)
        else:
            model = model_class.from_pretrained(path, **model_kwargs)
            if device != "auto":
                model = model.to(device)

        if multimodal:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", processor)
        else:
            processor = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            tokenizer = processor

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Some checkpoints ship a generation_config.json with max_length set.
        # When the caller asks for max_new_tokens, that value should control the
        # response length without producing a conflicting-arguments warning.
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
            generation_config.max_length = None

        max_context_tokens = _max_context_tokens(model)

        history: list[dict[str, str]] = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})

        return cls(
            model_path=path,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            system_prompt=system_prompt,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            history=history,
        )

    def say(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        self.history.append({"role": "user", "content": prompt})
        inputs = self._build_inputs()
        input_length = int(inputs["input_ids"].shape[-1])
        max_new_tokens = self.max_new_tokens

        if self.max_context_tokens is not None:
            available_tokens = self.max_context_tokens - input_length
            if available_tokens <= 0:
                inputs = _truncate_inputs(inputs, self.max_context_tokens)
                input_length = self.max_context_tokens
                available_tokens = 1
            max_new_tokens = min(max_new_tokens, available_tokens)

        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = _move_to_device(inputs, device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_ids = output_ids[0, input_length:]
        reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _build_inputs(self) -> Any:
        try:
            if self.processor is not self.tokenizer and hasattr(
                self.processor, "apply_chat_template"
            ):
                # Multimodal processors return a plain string unless tokenization
                # and a dictionary result are requested explicitly.
                processor_kwargs = {}
                if self.max_context_tokens is not None:
                    processor_kwargs = {
                        "truncation": True,
                        "max_length": self.max_context_tokens,
                    }
                return self.processor.apply_chat_template(
                    self.history,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs=processor_kwargs,
                )

            tokenizer_kwargs = {}
            if self.max_context_tokens is not None:
                tokenizer_kwargs = {
                    "truncation": True,
                    "max_length": self.max_context_tokens,
                }
            result = self.tokenizer.apply_chat_template(
                self.history,
                add_generation_prompt=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
            if isinstance(result, dict):
                return result
            return {"input_ids": result}
        except Exception:
            # Some local checkpoints do not ship a chat template.  A plain
            # text fallback keeps local testing usable.
            text = "\n".join(
                f"{message['role']}: {message['content']}" for message in self.history
            )
            fallback_kwargs = {"return_tensors": "pt"}
            if self.max_context_tokens is not None:
                fallback_kwargs.update(
                    {
                        "truncation": True,
                        "max_length": self.max_context_tokens,
                    }
                )
            return self.tokenizer(text, **fallback_kwargs)


def _read_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing config.json in model directory: {path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config.json in model directory: {path}")
    return config


def _is_multimodal(config: dict[str, Any]) -> bool:
    if "vision_config" in config:
        return True
    architectures = config.get("architectures", [])
    if isinstance(architectures, Sequence) and not isinstance(architectures, str):
        return any("ForConditionalGeneration" in str(architecture) for architecture in architectures)
    return "ForConditionalGeneration" in str(architectures)


def _move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_to_device(item, device) for item in value]
        return type(value)(moved)
    return value


def _max_context_tokens(model: Any) -> int | None:
    config = getattr(model, "config", None)
    candidates = [config, getattr(config, "text_config", None)]
    for candidate in candidates:
        for attribute in ("max_position_embeddings", "n_positions", "max_length"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _truncate_inputs(inputs: Any, max_context_tokens: int) -> Any:
    """Keep only the most recent context tokens from dictionary-like inputs."""

    try:
        if isinstance(inputs, dict):
            items = list(inputs.items())
        else:
            items = [(key, inputs[key]) for key in inputs.keys()]
    except (AttributeError, TypeError):
        return inputs

    sequence_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
    }
    truncated: dict[str, Any] = {}
    for key, value in items:
        if (
            key in sequence_keys
            and hasattr(value, "shape")
            and len(value.shape) >= 2
            and value.shape[-1] > max_context_tokens
        ):
            truncated[key] = value[..., -max_context_tokens:]
        else:
            truncated[key] = value
    return truncated


__all__ = ["RealModelChatSession"]
