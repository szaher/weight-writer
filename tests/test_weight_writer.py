from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

from weight_writer.compare import compare_models
from weight_writer.report import choose_metric, write_html_report, write_json_report
from weight_writer.safetensors import read_checkpoint_index
from weight_writer.hf_synthetic import generate_real_synthetic_models
from weight_writer.real_chat import (
    RealModelChatSession,
    _is_multimodal,
    _move_to_device,
    _truncate_inputs,
)
from weight_writer.synthetic import (
    SyntheticChatSession,
    decode_tokens,
    generate_from_synthetic_model,
    generate_tokens,
    generate_synthetic_models,
    load_synthetic_model,
)


def encode_tensor(dtype: str, values) -> bytes:
    if dtype == "F32":
        return struct.pack("<" + "f" * len(values), *values)
    if dtype == "F16":
        return struct.pack("<" + "e" * len(values), *values)
    if dtype == "BF16":
        encoded = bytearray()
        for value in values:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            encoded.extend(struct.pack("<H", bits >> 16))
        return bytes(encoded)
    if dtype == "I32":
        return struct.pack("<" + "i" * len(values), *values)
    raise AssertionError(f"Unsupported test dtype: {dtype}")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], list[float]]]) -> None:
    header = {}
    payload = bytearray()
    offset = 0
    for name, (dtype, shape, values) in tensors.items():
        encoded = encode_tensor(dtype, values)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
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


class WeightWriterTests(unittest.TestCase):
    def test_read_safetensors_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            write_safetensors(path, {"weight": ("F32", [2, 2], [1, 2, 3, 4])})
            index = read_checkpoint_index(path)
            self.assertEqual(set(index.tensors), {"weight"})
            info = index.tensors["weight"]
            self.assertEqual(info.dtype, "F32")
            self.assertEqual(info.shape, (2, 2))
            self.assertEqual(info.numel, 4)
            self.assertEqual(info.nbytes, 16)

    def test_compare_identical_and_changed_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a = root / "a"
            model_b = root / "b"
            model_a.mkdir()
            model_b.mkdir()
            write_safetensors(
                model_a / "model.safetensors",
                {
                    "model.layers.0.self_attn.q_proj.weight": ("F32", [2, 3], [1, 2, 3, 4, 5, 6]),
                    "model.layers.0.mlp.up_proj.weight": ("F32", [2], [7, 8]),
                },
            )
            write_safetensors(
                model_b / "model.safetensors",
                {
                    "model.layers.0.self_attn.q_proj.weight": ("F32", [2, 3], [1, 2, 3.5, 4, 5, 6]),
                    "model.layers.0.mlp.up_proj.weight": ("F32", [2], [7, 8]),
                },
            )

            diff = compare_models(model_a, model_b)
            self.assertFalse(diff.identical)
            self.assertEqual(diff.missing_in_a, [])
            self.assertEqual(diff.missing_in_b, [])

            changed = next(
                tensor for tensor in diff.tensors if tensor.name.endswith("q_proj.weight")
            )
            identical = next(
                tensor for tensor in diff.tensors if tensor.name.endswith("up_proj.weight")
            )
            self.assertEqual(changed.status, "changed")
            self.assertEqual(changed.elements_changed, 1)
            self.assertEqual(changed.max_abs_diff, 0.5)
            self.assertAlmostEqual(changed.mean_abs_diff, 0.5 / 6)
            self.assertAlmostEqual(changed.l2_diff, 0.5)
            self.assertGreater(changed.relative_l2, 0.0)
            self.assertEqual(identical.status, "identical")
            self.assertEqual(identical.bytes_changed, 0)

    def test_fast_mode_reports_byte_changes_without_numeric_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_safetensors(root / "a.safetensors", {"x": ("F32", [1], [1])})
            write_safetensors(root / "b.safetensors", {"x": ("F32", [1], [2])})
            diff = compare_models(root / "a.safetensors", root / "b.safetensors", fast=True)
            self.assertEqual(diff.tensors[0].status, "changed")
            expected_bytes = sum(
                byte_a != byte_b
                for byte_a, byte_b in zip(struct.pack("<f", 1), struct.pack("<f", 2))
            )
            self.assertEqual(diff.tensors[0].bytes_changed, expected_bytes)
            self.assertIsNone(diff.tensors[0].elements_changed)
            self.assertIsNone(diff.tensors[0].mean_abs_diff)

    def test_missing_tensors_and_dtype_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_safetensors(root / "a.safetensors", {"a": ("F32", [1], [1]), "only_a": ("F32", [1], [2])})
            write_safetensors(root / "b.safetensors", {"a": ("F16", [1], [1]), "only_b": ("F32", [1], [3])})
            diff = compare_models(root / "a.safetensors", root / "b.safetensors")
            self.assertEqual(diff.missing_in_a, ["only_b"])
            self.assertEqual(diff.missing_in_b, ["only_a"])
            self.assertEqual(diff.tensors[0].status, "shape_or_dtype_mismatch")

    def test_reports_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_safetensors(root / "a.safetensors", {"model.layers.0.weight": ("F32", [2], [1, 2])})
            write_safetensors(root / "b.safetensors", {"model.layers.0.weight": ("F32", [2], [1, 3])})
            diff = compare_models(root / "a.safetensors", root / "b.safetensors")
            metric = choose_metric(diff)
            output = root / "report"
            output.mkdir()
            write_json_report(diff, output / "report.json")
            write_html_report(diff, output / "report.html", metric=metric, svg_path=output / "heatmap.svg")

            json_document = json.loads((output / "report.json").read_text())
            self.assertFalse(json_document["identical"])
            html_document = (output / "report.html").read_text()
            svg_document = (output / "heatmap.svg").read_text()
            self.assertIn("LLM weight difference report", html_document)
            self.assertIn("<svg", svg_document)
            self.assertIn("model.layers.0.weight", json_document["tensors"][0]["name"])

    def test_report_with_no_common_tensor_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_safetensors(root / "a.safetensors", {"a.weight": ("F32", [1], [1])})
            write_safetensors(root / "b.safetensors", {"b.weight": ("F32", [1], [2])})
            diff = compare_models(root / "a.safetensors", root / "b.safetensors")
            output = root / "report"
            output.mkdir()
            write_html_report(diff, output / "report.html")
            document = (output / "report.html").read_text()
            self.assertIn("Missing in A", document)
            self.assertIn("Missing in B", document)

    def test_synthetic_models_generate_same_output_but_differ_in_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a, model_b = generate_synthetic_models(root, seed=42)
            self.assertTrue((model_a / "model.safetensors").is_file())
            self.assertTrue((model_b / "model.safetensors").is_file())
            self.assertTrue((model_a / "config.json").is_file())
            self.assertTrue((model_a / "vocab.txt").is_file())

            output_a = generate_from_synthetic_model(model_a, "alpha", max_new_tokens=8)
            output_b = generate_from_synthetic_model(model_b, "alpha", max_new_tokens=8)
            self.assertEqual(output_a, output_b)
            self.assertEqual(
                output_a,
                "beta gamma delta epsilon zeta eta theta iota",
            )

            diff = compare_models(model_a, model_b)
            self.assertFalse(diff.identical)
            self.assertGreater(
                sum(tensor.status == "changed" for tensor in diff.tensors),
                0,
            )

    def test_generate_tokens_and_decode_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a, _ = generate_synthetic_models(root, seed=42)
            runtime = load_synthetic_model(model_a)
            generated = generate_tokens(runtime, [0], max_new_tokens=4)
            self.assertEqual(generated, [1, 2, 3, 4])
            self.assertEqual(decode_tokens(runtime, generated), "beta gamma delta epsilon")

    def test_synthetic_chat_session_continues_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a, _ = generate_synthetic_models(root, seed=42)
            session = SyntheticChatSession.from_model(model_a)
            self.assertEqual(session.say("alpha", max_new_tokens=3), "beta gamma delta")
            self.assertEqual(session.say("beta", max_new_tokens=3), "gamma delta epsilon")

    def test_real_chat_helpers(self) -> None:
        self.assertTrue(_is_multimodal({"vision_config": {}}))
        self.assertTrue(_is_multimodal({"architectures": ["FooForConditionalGeneration"]}))
        self.assertFalse(_is_multimodal({"architectures": ["GPT2LMHeadModel"]}))

        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def to(self, device):
                return FakeTensor((self.value, device))

        moved = _move_to_device(
            {"input_ids": FakeTensor("ids"), "mask": [FakeTensor("mask")]},
            "cpu",
        )
        self.assertEqual(moved["input_ids"].value, ("ids", "cpu"))
        self.assertEqual(moved["mask"][0].value, ("mask", "cpu"))

    def test_real_chat_builds_multimodal_processor_inputs(self) -> None:
        class FakeProcessor:
            def apply_chat_template(self, history, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": "ids", "attention_mask": "mask"}

        class FakeTokenizer:
            pass

        processor = FakeProcessor()
        tokenizer = FakeTokenizer()
        session = RealModelChatSession(
            model_path=Path("."),
            max_new_tokens=4,
            max_context_tokens=32,
            system_prompt=None,
            model=None,
            processor=processor,
            tokenizer=tokenizer,
            history=[{"role": "user", "content": "hello"}],
        )
        inputs = session._build_inputs()
        self.assertEqual(inputs, {"input_ids": "ids", "attention_mask": "mask"})
        self.assertTrue(processor.kwargs["tokenize"])
        self.assertTrue(processor.kwargs["return_dict"])
        self.assertEqual(processor.kwargs["return_tensors"], "pt")
        self.assertEqual(
            processor.kwargs["processor_kwargs"],
            {"truncation": True, "max_length": 32},
        )

    def test_real_chat_truncates_sequence_inputs(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values[0])) if values else (0,)

            def __getitem__(self, key):
                if isinstance(key, tuple) and key and key[0] is Ellipsis:
                    return FakeTensor([row[key[1]] for row in self.values])
                return FakeTensor(self.values[key])

        inputs = {
            "input_ids": FakeTensor([[1, 2, 3, 4, 5]]),
            "attention_mask": FakeTensor([[1, 1, 1, 1, 1]]),
            "other": "unchanged",
        }
        truncated = _truncate_inputs(inputs, 3)
        self.assertEqual(truncated["input_ids"].values, [[3, 4, 5]])
        self.assertEqual(truncated["attention_mask"].values, [[1, 1, 1]])
        self.assertEqual(truncated["other"], "unchanged")

    def test_real_chat_wraps_tokenizer_tensor_output(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, history, **kwargs):
                return "token-ids"

        tokenizer = FakeTokenizer()
        session = RealModelChatSession(
            model_path=Path("."),
            max_new_tokens=4,
            max_context_tokens=None,
            system_prompt=None,
            model=None,
            processor=tokenizer,
            tokenizer=tokenizer,
            history=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(session._build_inputs(), {"input_ids": "token-ids"})

    def test_real_chat_initializes_real_synthetic_model(self) -> None:
        try:
            import torch
            import transformers
        except ImportError:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "config.json").write_text(
                    json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
                    encoding="utf-8",
                )
                with self.assertRaises(RuntimeError) as context:
                    RealModelChatSession.from_model(root)
                self.assertIn("uv sync --extra chat", str(context.exception))
            return

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a, _ = generate_real_synthetic_models(root, seed=42)
            session = RealModelChatSession.from_model(
                model_a,
                max_new_tokens=4,
                device="cpu",
                dtype="float32",
            )
            self.assertEqual(session.model_path, model_a)
            self.assertEqual(session.max_new_tokens, 4)
            self.assertEqual(session.max_context_tokens, 32)
            self.assertIsNone(session.model.generation_config.max_length)

    def test_real_synthetic_models_are_transformers_style_gpt2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a, model_b = generate_real_synthetic_models(root, seed=42)
            for model in (model_a, model_b):
                self.assertTrue((model / "model.safetensors").is_file())
                self.assertTrue((model / "config.json").is_file())
                self.assertTrue((model / "tokenizer.json").is_file())
                self.assertTrue((model / "tokenizer_config.json").is_file())
                self.assertTrue((model / "special_tokens_map.json").is_file())
                self.assertTrue((model / "generation_config.json").is_file())
                with open(model / "config.json", "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                self.assertEqual(config["model_type"], "gpt2")
                self.assertEqual(config["architectures"], ["GPT2LMHeadModel"])
                with open(model / "tokenizer.json", "r", encoding="utf-8") as handle:
                    tokenizer = json.load(handle)
                self.assertEqual(tokenizer["model"]["type"], "WordLevel")

            index_a = read_checkpoint_index(model_a / "model.safetensors")
            self.assertIn("wte.weight", index_a.tensors)
            self.assertIn("h.0.attn.c_attn.weight", index_a.tensors)
            self.assertIn("ln_f.weight", index_a.tensors)

            diff = compare_models(model_a, model_b)
            self.assertFalse(diff.identical)
            self.assertGreater(sum(tensor.status == "changed" for tensor in diff.tensors), 0)

    def test_synthetic_generation_is_deterministic(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            model_a1, model_b1 = generate_synthetic_models(first, seed=123)
            model_a2, model_b2 = generate_synthetic_models(second, seed=123)
            hash_a1 = hashlib.sha256((model_a1 / "model.safetensors").read_bytes()).hexdigest()
            hash_a2 = hashlib.sha256((model_a2 / "model.safetensors").read_bytes()).hexdigest()
            hash_b1 = hashlib.sha256((model_b1 / "model.safetensors").read_bytes()).hexdigest()
            hash_b2 = hashlib.sha256((model_b2 / "model.safetensors").read_bytes()).hexdigest()
            self.assertEqual(hash_a1, hash_a2)
            self.assertEqual(hash_b1, hash_b2)

    def test_bf16_values_round_trip(self) -> None:
        from weight_writer.safetensors import iter_decoded_chunks, read_checkpoint_index

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bf16.safetensors"
            write_safetensors(path, {"x": ("BF16", [2], [1.0, -0.5])})
            index = read_checkpoint_index(path)
            values = next(iter_decoded_chunks(path, index.tensors["x"]))
            self.assertEqual(values, (1.0, -0.5))


if __name__ == "__main__":
    unittest.main()
