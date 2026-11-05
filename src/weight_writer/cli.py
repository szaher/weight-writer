"""Command-line interface for weight-writer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .compare import compare_models
from .report import METRIC_LABELS, choose_metric, metric_value, write_html_report, write_json_report
from .safetensors import build_model_index
from .hf_synthetic import generate_real_synthetic_models
from .real_chat import RealModelChatSession
from .synthetic import SyntheticChatSession, generate_from_synthetic_model, generate_synthetic_models

_METRIC_CHOICES = ["auto", *METRIC_LABELS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weight-writer",
        description=(
            "Compare and visualize safetensors LLM checkpoints, generate synthetic models, "
            "and chat with local synthetic or real models."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two safetensors models and generate reports",
    )
    compare.add_argument("model_a", help="First model directory or .safetensors file")
    compare.add_argument("model_b", help="Second model directory or .safetensors file")
    compare.add_argument(
        "-o",
        "--output-dir",
        default="weight-diff-report",
        help="Directory for report.json, report.html, and heatmap.svg (default: weight-diff-report)",
    )
    compare.add_argument(
        "--metric",
        choices=_METRIC_CHOICES,
        default="auto",
        help="Heatmap metric; auto prefers relative L2, then progressively simpler metrics",
    )
    compare.add_argument(
        "--fast",
        action="store_true",
        help="Only check exact bytes; skip numeric difference statistics",
    )
    compare.add_argument(
        "--chunk-bytes",
        type=lambda value: _positive_int(value, "--chunk-bytes"),
        default=1024 * 1024,
        help="Tensor streaming chunk size in bytes (default: 1048576)",
    )
    compare.add_argument(
        "--top",
        type=lambda value: _positive_int(value, "--top"),
        default=None,
        help="Limit the HTML table to the N most-changed tensors",
    )
    compare.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-tensor progress on stderr",
    )
    compare.set_defaults(handler=_run_compare)

    index = subparsers.add_parser(
        "index",
        help="List tensors in a safetensors model",
    )
    index.add_argument("model", help="Model directory or .safetensors file")
    index.set_defaults(handler=_run_index)

    synthetic = subparsers.add_parser(
        "synthetic",
        help="Generate two tiny synthetic safetensors models",
    )
    synthetic.add_argument(
        "-o",
        "--output-dir",
        default="synthetic-models",
        help="Directory in which to create model-a and model-b (default: synthetic-models)",
    )
    synthetic.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic seed for the generated weights (default: 1337)",
    )
    synthetic.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Generate the older custom tiny models used by the built-in pure-Python "
            "generate/chat commands; by default this command now creates "
            "Transformers-compatible GPT-2 models"
        ),
    )
    synthetic.set_defaults(handler=_run_synthetic)

    generate = subparsers.add_parser(
        "generate",
        help="Generate text from a synthetic model",
    )
    generate.add_argument("model", help="Synthetic model directory")
    generate.add_argument("prompt", help="Space-separated prompt tokens")
    generate.add_argument(
        "--max-new-tokens",
        type=lambda value: _positive_int(value, "--max-new-tokens"),
        default=16,
        help="Number of tokens to generate (default: 16)",
    )
    generate.set_defaults(handler=_run_generate)

    chat = subparsers.add_parser(
        "chat",
        help="Chat interactively with a synthetic or real local model",
    )
    chat.add_argument("model", help="Synthetic or real model directory")
    chat.add_argument(
        "--max-new-tokens",
        type=lambda value: _positive_int(value, "--max-new-tokens"),
        default=64,
        help="Tokens to generate per reply (default: 64)",
    )
    chat.add_argument(
        "--device",
        default="auto",
        help="Real-model device: auto, cpu, cuda, or mps (default: auto)",
    )
    chat.add_argument(
        "--dtype",
        default="auto",
        help="Real-model dtype passed to Transformers (default: auto)",
    )
    chat.add_argument(
        "--system",
        default=None,
        help="Optional system prompt for real-model chat",
    )
    chat.set_defaults(handler=_run_chat)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _run_compare(args: argparse.Namespace) -> int:
    requested_metric = None if args.metric == "auto" else args.metric
    progress = None
    if not args.quiet:
        def progress(name: str, position: int, total: int) -> None:
            shown = name if len(name) <= 64 else name[:61] + "..."
            print(f"\r[{position}/{total}] {shown}", end="", file=sys.stderr, flush=True)

    try:
        diff = compare_models(
            args.model_a,
            args.model_b,
            fast=args.fast,
            chunk_bytes=args.chunk_bytes,
            progress=progress,
        )
        selected_metric = choose_metric(diff, requested_metric)
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "report.json")
        html_path = os.path.join(args.output_dir, "report.html")
        svg_path = os.path.join(args.output_dir, "heatmap.svg")
        write_json_report(diff, json_path)
        write_html_report(
            diff,
            html_path,
            metric=selected_metric,
            svg_path=svg_path,
            top_rows=args.top,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if progress is not None:
            print(file=sys.stderr)

    identical_count = sum(tensor.status == "identical" for tensor in diff.tensors)
    mismatch_count = sum(tensor.status == "shape_or_dtype_mismatch" for tensor in diff.tensors)
    changed_count = len(diff.tensors) - identical_count - mismatch_count
    print(f"Compared {len(diff.tensors):,} tensors")
    print(f"Identical: {identical_count:,}; changed: {changed_count:,}; mismatched: {mismatch_count:,}")
    if not diff.tensors:
        print("No tensors with matching names were found; the models are not directly comparable.")
    if diff.missing_in_a:
        print(f"Missing in A: {len(diff.missing_in_a):,}")
    if diff.missing_in_b:
        print(f"Missing in B: {len(diff.missing_in_b):,}")
    print(f"Reports: {html_path}")
    print(f"JSON: {json_path}")
    print(f"Heatmap: {svg_path}")

    ranked = [
        tensor
        for tensor in diff.tensors
        if metric_value(tensor, selected_metric) is not None
    ]
    ranked.sort(
        key=lambda tensor: (
            -(metric_value(tensor, selected_metric) or 0.0),
            tensor.name,
        )
    )
    if ranked:
        print("\nLargest differences:")
        for tensor in ranked[:5]:
            value = metric_value(tensor, selected_metric)
            print(f"  {value:.6g}  {tensor.name}")
    return 0


def _run_index(args: argparse.Namespace) -> int:
    try:
        model_index = build_model_index(args.model)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("name\tdtype\tshape\tnumel\tfile")
    for name in sorted(model_index):
        info, path = model_index[name]
        shape = "x".join(str(dimension) for dimension in info.shape)
        print(f"{name}\t{info.dtype}\t{shape}\t{info.numel}\t{path}")
    return 0


def _run_synthetic(args: argparse.Namespace) -> int:
    try:
        if args.legacy:
            model_a, model_b = generate_synthetic_models(args.output_dir, seed=args.seed)
            print("Generated legacy synthetic models:")
            print(f"Model A: {model_a}")
            print(f"Model B: {model_b}")
            print("Generate from model A with the built-in runner:")
            print(f"  weight-writer generate {model_a} alpha --max-new-tokens 8")
        else:
            model_a, model_b = generate_real_synthetic_models(args.output_dir, seed=args.seed)
            print("Generated Transformers-compatible GPT-2 models:")
            print(f"Model A: {model_a}")
            print(f"Model B: {model_b}")
            print("Load one with Transformers, for example:")
            print(f"  AutoModelForCausalLM.from_pretrained({str(model_a)!r})")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    try:
        output = generate_from_synthetic_model(
            args.model,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


def _run_chat(args: argparse.Namespace) -> int:
    is_legacy = _is_legacy_synthetic_model(args.model)
    try:
        if is_legacy:
            session = SyntheticChatSession.from_model(args.model)
            real_session = None
        else:
            session = None
            real_session = RealModelChatSession.from_model(
                args.model,
                max_new_tokens=args.max_new_tokens,
                system_prompt=args.system,
                device=args.device,
                dtype=args.dtype,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if is_legacy:
        print("Synthetic model chat")
        print("Type 'exit' or 'quit' to leave.")
        print(f"Vocabulary: {' '.join(session.runtime.vocab)}")
    else:
        print(f"Real model chat: {args.model}")
        print("Type 'exit' or 'quit' to leave.")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        try:
            if is_legacy:
                response = session.say(prompt, max_new_tokens=args.max_new_tokens)
            else:
                response = real_session.say(prompt)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue
        print(f"Model> {response}")
    return 0


def _is_legacy_synthetic_model(model_path: str) -> bool:
    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == "weight_writer_synthetic"


def _positive_int(value: str, option_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{option_name} must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
