# weight-writer

`weight-writer` compares two local Hugging Face-style model checkpoints stored as
`.safetensors` files. It does **not** load either model into PyTorch, Transformers,
or another inference framework. Instead, it reads the safetensors headers,
streams the tensor bytes, and produces:

- a machine-readable JSON diff,
- a human-readable HTML report,
- an SVG heatmap aggregated by model layer and parameter type.

The tool is useful for comparing two 1B–1.5B models, a base model and a fine-tune,
two checkpoints, or a model before and after an in-place weight edit.

## Install

```bash
uv sync
```

To enable real-model chat:

```bash
uv sync --extra chat
```

The `chat` extra includes `transformers`, `torch`, `accelerate`, `torchvision`, and `pillow`.

Or run it directly from a checkout:

```bash
PYTHONPATH=src .venv/bin/python -m weight_writer.cli --help
```

## Compare two models

Each input can be either a single `.safetensors` file or a directory containing
one or more `.safetensors` shards:

```bash
weight-writer compare model-a model-b
```

The command creates:

```text
weight-diff-report/
├── report.html
├── report.json
└── heatmap.svg
```

Open `report.html` in a browser for the full report.

### Useful options

Use a different output directory:

```bash
weight-writer compare model-a model-b --output-dir my-report
```

Choose the heatmap metric:

```bash
weight-writer compare model-a model-b --metric relative_l2
```

Available metrics are:

- `relative_l2`
- `mean_abs_diff`
- `max_abs_diff`
- `l2_diff`
- `changed_fraction`
- `bytes_changed_per_element`
- `bytes_changed`

Limit the HTML table to the most-changed tensors:

```bash
weight-writer compare model-a model-b --top 200
```

For very large checkpoints, use byte-only mode when exact numeric statistics are
not required:

```bash
weight-writer compare model-a model-b --fast
```

## List tensors

```bash
weight-writer index model-a
```

This prints each tensor's name, dtype, shape, element count, and source file.

## Generate synthetic models

By default, `weight-writer synthetic` now creates **real, Hugging Face
Transformers-compatible GPT-2 models**. They are tiny, but they use the normal
`GPT2LMHeadModel` architecture, safetensors weight names, and tokenizer files.
They can be loaded, trained, and fine-tuned with the standard Transformers
stack.

```bash
weight-writer synthetic --output-dir synthetic-models
```

This creates:

```text
synthetic-models/
├── model-a/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── model-b/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

Example Transformers usage:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("synthetic-models/model-a")
tokenizer = AutoTokenizer.from_pretrained("synthetic-models/model-a")
```

Because these are tiny random-weight models, their generated text is not
meaningful. They are intended for testing loading, training, fine-tuning, and
weight-diff tooling.

You can compare them directly:

```bash
weight-writer compare synthetic-models/model-a synthetic-models/model-b
```

### Legacy pure-Python synthetic models

If you want the older tiny custom models used by the built-in `generate` and
`chat` commands, pass `--legacy`:

```bash
weight-writer synthetic --legacy --output-dir synthetic-models
```

Those models use a custom architecture and a built-in pure-Python runner. They
are useful for lightweight testing, but they are not intended to be loaded by
Hugging Face Transformers.

## Chat with a model

For a legacy synthetic model:

```bash
weight-writer chat synthetic-models/model-a --max-new-tokens 8
```

For a real local model, `chat` automatically uses a Transformers-backed runner:

```bash
uv sync --extra chat
uv run weight-writer chat ./model-qwen35 --max-new-tokens 64
```

Useful real-model options:

```bash
uv run weight-writer chat ./model-qwen35 \
  --device auto \
  --dtype auto \
  --system "You are a concise assistant." \
  --max-new-tokens 128
```

The real-model runner supports both ordinary causal language models and
multimodal image-text-to-text models for text-only prompts.

## What the comparison reports

For each tensor with the same name, shape, and dtype, the numeric report includes:

- changed byte count,
- changed element count,
- mean absolute difference,
- maximum absolute difference,
- L2 difference,
- relative L2 difference,
- cosine similarity.

Tensors that exist in only one model are listed separately. Tensors with
different shapes or dtypes are marked as mismatches.

## Notes and limitations

- Only local `.safetensors` checkpoints are supported.
- The numeric path streams tensors in chunks and does not instantiate the model.
- Identical byte streams are detected before numeric decoding.
- `--fast` is much faster for billion-parameter models, but reports byte changes
  rather than numeric weight differences.
- Exotic dtypes such as float8 can be compared byte-wise, but numeric statistics
  are not currently decoded for them.
- Quantized container formats such as GGUF are not supported.
