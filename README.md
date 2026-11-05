# weight-writer

A local toolkit for inspecting, comparing, visualizing, and chatting with Hugging Face-style safetensors model checkpoints.

`weight-writer` is designed for developers who need to understand what changed between two model checkpoints, generate tiny but real test models, or experiment with local model inference without setting up a full production serving stack.

---

## Why this tool exists

When working with language models, it is often useful to answer questions such as:

- Are two checkpoints byte-for-byte identical?
- Which tensors changed between a base model and a fine-tune?
- How large are the numerical differences?
- Are changes concentrated in particular layers or parameter types?
- Can I create a small, valid test model without downloading a large checkpoint?
- Can I load and chat with a real local model for quick experimentation?

`weight-writer` provides focused utilities for those tasks.

---

## Main features

### 1. Compare two safetensors models

Compare two local model directories or individual `.safetensors` files without loading either model into PyTorch or Transformers.

```bash
weight-writer compare model-a model-b
```

Each input can be:

- a model directory containing one or more `.safetensors` shards, or
- a single `.safetensors` file.

The command writes:

```text
weight-diff-report/
├── report.html
├── report.json
└── heatmap.svg
```

Open `report.html` in a browser for a human-readable report.

#### Useful options

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

For very large checkpoints, use byte-only mode when exact numeric statistics are not required:

```bash
weight-writer compare model-a model-b --fast
```

#### What the comparison reports

For each tensor with the same name, shape, and dtype, the numeric report includes:

- changed byte count,
- changed element count,
- mean absolute difference,
- maximum absolute difference,
- L2 difference,
- relative L2 difference,
- cosine similarity.

Tensors that exist in only one model are reported separately. Tensors with different shapes or dtypes are marked as mismatches.

#### Important limitation

The comparison matches tensors by exact safetensors tensor name.

Models with different architectures often use different tensor layouts and may have no directly comparable tensors, even if their parameter counts are similar.

---

### 2. Generate real Transformers-compatible synthetic models

Create two tiny but valid GPT-2 models:

```bash
weight-writer synthetic --output-dir synthetic-models
```

This produces:

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

These are real `GPT2LMHeadModel` checkpoints with standard tensor names and tokenizer files. They can be loaded, trained, and fine-tuned with the normal Hugging Face Transformers stack.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("synthetic-models/model-a")
tokenizer = AutoTokenizer.from_pretrained("synthetic-models/model-a")
```

Because the weights are randomly initialized, generated text will not be meaningful. These models are intended for testing loading, training, fine-tuning, and weight-diff tooling.

---

### 3. Generate legacy pure-Python synthetic models

The older custom synthetic models are still available:

```bash
weight-writer synthetic --legacy --output-dir synthetic-models
```

These use a small custom transformer-style architecture and a built-in pure-Python runner. They are useful for lightweight testing but are not intended to be loaded by Hugging Face Transformers.

Generate text from one:

```bash
weight-writer generate synthetic-models/model-a alpha --max-new-tokens 8
```

Example output:

```text
beta gamma delta epsilon zeta eta theta iota
```

---

### 4. Chat with a local model

`weight-writer chat` supports both:

- legacy synthetic models, and
- real local Transformers models.

#### Legacy synthetic model

```bash
weight-writer chat synthetic-models/model-a
```

#### Real local model

Install the optional chat dependencies:

```bash
uv sync --extra chat
```

Then run:

```bash
uv run weight-writer chat ./my-model
```

Useful options:

```bash
uv run weight-writer chat ./my-model \
  --device auto \
  --dtype auto \
  --system "You are a concise assistant." \
  --max-new-tokens 128
```

Supported devices include:

- `auto`
- `cpu`
- `cuda`
- `mps`

The real-model runner supports:

- ordinary causal language models,
- multimodal image-text-to-text models for text-only prompts,
- multi-turn conversation history,
- models with or without a chat template.

If a checkpoint does not provide a chat template, the runner falls back to a simple plain-text format.

---

### 5. List tensors in a checkpoint

```bash
weight-writer index model-a
```

This prints each tensor’s:

- name,
- dtype,
- shape,
- element count,
- source file.

---

## Installation

Install the base package:

```bash
uv sync
```

Install the optional real-model chat dependencies:

```bash
uv sync --extra chat
```

The `chat` extra includes:

- `transformers`
- `torch`
- `accelerate`
- `torchvision`
- `pillow`

Run directly from a checkout without installing the console script:

```bash
PYTHONPATH=src .venv/bin/python -m weight_writer --help
```

---

## Command overview

```text
weight-writer compare     Compare two safetensors models and generate reports
weight-writer index       List tensors in a safetensors model
weight-writer synthetic   Generate synthetic model checkpoints
weight-writer generate    Generate text from a legacy synthetic model
weight-writer chat        Chat with a synthetic or real local model
```

View all options:

```bash
weight-writer --help
weight-writer compare --help
weight-writer synthetic --help
weight-writer chat --help
```

---

## Testing

Run the test suite:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The tests cover:

- safetensors header parsing,
- tensor streaming and decoding,
- numeric and byte-level comparison,
- report generation,
- real and legacy synthetic model generation,
- deterministic checkpoint creation,
- real-model chat input construction,
- context-length handling.

---

## Current limitations

- Only local `.safetensors` checkpoints are supported for comparison.
- Quantized container formats such as GGUF are not supported.
- Exotic dtypes such as float8 can be compared byte-wise, but numeric statistics are not currently decoded for them.
- The comparison matches tensors by exact name and does not perform cross-architecture semantic matching.
- Real-model chat is intended for local testing, not production serving.
- Generated synthetic models have random weights and do not produce meaningful text unless trained.

---

## Design notes

The comparison tool intentionally avoids loading models into an inference framework. Instead, it:

1. reads each safetensors header,
2. builds an index of tensor names, shapes, dtypes, and byte offsets,
3. streams tensor data in chunks,
4. compares bytes and, when requested, decodes numeric values,
5. aggregates results into JSON, HTML, and SVG reports.

This makes it practical to inspect large checkpoints without requiring enough memory to instantiate the full model.

---
