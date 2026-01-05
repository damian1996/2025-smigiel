# Experiment Runner

This repository contains a simple setup for running model comparison experiments using a shared dataset and configurable parameters.

## Prerequisites

* Python 3.10+
* [`uv`](https://github.com/astral-sh/uv) installed

## Installation

Install all required dependencies using `uv`:

```bash
uv sync
```

This will create a virtual environment and install the exact dependency versions defined for the project.

## Running Experiments

Use the command below to run an experiment comparing two models on the training dataset:

```bash
uv run python3 main.py \
  --data-path data/train/data.tsv \
  --labels-path data/train/labels.tsv \
  --n-samples 1000 \
  --observer-model speakleash/Bielik-4.5B-v3 \
  --performer-model speakleash/Bielik-4.5B-v3.0-Instruct \
  --use-bfloat16 \
  --verbose
```

### Parameters

* `--data-path` – Path to the input data file (TSV format)
* `--labels-path` – Path to the labels file (TSV format)
* `--n-samples` – Number of samples to evaluate
* `--observer-model` – First model to evaluate
* `--performer-model` – Second model to evaluate
* `--use-bfloat16` – Enable bfloat16 precision for faster inference (if supported)
* `--verbose` – Enable detailed logging