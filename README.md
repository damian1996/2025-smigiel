## Install dependencies

uv sync

## Run experiments

uv run python3 main.py \
  --data-path data/train/data.tsv \
  --labels-path data/train/labels.tsv \
  --n-samples 2000 \
  --model-name-1 speakleash/Bielik-4.5B-v3 \
  --model-name-2 speakleash/Bielik-4.5B-v3.0-Instruct \
  --use-bfloat16 \
  --verbose
