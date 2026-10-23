#!/usr/bin/env bash
# Full pipeline (classical part). Neural stages are opt-in because they need torch.
set -euo pipefail

python -m fakenews.data.download --config configs/base.yaml
python -m fakenews.data.eda      --config configs/base.yaml
python -m fakenews.data.clean    --config configs/base.yaml --csv
python -m fakenews.data.split    --config configs/base.yaml

python -m fakenews.train.classical --config configs/base.yaml --model lr
python -m fakenews.train.classical --config configs/base.yaml --model svm
python -m fakenews.train.shortcut  --config configs/shortcut.yaml

python -m fakenews.eval.evaluate --config configs/base.yaml
python -m fakenews.viz.figures   --config configs/base.yaml
