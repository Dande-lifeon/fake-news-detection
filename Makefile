PY ?= python
PYTHONPATH := .

.PHONY: install install-neural data download eda clean split classical shortcut source-cue collect-metrics eval figures test lint smoke all

install:
	$(PY) -m pip install -e ".[dev]"

install-neural:
	$(PY) -m pip install -e ".[dev,neural]"

download:
	$(PY) -m fakenews.data.download --config configs/base.yaml

eda:
	$(PY) -m fakenews.data.eda --config configs/base.yaml

clean:
	$(PY) -m fakenews.data.clean --config configs/base.yaml --csv

split:
	$(PY) -m fakenews.data.split --config configs/base.yaml

data: download clean split

classical:
	$(PY) -m fakenews.train.classical --config configs/base.yaml --model lr --final
	$(PY) -m fakenews.train.classical --config configs/base.yaml --model svm --final

shortcut:
	$(PY) -m fakenews.train.shortcut --config configs/shortcut.yaml --final

source-cue:
	$(PY) -m fakenews.experiments.source_cue --config configs/base.yaml --model lr
	$(PY) -m fakenews.experiments.source_cue --config configs/base.yaml --model svm

collect-metrics:
	$(PY) -m fakenews.eval.collect_metrics --config configs/base.yaml

bilstm:
	$(PY) -m fakenews.train.bilstm --config configs/bilstm.yaml --seed 42 --condition fusion

distilbert:
	$(PY) -m fakenews.train.distilbert --config configs/distilbert.yaml --seed 42 --condition fusion

eval:
	$(PY) -m fakenews.eval.evaluate --config configs/base.yaml

figures:
	$(PY) -m fakenews.viz.figures --config configs/base.yaml

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

smoke:
	$(PY) -m fakenews.train.classical --config configs/base.yaml --smoke

all: data classical shortcut source-cue collect-metrics
