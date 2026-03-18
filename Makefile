PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: generate_data train evaluate figures videos all

generate_data:
	$(PYTHON) scripts/generate_data.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

figures:
	$(PYTHON) scripts/make_figures.py --config $(CONFIG)

videos:
	$(PYTHON) scripts/make_videos.py --config $(CONFIG)

all:
	$(PYTHON) scripts/run_all.py --config $(CONFIG)
