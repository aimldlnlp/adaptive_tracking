PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: generate_data train evaluate all test

generate_data:
	$(PYTHON) scripts/generate_data.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

all:
	$(PYTHON) scripts/run_all.py --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -v
