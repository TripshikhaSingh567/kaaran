.PHONY: help setup data train eval export test web clean

PYTHON = ./.venv/Scripts/python.exe
PIP = ./.venv/Scripts/pip.exe
NPM = npm

help:
	@echo "KAARAN - Causal World Model Prototype"
	@echo "Available commands:"
	@echo "  make setup   - Set up Python venv and install requirements"
	@echo "  make data    - Run simulator and generate datasets"
	@echo "  make train   - Train Forecast, Action, and SCM ensembles"
	@echo "  make eval    - Evaluate lanes, failure modes, and calibration"
	@echo "  make export  - Export model weights and evaluation JSONs"
	@echo "  make test    - Run test suite"
	@echo "  make all     - Run pipeline (data -> train -> eval -> export -> test)"
	@echo "  make web     - Launch Next.js web application"

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r python/requirements.txt
	cd web && $(NPM) install

data:
	$(PYTHON) python/data.py

train:
	$(PYTHON) python/train.py

eval:
	$(PYTHON) python/evaluate.py
	$(PYTHON) python/trustmap.py
	$(PYTHON) python/dowhy_check.py

export:
	$(PYTHON) python/export.py

test:
	$(PYTHON) -m pytest python/tests/

all: data train eval export test

web:
	cd web && $(NPM) run dev

clean:
	rm -rf .pytest_cache mlruns web/.next
