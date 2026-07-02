# Developer workflow — `make help` lists targets.

.PHONY: help install pipeline simulate features train evaluate explain test app docker

help:
	@echo "install    - install package + dev/app extras (editable)"
	@echo "pipeline   - run the full pipeline (simulate -> explain)"
	@echo "test       - run the pytest suite"
	@echo "app        - launch the Streamlit dashboard"
	@echo "docker     - build the inference image"

install:
	pip install -e ".[app,dev]"

pipeline:
	python -m gym_churn.cli all

simulate:
	python -m gym_churn.cli simulate

features:
	python -m gym_churn.cli features

train:
	python -m gym_churn.cli train

evaluate:
	python -m gym_churn.cli evaluate

explain:
	python -m gym_churn.cli explain

test:
	pytest

app:
	streamlit run app.py

docker:
	docker build -t gym-churn .
