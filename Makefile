.PHONY: install validate split train evaluate repro mlflow-ui api test lint clean docker-build docker-run

install:
	pip install -r requirements.txt

validate:
	python main.py validate

split:
	python main.py split

train:
	python main.py train

evaluate:
	python main.py evaluate

repro:
	dvc repro

mlflow-ui:
	mlflow ui --backend-store-uri mlruns

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

lint:
	black --check src tests main.py
	flake8 src tests main.py
	isort --check-only src tests main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache reports/figures/* logs/*.log

docker-build:
	docker build -t seguro-antirobos-mlops:latest .

docker-run:
	docker run -p 8000:8000 seguro-antirobos-mlops:latest
