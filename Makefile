.PHONY: requirements data train test clean

requirements:
	pip install -r requirements.txt

data:
	python src/data/make_dataset.py

features:
	python src/features/build_features.py

train:
	python src/models/train_model.py

test:
	pytest tests/

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
