install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m src.trainer

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ app/ tests/
	black --check src/ app/
	isort --check-only src/ app/

format:
	black src/ app/
	isort src/ app/

run:
	streamlit run app/main.py
