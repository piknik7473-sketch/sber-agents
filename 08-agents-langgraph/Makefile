.PHONY: install run dataset dataset-upload

install:
	uv sync

run:
	PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv run python src/bot.py

dataset:
	uv run python src/dataset_synthesizer.py --create

dataset-upload:
	uv run python src/dataset_synthesizer.py --upload

