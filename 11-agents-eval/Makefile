.PHONY: help install run run-mcp-bank dataset dataset-upload
.PHONY: test-deterministic test-llm-judge test-all

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "📋 Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	uv sync

run: ## Start the Telegram bot
	uv run python src/bot.py

run-mcp-bank: ## Start MCP bank agent server (port 8000)
	cd mcp/mcp-bank-agent && make run

dataset: ## Create QA dataset from documents
	uv run python src/dataset_synthesizer.py --create

dataset-upload: ## Upload dataset to LangSmith
	uv run python src/dataset_synthesizer.py --upload

# ============================================================
# TESTING COMMANDS
# ============================================================

test-deterministic: ## Run deterministic tests (match-based evaluators, fast)
	uv run pytest tests/test_agent_deterministic.py -v

test-llm-judge: ## Run LLM-as-Judge tests (slow, requires good model)
	uv run pytest tests/test_agent_llm_judge.py -v

test-all: ## Run all tests (deterministic + LLM-as-Judge)
	uv run pytest tests/ -v

