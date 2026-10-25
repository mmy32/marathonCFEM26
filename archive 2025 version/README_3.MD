# Unified Pipeline: Seniority + Sector

## What this does

This script reads a loan-level CSV and enriches it with:

- **instrument_seniority**: rules-based seniority classifier derived from `investment_identifier`
- **sector**: rules-based sector mapping with an OpenAI LLM step for unresolved names

It also produces coverage and distribution statistics and exports one final enriched CSV.

## Inputs

Your input CSV must include:

- `investment_identifier`

## Outputs

The script writes a single CSV (default name:  
**FINAL_CLEANED_DATA_with_seniority_and_sector.csv**)

The output includes:

- `instrument_seniority`
- `sector`
- All original columns, excluding intermediate sector helper columns

## Setup

### 1. Install dependencies

pip install -r requirements.txt

### 2. Set your OpenAI API key

export OPENAI_API_KEY="YOUR_KEY_HERE"

## Run

python get_seniority_debttype.py

## Configuration

You can control the LLM acceptance threshold and the cost/coverage tradeoff:

- `conf_thresh = 0.70` gives higher coverage but more risk of misclassification
- `conf_thresh = 0.90` is stricter (higher precision, lower coverage)

You can also tune:

- `llm_model` (default: gpt-4.1)
- `batch_size` (default: 32)

## Notes

- The rules-based sector step runs first and does not require the LLM
- The LLM is only called for borrower names still labeled as **Other / Unknown** by rules
- Final sector uses the LLM output only if `sector_llm_conf >= conf_thresh`
