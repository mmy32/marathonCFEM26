# Unified Pipeline: Seniority + Sector

## What this does

This script reads a loan-level CSV and enriches it with:

- **instrument_seniority**: rules-based seniority classifier derived from `investment_identifier`
- **sector**: rules-based sector mapping with a Claude LLM step for unresolved names

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

### 2. Set your Anthropic API key

export ANTHROPIC_API_KEY="YOUR_KEY_HERE"

If your key is an identity-linked personal API key (rather than a classic workspace-scoped
key), the API will reject requests with a 400 asking for a workspace id. In that case also set:

export ANTHROPIC_WORKSPACE_ID="YOUR_WORKSPACE_ID_HERE"

(Find this in the Anthropic Console under Settings → Workspaces. Not needed for classic keys.)

## Run

python get_seniority_debttype.py

Or from `index_construction_pipeline.ipynb` (the "Adding the Sector and Seniority using LLM"
section), which calls the same entry point:

```python
build_final_dataset_with_seniority_and_sector(
    input_csv_path=OUTPUT_PATH,
    output_csv_path=OUTPUT_PATH_LLM,
    conf_thresh=0.70,
    llm_model="claude-haiku-4-5",
    batch_size=32,
    anthropic_api_key=API_KEY,  # None => uses env var ANTHROPIC_API_KEY
    verbose=True,
)
```

This step was migrated from OpenAI to Claude. The parameter is `anthropic_api_key`, not
`openai_api_key`, and there is no OpenAI code path left - a call using the old keyword raises
`TypeError: unexpected keyword argument`.

## Configuration

You can control the LLM acceptance threshold and the cost/coverage tradeoff:

- `conf_thresh = 0.70` (default) gives higher coverage but more risk of misclassification - this is what produced `data/processed/data_private_credit_FINAL_enriched.csv`
- `conf_thresh = 0.90` is stricter (higher precision, lower coverage)

You can also tune:

- `llm_model` (default: claude-haiku-4-5)
- `batch_size` (default: 32)
- `checkpoint_path` (default: None) — if set, LLM results are appended to this CSV after every
  batch, and any borrower already in it is skipped on a re-run. Recommended for large datasets:
  a run of tens of thousands of names makes many sequential API calls and can take a while, so
  checkpointing lets you resume after an interruption instead of starting over and re-paying for
  batches you already ran.

## Notes

- The rules-based sector step runs first and does not require the LLM
- The LLM is only called for borrower names still labeled as **Other / Unknown** by rules
- Final sector uses the LLM output only if `sector_llm_conf >= conf_thresh`
