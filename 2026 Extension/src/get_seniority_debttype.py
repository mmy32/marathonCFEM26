"""

This script takes a raw loan-level CSV (must include `investment_identifier`),
adds:
  - `instrument_seniority` (rules-based seniority classifier)
  - `sector` (rules-based sector, then LLM for unresolved names above confidence threshold)


It exports a single final dataset for downstream index construction.


"""

import os
import json
import math
import csv
from typing import Optional, List, Dict, Any

import polars as pl
import anthropic


def build_final_dataset_with_seniority_and_sector(
    input_csv_path: str,
    output_csv_path: str = "FINAL_CLEANED_DATA_with_seniority_and_sector.csv",
    conf_thresh: float = 0.70,
    llm_model: str = "claude-haiku-4-5",
    batch_size: int = 32,
    anthropic_api_key: Optional[str] = None,
    anthropic_workspace_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Build a single enriched dataset that contains both:
      - instrument_seniority (seniority / debt type bucket)
      - sector (final sector after rules + optional LLM)

    Parameters
    ----------
    input_csv_path : str
        Path to the input CSV (e.g., "data_1105.csv").
        Must include column: `investment_identifier`.

    output_csv_path : str
        Path to save the final enriched dataset.

    conf_thresh : float
        Minimum LLM confidence required to accept LLM sector classification
        when rules-based sector is unresolved. Typical values: 0.70 or 0.90;
        the dataset shipped in data/processed/data_private_credit_FINAL_enriched.csv
        was built at 0.70 (default), which trades some precision for coverage.

    llm_model : str
        Claude model name used for LLM sector classification (e.g., "claude-haiku-4-5").

    batch_size : int
        Number of borrower names sent per LLM call.

    anthropic_api_key : Optional[str]
        Anthropic API key. If None, the function will read from environment variable ANTHROPIC_API_KEY.

    anthropic_workspace_id : Optional[str]
        Required only for identity-linked personal API keys (the Console will return a 400
        asking for "anthropic-workspace-id" if this is missing). If None, the function will
        read from environment variable ANTHROPIC_WORKSPACE_ID. Not needed for classic
        workspace-scoped API keys.

    checkpoint_path : Optional[str]
        If given, LLM classifications are appended to this CSV after every batch, and any
        borrower_name already present in it is skipped (not re-sent to the LLM) on a re-run.
        Lets a long run resume after an interruption instead of starting over. If None, no
        checkpointing is done (matches prior behavior).

    verbose : bool
        If True, prints shapes, head, coverage stats, and sample rows.

    Returns
    -------
    pl.DataFrame
        The final enriched DataFrame written to `output_csv_path`.
        (Returned in-memory for convenience.)
    """

    # ---------------------------------------------------------------------
    # 0) Load raw data (single read) + basic checks
    # ---------------------------------------------------------------------
    df = pl.read_csv(input_csv_path)

    if verbose:
        print("Raw df shape:", df.shape)
        print("Raw df columns:", df.columns)
        print("\nHead:")
        print(df.head(5))

    if "investment_identifier" not in df.columns:
        raise ValueError("Expected column 'investment_identifier' not found in df.")

    # ---------------------------------------------------------------------
    # 1) Seniority classification (Debt type rules)
    #   
    # ---------------------------------------------------------------------
    def classify_seniority_v2(text: str) -> str:
        if text is None:
            return "OTHER_UNKNOWN"

        t = text.lower()

        # CLO / structured equity
        if "collateralized loan obligation" in t or "clo subordinated" in t or "clo equity" in t:
            return "CLO_STRUCTURED_EQUITY"
        if "structured finance" in t and "membership interests" in t:
            return "CLO_STRUCTURED_EQUITY"

        # Subordinated / mezz / junior / holdco
        if "mezz" in t or "subordinated" in t or "junior debt" in t or "junior notes" in t:
            return "SUBORDINATED_MEZZ"
        if "holdco notes" in t or "holdco loan" in t or "holdco debt" in t:
            return "SUBORDINATED_MEZZ"
        if "pik toggle notes" in t:
            return "SUBORDINATED_MEZZ"

        # Unitranche / FO-LO
        if "unitranche" in t or "first out last out" in t or "first out / last out" in t or "fo/lo" in t:
            return "UNITRANCHE"

        # Second lien
        if "second lien" in t or "2nd lien" in t or "second-lien" in t or "junior secured" in t:
            return "SENIOR_SECURED_2L"

        # >>> UNSECURED BEFORE FIRST-LIEN/SECURED <<<
        if "unsecured" in t:
            return "SENIOR_UNSECURED"
        if ("senior notes" in t or "senior debt" in t) and "secured" not in t:
            return "SENIOR_UNSECURED"

        # First lien explicitly
        if "first lien" in t or "1st lien" in t:
            return "SENIOR_SECURED_1L"

        # Secured but lien not specified → separate bucket
        if (
            "secured" in t
            and "first lien" not in t
            and "1st lien" not in t
            and "second lien" not in t
            and "2nd lien" not in t
            and "junior" not in t
            and "mezz" not in t
        ):
            return "SENIOR_SECURED_UNSPEC"

        return "OTHER_UNKNOWN"

    # Keep old version if present
    if "instrument_seniority" in df.columns and "instrument_seniority_old" not in df.columns:
        df = df.with_columns(
            pl.col("instrument_seniority").alias("instrument_seniority_old")
        )

    # Apply classifier on investment_identifier
    df = df.with_columns(
        pl.col("investment_identifier")
        .map_elements(classify_seniority_v2, return_dtype=pl.String)
        .alias("instrument_seniority")
    )

    total_rows = df.height
    if verbose:
        print("\nTotal rows:", total_rows)

    # Classified vs unclassified (seniority rules)
    coverage_flag_v2 = (
        df
        .with_columns(
            pl.when(pl.col("instrument_seniority") == "OTHER_UNKNOWN")
            .then(pl.lit("Unclassified_by_seniority_rules"))
            .otherwise(pl.lit("Classified_by_seniority_rules"))
            .alias("seniority_flag_v2")
        )
        .group_by("seniority_flag_v2")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            (pl.col("n_rows") / total_rows).alias("row_share")
        )
        .sort("seniority_flag_v2")
    )

    if verbose:
        print("\n=== Coverage AFTER seniority rules (v3) ===")
        print(coverage_flag_v2)

    # Distribution by seniority bucket
    seniority_distribution_v2 = (
        df
        .group_by("instrument_seniority")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            (pl.col("n_rows") / total_rows).alias("row_share")
        )
        .sort("instrument_seniority")
    )

    if verbose:
        print("\n=== Distribution by instrument_seniority (v3) ===")
        print(seniority_distribution_v2)

        print("\nSample rows with instrument_seniority (v3):")
        print(
            df
            .select("investment_identifier", "instrument_seniority")
            .head(10)
        )

    # ---------------------------------------------------------------------
    # 2) Sector classification (Rules + LLM)
    
    # ---------------------------------------------------------------------

    # Sector universe 
    ALLOWED_SECTORS = [
        "Healthcare",
        "Technology",
        "Industrials",
        "Transportation",
        "Consumer",
        "Financials & Insurance",
        "Real Estate",
        "Communication & Media",
        "Energy & Utilities",
        "Materials",
        "Other / Unknown",  # includes unclear, unresolved, not confident
    ]

    def add_clean_names(df_in: pl.DataFrame) -> pl.DataFrame:
        """
        - borrower_name: copy of investment_identifier as main name field
        - inv_clean: lowercased, alphanumeric + spaces, for regex sector rules
        """
        return df_in.with_columns(
            pl.col("investment_identifier")
            .cast(pl.Utf8)
            .alias("borrower_name"),
            pl.col("investment_identifier")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", " ")
            .str.strip_chars()
            .alias("inv_clean"),
        )

    def sector_rule_expr() -> pl.Expr:
        """
        Rules-based classifier using inv_clean with a big set of keyword patterns.
        Returns a Polars expression that produces a 'sector_rule' column.
        """
        c = pl.col("inv_clean")

        # Structured / CLO etc. (will go into Financials & Insurance)
        structured_finance_pat = r"(collateralized loan obligation|clo subordinated notes|structured finance)"

        # Healthcare / Pharma
        healthcare_pat = (
            r"\b(healthcare|health care|clinic|clinics?|medical|hospital|hospitals|"
            r"dental|vision|pharma|pharmaceuticals?|biotech|biosciences?|"
            r"imaging|fertility|rx\b|pharmacy)\b"
        )

        # Technology / Software
        tech_pat = (
            r"\b(software|saas|technology|technologies|it services?|cyber|"
            r"cloud|data center|digital)\b"
        )

        # Financials / lenders / funds / banks / insurance / structured
        financials_ins_pat = (
            r"\b(bank|banking|financial|finance|funding|fund\b|funds\b|"
            r"capital partners|capital management|asset management|"
            r"insurance|brokerage)"
        )

        # Consumer staples (food, groceries ...)
        consumer_staples_pat = (
            r"\b(food|foods|beverage|beverages|brew|bakery|baking|"
            r"tobacco|grocery|grocer|supermarket|convenience store)\b"
        )

        # Consumer discretionary (restaurants, leisure, auto, retail ...)
        consumer_disc_pat = (
            r"\b(restaurant|restaurants|cafe|caf[eé]|hotel|resort|casino|gaming|"
            r"leisure|fitness|gym|auto dealer|car wash|cinema|theater|theatre|"
            r"retail|retailer|mall|shopping)\b"
        )

        # Extra catch-all for retail / wholesale
        retail_wholesale_pat = (
            r"\b(retail|retailer|wholesale|wholesaler|distributor)\b"
        )

        # Education (we fold into Consumer)
        education_pat = (
            r"\b(education|educational|school|university|college|learning|training)\b"
        )

        # Industrials / manufacturing / logistics
        industrials_pat = (
            r"\b(industrial|manufactur|factory|plant|engineering|equipment|"
            r"logistics|distribution|warehouse|trucking|freight|railroad|railway)\b"
        )

        # Transportation (airlines, shipping, etc.)
        transportation_pat = (
            r"\b(airline|airlines|airways|shipping|shipper|marine transport|"
            r"fleet services|logistics carrier)\b"
        )

        # Energy + Utilities
        energy_utils_pat = (
            r"\b(energy|power|electricity|solar|wind|renewable|"
            r"pipeline|oil\b|gas\b|utilities|water utility|electric utility|gas utility)\b"
        )

        # Materials / chemicals / packaging
        materials_pat = (
            r"\b(materials?|chemicals?|plastics?|rubber|packaging)\b"
        )

        # Real Estate
        real_estate_pat = (
            r"\b(real estate|properties|property|reit|apartment|apartments|"
            r"office park|logistics park|industrial park)\b"
        )

        # Communication + Media
        comms_media_pat = (
            r"\b(telecom|telecommunications?|cable tv|wireless carrier|"
            r"media|advertising|marketing agency|"
            r"entertainment|streaming|content studio|film studio|music label)\b"
        )

        # NOTE: order matters → more specific patterns first, then broader ones
        return (
            # Structured goes into Financials & Insurance
            pl.when(c.str.contains(structured_finance_pat))
            .then(pl.lit("Financials & Insurance"))

            .when(c.str.contains(healthcare_pat))
            .then(pl.lit("Healthcare"))

            .when(c.str.contains(tech_pat))
            .then(pl.lit("Technology"))

            .when(c.str.contains(financials_ins_pat))
            .then(pl.lit("Financials & Insurance"))

            .when(c.str.contains(transportation_pat))
            .then(pl.lit("Transportation"))

            # All consumer-like stuff in a single bucket
            .when(
                c.str.contains(consumer_staples_pat)
                | c.str.contains(consumer_disc_pat)
                | c.str.contains(retail_wholesale_pat)
                | c.str.contains(education_pat)
            )
            .then(pl.lit("Consumer"))

            .when(c.str.contains(industrials_pat))
            .then(pl.lit("Industrials"))

            .when(c.str.contains(energy_utils_pat))
            .then(pl.lit("Energy & Utilities"))

            .when(c.str.contains(materials_pat))
            .then(pl.lit("Materials"))

            .when(c.str.contains(real_estate_pat))
            .then(pl.lit("Real Estate"))

            .when(c.str.contains(comms_media_pat))
            .then(pl.lit("Communication & Media"))

            # Fallback: goes directly to Other / Unknown
            .otherwise(pl.lit("Other / Unknown"))
            .alias("sector_rule")
        )

    # Apply keyword rules
    df = add_clean_names(df)
    df = df.with_columns(sector_rule_expr())

    total_rows = df.height
    if verbose:
        print(f"\nTotal rows in df: {total_rows}")

    # Coverage after RULES ONLY
    coverage_rules = (
        df
        .with_columns(
            pl.when(pl.col("sector_rule") == pl.lit("Other / Unknown"))
            .then(pl.lit("Other / Unknown"))
            .otherwise(pl.lit("Classified_by_rules"))
            .alias("rules_flag")
        )
        .group_by("rules_flag")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            (pl.col("n_rows") / pl.lit(total_rows)).alias("row_share")
        )
    )

    if verbose:
        print("\n=== Coverage AFTER keyword rules ===")
        print(coverage_rules)

    # Per-sector distribution (rules)
    sector_stats_rules = (
        df
        .group_by("sector_rule")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            (pl.col("n_rows") / pl.lit(total_rows)).alias("row_share")
        )
        .sort("sector_rule")
    )

    if verbose:
        print("\n=== Sector distribution (sector_rule) ===")
        print(sector_stats_rules)

    # Prep rows for LLM: those still in Other / Unknown
    unresolved_for_llm = (
        df
        .filter(pl.col("sector_rule") == "Other / Unknown")
        .select("borrower_name")
        .unique()
        .drop_nulls()
        .sort("borrower_name")
    )

    names_list = unresolved_for_llm["borrower_name"].to_list()
    if verbose:
        print(f"\nNumber of UNIQUE borrower_name to send to LLM: {len(names_list)}")

    # Anthropic client (key handling)
    if anthropic_api_key is None:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_api_key == "":
        raise ValueError(
            "ANTHROPIC_API_KEY is missing. Set environment variable ANTHROPIC_API_KEY "
            "or pass anthropic_api_key=... to the function."
        )

    if anthropic_workspace_id is None:
        anthropic_workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "")

    default_headers = (
        {"anthropic-workspace-id": anthropic_workspace_id} if anthropic_workspace_id else None
    )
    client = anthropic.Anthropic(api_key=anthropic_api_key, default_headers=default_headers)

    # Deduplicate borrower names
    names_unique = sorted(set(names_list))
    if verbose:
        print(f"Total unclassified rows (from Cell 2)   : {len(names_list)}")
        print(f"Unique unclassified borrower_name count : {len(names_unique)}")

    # SYSTEM PROMPT (kept identical)
    SYSTEM_PROMPT = """
You help a private credit investor classify borrowers into economic sectors.

ALLOWED SECTORS (choose exactly one per borrower, and reproduce the label EXACTLY as written below):
- Healthcare
- Technology
- Industrials
- Transportation
- Consumer
- Financials & Insurance
- Real Estate
- Communication & Media
- Energy & Utilities
- Materials
- Other / Unknown

For EACH borrower, do:
1. Infer the most likely sector from the allowed list.
2. Assign a confidence score in [0, 1] (higher when you are very sure).
3. Be conservative: if you are not reasonably sure, use
   "Other / Unknown" with low confidence.
4. You may use your own general knowledge about companies and sectors.
5. You have to be precise.
6. Ignore all references to financing instruments such as “loan”, “term loan”,
“senior secured”, “first lien”, “revolver”, “facility”, “credit agreement”,
“secured debt”, “debtor”, or any similar wording. These describe the TYPE OF
FINANCING, not the company’s business sector. Do NOT use them when deciding
the sector.

IMPORTANT OUTPUT FORMAT:
- I will give you a list of borrower names.
- You must output EXACTLY ONE LINE per borrower, in the SAME ORDER.
- Each line must be:

  <sector_label> || <confidence>

  where:
  - <sector_label> is EXACTLY one of the allowed sectors above.
  - <confidence> is a float between 0 and 1.

- Do NOT add any extra text, explanations, numbering, or JSON.
- Only the lines with "<sector_label> || <confidence>".
"""

    def _build_prompt_for_batch(borrower_names: List[str]) -> str:
        """Build the user message listing borrowers in order."""
        header = "Classify the following borrowers into sectors.\n\nBorrowers:\n"
        borrower_lines = "\n".join(f"- {name}" for name in borrower_names)
        prompt = (
            header
            + borrower_lines
            + "\n\nOutput one line per borrower, in the SAME ORDER:\n<sector_label> || <confidence>\n"
        )
        return prompt

    def classify_batch_with_llm(
        borrower_names: List[str],
        model: str = "claude-haiku-4-5"
    ) -> List[Dict[str, Any]]:
        """
        Call Claude for a batch of borrower names.

        Output format:
          [
            {"borrower_name": ..., "sector_llm": ..., "sector_llm_conf": float},
            ...
          ]
        """
        if not borrower_names:
            return []

        user_prompt = _build_prompt_for_batch(borrower_names)

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0,  # deterministic / precise
            )
            content = "".join(
                block.text for block in resp.content if block.type == "text"
            )
        except Exception as e:
            print("Anthropic API error on this batch:", e)
            # Fallback: mark all as unresolved
            return [
                {
                    "borrower_name": name,
                    "sector_llm": "Other / Unknown / Unresolved",
                    "sector_llm_conf": 0.0,
                }
                for name in borrower_names
            ]

        # Parse lines of the form "<sector_label> || <confidence>"
        raw_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        results: List[Dict[str, Any]] = []
        for idx, name in enumerate(borrower_names):
            if idx < len(raw_lines):
                line = raw_lines[idx]
                if "||" in line:
                    parts = [p.strip() for p in line.split("||", maxsplit=1)]
                    label_raw = parts[0]
                    conf_raw = parts[1] if len(parts) > 1 else "0.0"
                else:
                    # If model forgot the "||", treat whole line as label
                    label_raw = line
                    conf_raw = "0.0"
            else:
                # Model produced fewer lines than borrowers
                label_raw = "Other / Unknown / Unresolved"
                conf_raw = "0.0"

            # Validate sector label against ALLOWED_SECTORS list
            # (kept identical to your logic)
            if label_raw not in ALLOWED_SECTORS:
                label = "Other / Unknown / Unresolved"
            else:
                label = label_raw

            # Parse confidence
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            results.append(
                {
                    "borrower_name": name,
                    "sector_llm": label,
                    "sector_llm_conf": conf,
                }
            )

        return results

    # Run LLM in batches, resuming from checkpoint_path if given
    checkpoint_fields = ["borrower_name", "sector_llm", "sector_llm_conf"]
    completed: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed[row["borrower_name"]] = {
                    "borrower_name": row["borrower_name"],
                    "sector_llm": row["sector_llm"],
                    "sector_llm_conf": float(row["sector_llm_conf"]),
                }
        if verbose:
            print(f"\nResuming from checkpoint: {len(completed)} borrowers already classified.")

    remaining_names = [name for name in names_unique if name not in completed]

    checkpoint_file = None
    checkpoint_writer = None
    if checkpoint_path:
        is_new_checkpoint = not os.path.exists(checkpoint_path)
        checkpoint_file = open(checkpoint_path, "a", newline="", encoding="utf-8")
        checkpoint_writer = csv.DictWriter(checkpoint_file, fieldnames=checkpoint_fields)
        if is_new_checkpoint:
            checkpoint_writer.writeheader()

    all_results: List[Dict[str, Any]] = list(completed.values())
    n = len(remaining_names)
    n_batches = math.ceil(n / batch_size) if batch_size > 0 else 0

    if verbose:
        print(f"\nRunning Claude LLM on {n} unique unclassified borrowers in {n_batches} batches...")

    try:
        for i in range(0, n, batch_size):
            batch = remaining_names[i: i + batch_size]
            if verbose:
                print(f"Classifying batch {i}–{i+len(batch)-1} / {n-1}")
            batch_res = classify_batch_with_llm(batch, model=llm_model)
            all_results.extend(batch_res)
            if checkpoint_writer:
                checkpoint_writer.writerows(batch_res)
                checkpoint_file.flush()
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    if verbose:
        print(f"\nTotal LLM classifications returned: {len(all_results)}")

    llm_df = pl.DataFrame(all_results)

    if verbose:
        print("\nHead of llm_df:")
        print(llm_df.head(10))

    # Join LLM results back to main df
    df = df.join(llm_df, on="borrower_name", how="left")

    if verbose:
        print("\nColumns after LLM join:", df.columns)
        print(
            "\nSample rows with rule-based sector + LLM suggestion:\n",
            df.select(["borrower_name", "sector_rule", "sector_llm", "sector_llm_conf"]).head(10)
        )

    # Final sector with threshold + coverage stats
    df = df.with_columns(
        [
            # Final sector:
            #   - If rules-based sector is not "Other / Unknown" → keep it
            #   - Else if LLM is confident enough → use LLM sector
            #   - Else → "Other / Unknown / Unresolved"
            pl.when(pl.col("sector_rule") != pl.lit("Other / Unknown"))
            .then(pl.col("sector_rule"))
            .otherwise(
                pl.when(
                    pl.col("sector_llm").is_not_null()
                    & (pl.col("sector_llm_conf") >= conf_thresh)
                )
                .then(pl.col("sector_llm"))
                .otherwise(pl.lit("Other / Unknown / Unresolved"))
            )
            .alias("sector_final"),

            # Keep source info (even if we don't use it in stats)
            pl.when(pl.col("sector_rule") != pl.lit("Other / Unknown"))
            .then(pl.lit("rules_based"))
            .otherwise(
                pl.when(
                    pl.col("sector_llm").is_not_null()
                    & (pl.col("sector_llm_conf") >= conf_thresh)
                )
                .then(pl.lit("llm_high_conf"))
                .otherwise(pl.lit("unresolved"))
            )
            .alias("sector_source"),
        ]
    )

    # Single sector column for downstream use
    df = df.with_columns(
        pl.col("sector_final").alias("sector")
    )

    # Coverage stats (final sector)
    total_rows = df.height
    if verbose:
        print(f"\nTotal rows in df: {total_rows}\n")

    classified_stats = (
        df.with_columns(
            pl.when(pl.col("sector") == pl.lit("Other / Unknown / Unresolved"))
            .then(pl.lit("Unclassified"))
            .otherwise(pl.lit("Classified"))
            .alias("classification_flag")
        )
        .group_by("classification_flag")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            (pl.col("n_rows") / total_rows).alias("row_share")
        )
        .sort("classification_flag")
    )

    if verbose:
        print("=== Classified vs Unclassified (FINAL sector) ===")
        print(classified_stats)

    sector_stats = (
        df
        .group_by("sector")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            (pl.col("n_rows") / total_rows).alias("row_share")
        )
        .sort("sector")
    )

    if verbose:
        print("\n=== Sector Breakdown (FINAL sector) ===")
        print(sector_stats)

        print("\nSample rows with final sector mapping:")
        print(
            df
            .select(
                "borrower_name",
                "sector_rule",
                "sector_llm",
                "sector_llm_conf",
                "sector",
                "sector_source",
            )
            .head(10)
        )

    # ---------------------------------------------------------------------
    # 3) Export single FINAL dataset (keep both sector + seniority columns)
    #    Here we only drop intermediate sector columns.
    # ---------------------------------------------------------------------
    cols_to_drop = [
        "sector_rule",
        "sector_llm",
        "sector_llm_conf",
        "sector_final",
        "sector_source",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df_export = df.drop(cols_to_drop)

    df_export.write_csv(output_csv_path)

    if verbose:
        print(f"\n✅ Exported final dataset with `sector` + `instrument_seniority` to: {output_csv_path}")
        print("Exported shape:", df_export.shape)
        print("\nHead of exported df (borrower_name + sector + instrument_seniority):")
        print(
            df_export
            .select(["borrower_name", "sector", "instrument_seniority"])
            .head(10)
        )

    return df_export


if __name__ == "__main__":
    # Example usage:
    # export ANTHROPIC_API_KEY="your_key_here"
    # export ANTHROPIC_WORKSPACE_ID="your_workspace_id_here"  # only needed for identity-linked keys
    build_final_dataset_with_seniority_and_sector(
        input_csv_path="data_1105.csv",
        output_csv_path="data_1105_FINAL_enriched.csv",
        conf_thresh=0.70,
        llm_model="claude-haiku-4-5",
        batch_size=32,
        anthropic_api_key=None,  # uses env var ANTHROPIC_API_KEY
        anthropic_workspace_id=None,  # uses env var ANTHROPIC_WORKSPACE_ID
        checkpoint_path="data_1105_sector_llm_checkpoint.csv",
        verbose=True,
    )