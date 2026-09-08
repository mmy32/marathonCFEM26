# src/paths.py
from pathlib import Path

# ========= Project root =========
ROOT = Path(__file__).resolve().parents[1]   # 2026 Extension/

# ========= Data =========
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ========= Cache (downloader outputs) =========
CACHE_DIR = ROOT / "cache"
METADATA_CSV = CACHE_DIR / "all_metadata.csv"
FAILURES_CSV = CACHE_DIR / "download_failures.csv"

def cik_cache_dir(cik: str) -> Path:
    """cache/{cik}/ (HTML filings live here)."""
    return CACHE_DIR / str(cik)

# ========= iXBRL outputs (new) =========
IXBRL_DIR = DATA_DIR / "ixbrl"
IXBRL_RAW_XML_DIR = IXBRL_DIR / "raw_xml"
IXBRL_PER_FILE_DIR = IXBRL_DIR / "per_file"

IXBRL_COMBINED_DIR = IXBRL_DIR / "combined"
IXBRL_WIDE_ALL_CSV = IXBRL_COMBINED_DIR / "wide_all.csv"

IXBRL_CLEANED_DIR = IXBRL_DIR / "cleaned"
IXBRL_PANEL_CSV = IXBRL_CLEANED_DIR / "panel_clean.csv"

IXBRL_METADATA_AUGMENTED_CSV = IXBRL_DIR / "metadata_augmented.csv"

def ensure_dirs() -> None:
    """Create standard project directories (safe to call repeatedly)."""
    for p in [
        DATA_DIR,
        PROCESSED_DIR,
        CACHE_DIR,
        IXBRL_RAW_XML_DIR,
        IXBRL_PER_FILE_DIR,
        IXBRL_COMBINED_DIR,
        IXBRL_CLEANED_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
