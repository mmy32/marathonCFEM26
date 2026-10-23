# paths.py: every data, plot and code location. Lives in `src/1_data_processing/`;
# modules in other src/ folders and the entry points in scripts/ add that folder to sys.path to import it.
import sys
from pathlib import Path

# ========= Project root =========
ROOT = Path(__file__).resolve().parents[2]   # project root (src/1_data_processing/ -> project)
PATHS_DIR = Path(__file__).resolve().parent  # folder holding this file

# ========= Data =========
DATA_DIR = ROOT / "data"
# One folder per pipeline stage (see README.md):
REFERENCE_DIR   = DATA_DIR / "reference"
UNIVERSE_DIR    = DATA_DIR / "universe"
PARSED_DIR      = DATA_DIR / "parsed"
CLEANED_DIR     = DATA_DIR / "cleaned"
ENRICHED_DIR    = DATA_DIR / "enriched"
FINAL_DIR       = DATA_DIR / "final_dataset"
DATA_CENTER_DIR = DATA_DIR / "data_center"

# ========= Plots =========
FIGURES_DIR = ROOT / "outputs"
INDEX_PLOTS_DIR         = FIGURES_DIR / "index_construction_plots"
SUBINDEX_PLOTS_DIR      = FIGURES_DIR / "sector_subindices_plots"
CONCENTRATION_PLOTS_DIR = FIGURES_DIR / "concentration_crowding_plots"
EDA_PLOTS_DIR           = FIGURES_DIR / "eda_plots"
PIK_PLOTS_DIR           = FIGURES_DIR / "pik_plots"
PIK_DIR                 = FIGURES_DIR / "pik"                # PIK summary tables
DATA_CENTER_PLOTS_DIR   = FIGURES_DIR / "data_center_plots"  # data center tables and charts
LOGS_DIR                = FIGURES_DIR / "logs"

# ========= Code folders (one per pipeline stage / analysis) =========
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"   # run_*.py entry points
CODE_DIRS = [
    SRC_DIR / "1_data_processing",
    SRC_DIR / "2_index_construction",
    SRC_DIR / "3_concentration_crowding",
    SRC_DIR / "4_val_dispersion",
    SRC_DIR / "5_pik",
    SRC_DIR / "6_data_center",
    SRC_DIR / "7_sector",
    SCRIPTS_DIR,
    FIGURES_DIR,   # outputs/exploratory_analysis.py
]

def add_code_dirs_to_path() -> None:
    """Make modules in every code folder importable by bare name (for notebooks
    and scripts that import across folders)."""
    for d in CODE_DIRS:
        if str(d) not in sys.path:
            sys.path.append(str(d))

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
        REFERENCE_DIR,
        UNIVERSE_DIR,
        PARSED_DIR,
        CLEANED_DIR,
        ENRICHED_DIR,
        FINAL_DIR,
        DATA_CENTER_DIR,
        FIGURES_DIR,
        INDEX_PLOTS_DIR,
        SUBINDEX_PLOTS_DIR,
        CONCENTRATION_PLOTS_DIR,
        EDA_PLOTS_DIR,
        PIK_PLOTS_DIR,
        PIK_DIR,
        DATA_CENTER_PLOTS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        IXBRL_RAW_XML_DIR,
        IXBRL_PER_FILE_DIR,
        IXBRL_COMBINED_DIR,
        IXBRL_CLEANED_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
