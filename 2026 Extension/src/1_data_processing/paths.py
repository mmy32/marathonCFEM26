"""Every data, output and code location. Modules in other src/ folders and the scripts in scripts/
add src/1_data_processing/ to sys.path to import it, then call add_code_dirs_to_path()."""
import sys
from pathlib import Path

# Print UTF-8 even when output goes to a file or pipe. On Windows that output defaults to cp1252,
# which cannot encode the box-drawing characters in Polars tables, and the scripts would crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower().replace("-", "") != "utf8":
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]   # project root (src/1_data_processing/ -> project)

# Data: one folder per pipeline stage (see README.md)
DATA_DIR = ROOT / "data"
REFERENCE_DIR   = DATA_DIR / "reference"
UNIVERSE_DIR    = DATA_DIR / "universe"
PARSED_DIR      = DATA_DIR / "parsed"
CLEANED_DIR     = DATA_DIR / "cleaned"
ENRICHED_DIR    = DATA_DIR / "enriched"
FINAL_DIR       = DATA_DIR / "final_dataset"
DATA_CENTER_DIR = DATA_DIR / "data_center"

# Outputs behind the midterm deck
FIGURES_DIR = ROOT / "outputs"
INDEX_PLOTS_DIR       = FIGURES_DIR / "index_construction_plots"
EDA_PLOTS_DIR         = FIGURES_DIR / "eda_plots"
DATA_CENTER_PLOTS_DIR = FIGURES_DIR / "data_center_plots"
RISK_MONITOR_DIR      = FIGURES_DIR / "risk_monitor"       # warning signs, exposure, stress events

# Code folders (one per pipeline stage / analysis)
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"   # run_*.py entry points
CODE_DIRS = [
    SRC_DIR / "1_data_processing",
    SRC_DIR / "2_index_construction",
    SRC_DIR / "3_concentration_crowding",
    SRC_DIR / "4_val_dispersion",
    SRC_DIR / "5_pik",
    SRC_DIR / "6_data_center",
    SRC_DIR / "8_risk_monitor",
    SCRIPTS_DIR,
]


def add_code_dirs_to_path() -> None:
    """Make modules in every code folder importable by bare name."""
    for d in CODE_DIRS:
        if str(d) not in sys.path:
            sys.path.append(str(d))


# SEC downloads (step 2)
CACHE_DIR = ROOT / "cache"
METADATA_CSV = CACHE_DIR / "all_metadata.csv"
FAILURES_CSV = CACHE_DIR / "download_failures.csv"

# iXBRL parsing (step 3)
IXBRL_DIR = DATA_DIR / "ixbrl"
IXBRL_RAW_XML_DIR = IXBRL_DIR / "raw_xml"
IXBRL_PER_FILE_DIR = IXBRL_DIR / "per_file"
IXBRL_COMBINED_DIR = IXBRL_DIR / "combined"
IXBRL_WIDE_ALL_CSV = IXBRL_COMBINED_DIR / "wide_all.csv"


def ensure_dirs() -> None:
    """Create the standard project directories (safe to call repeatedly)."""
    for p in [DATA_DIR, REFERENCE_DIR, UNIVERSE_DIR, PARSED_DIR, CLEANED_DIR, ENRICHED_DIR, FINAL_DIR,
              DATA_CENTER_DIR, FIGURES_DIR, INDEX_PLOTS_DIR, EDA_PLOTS_DIR, DATA_CENTER_PLOTS_DIR,
              RISK_MONITOR_DIR, CACHE_DIR, IXBRL_RAW_XML_DIR, IXBRL_PER_FILE_DIR, IXBRL_COMBINED_DIR]:
        p.mkdir(parents=True, exist_ok=True)
