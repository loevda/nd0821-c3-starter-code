"""Clean the census dataset and save it to file

The raw ``data/census.csv`` uses ``", "`` as the field separator, which leaves
leading spaces in the column names (e.g. ``" workclass"``). This script removes
those stray spaces and saves the result to ``data/census_clean.csv``, leaving
the raw file untouched.

Usage::

    python clean_data.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_FILE = DATA_DIR / "census.csv"
CLEAN_FILE = DATA_DIR / "census_clean.csv"


def clean_data(
    raw_path: Path = RAW_FILE, clean_path: Path = CLEAN_FILE
) -> pd.DataFrame:
    """Read the raw census CSV, strip stray spaces, and save a cleaned copy.

    Args:
        raw_path: Path to the raw (messy) census CSV.
        clean_path: Path where the cleaned CSV will be written.

    Returns:
        The cleaned DataFrame.
    """
    data = pd.read_csv(raw_path)
    cleaned = data.copy()
    cleaned.columns = [c.strip() for c in cleaned.columns]
    for col in cleaned.select_dtypes("object").columns:
        cleaned[col] = cleaned[col].str.strip()
    cleaned.to_csv(clean_path, index=False)
    return cleaned


if __name__ == "__main__":
    result = clean_data()
    print(
        f"Saved {len(result)} rows x {len(result.columns)} cols "
        f"to {CLEAN_FILE}"
    )
    print("Columns:", result.columns.tolist())
