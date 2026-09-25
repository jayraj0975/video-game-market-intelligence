"""Fetch the source dataset.

The dataset is a VGChartz sales scrape joined to Metacritic scores, originally
published on Kaggle as "Video Game Sales with Ratings". The data is not committed
(``data/`` is git-ignored); this script fetches it from public mirrors, and
``data_prep.py`` cleans it. The cleaned file is pinned by SHA-256 in
``config.CLEAN_DATA_SHA256``, and the unit tests need no data at all.

Run: ``python src/download_data.py``
"""

from __future__ import annotations

import io
import sys
import urllib.request

import pandas as pd

from config import DATA_DIR, RAW_CSV

# Mirrors of the same Kaggle dataset, tried in order. Each entry is a URL and
# the sheet name to read (None for a plain CSV).
SOURCES = [
    (
        "https://raw.githubusercontent.com/ethann-cao/PortfolioProjects/Master/"
        "VideoGameSales/Video_Games_Sales_02182024.xlsx",
        "Video_Games_Sales_as_at_22_Dec_",
    ),
    (
        "https://raw.githubusercontent.com/Maverick-512/gaming_data/main/"
        "Video_Games_Sales_as_at_22_Dec_2016.csv",
        None,
    ),
]

EXPECTED_COLUMNS = [
    "Name", "Platform", "Year_of_Release", "Genre", "Publisher",
    "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales",
    "Critic_Score", "Critic_Count", "User_Score", "User_Count",
    "Developer", "Rating",
]


def main() -> int:
    if RAW_CSV.exists():
        print(f"{RAW_CSV.name} already present -- nothing to do.")
        print("Delete it first if you want to re-fetch.")
        return 0

    df = None
    for url, sheet in SOURCES:
        print(f"fetching {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            candidate = (
                pd.read_excel(io.BytesIO(payload), sheet_name=sheet) if sheet
                else pd.read_csv(io.BytesIO(payload))
            )
        except Exception as exc:  # noqa: BLE001 - any failure, try the next mirror
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        missing = [c for c in EXPECTED_COLUMNS if c not in candidate.columns]
        if missing:
            print(f"  unexpected schema, missing: {missing}", file=sys.stderr)
            continue

        df = candidate
        break

    if df is None:
        print(
            "\nEvery mirror failed. The dataset is on Kaggle as 'Video Game Sales\n"
            f"with Ratings' -- download it and save it as {RAW_CSV}.",
            file=sys.stderr,
        )
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"wrote {RAW_CSV} ({len(df):,} rows)")
    if len(df) < 15_000:
        print(
            f"warning: only {len(df):,} rows -- this mirror looks truncated, and the\n"
            "numbers in reports/ will not match the committed ones.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
