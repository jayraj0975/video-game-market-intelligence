"""Facts about what produced a model or a report: code, data, library versions, feature schema.

Used by the training script (metrics.json, model_report.md) and by the serving build
(``app/model/serving_model.meta.json``), so both describe themselves the same way.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from config import ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def describe(features: list[str], *, data_files: dict[str, Path], extra: dict | None = None) -> dict:
    """Provenance record. ``data_files`` maps a label to a file whose SHA-256 is recorded."""
    return {
        "code_commit": _git("rev-parse", "HEAD"),
        # Only code and dependency changes count: a rebuild rewrites the reports themselves.
        "code_dirty": bool(_git("status", "--porcelain", "--", "src", "app/main.py",
                                "app/service.py", "requirements-lock.txt")),
        "data_sha256": {label: sha256_file(p) for label, p in data_files.items()},
        "feature_schema_sha256": sha256_text("\n".join(features)),
        "features": features,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **(extra or {}),
    }


def check_pinned_clean_data() -> str:
    """Return the cleaned dataset's SHA-256, raising if it is not the one the reports were built from."""
    from config import CLEAN_CSV, CLEAN_DATA_SHA256

    digest = sha256_file(CLEAN_CSV)
    if digest != CLEAN_DATA_SHA256:
        raise RuntimeError(
            f"{CLEAN_CSV.name} has SHA-256 {digest}, expected {CLEAN_DATA_SHA256}. The source data "
            "changed, so the committed reports would not reproduce. Review the change, rebuild, and "
            "update CLEAN_DATA_SHA256 on purpose."
        )
    return digest
