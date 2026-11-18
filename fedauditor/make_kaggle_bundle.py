"""Build the zip to upload as a private Kaggle dataset for the open-model audits.

    python -m fedauditor.make_kaggle_bundle            # -> results/kaggle/fedauditor_bundle.zip

Contents (under ``Final-Year/`` inside the zip): the root Python modules the
audit code imports, ``fedauditor/``, the cached BRFSS CSV + metadata, and any
audit-grid result files for resuming. Never included: ``.env`` (API keys),
``users.db``, ``venv/``, the 93 MB raw BRFSS zip.

Upload it at kaggle.com -> Datasets -> New Dataset (keep it private), then run
``notebooks/fedauditor_kaggle_qwen.ipynb`` with that dataset attached.
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_ROOT = {"app.py", "database.py", "seed_demo.py"}           # Flask app only
NEVER = (".env", "users.db")


def bundle_files(base: str = BASE_DIR, with_results: bool = True) -> list:
    files = [p for p in glob.glob(os.path.join(base, "*.py")) if os.path.basename(p) not in SKIP_ROOT]
    files += glob.glob(os.path.join(base, "fedauditor", "*.py"))
    files += glob.glob(os.path.join(base, "data", "brfss", "brfss_*.csv.gz"))
    files += glob.glob(os.path.join(base, "data", "brfss", "brfss_*.json"))
    if with_results:
        files += glob.glob(os.path.join(base, "results", "fedauditor", "audit_grid_*.jsonl"))
    files = sorted(set(files))
    bad = [f for f in files if os.path.basename(f) in NEVER]
    if bad:
        raise RuntimeError(f"refusing to bundle secrets: {bad}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "results", "kaggle", "fedauditor_bundle.zip"))
    ap.add_argument("--no-results", action="store_true", help="leave out existing audit_grid_*.jsonl")
    args = ap.parse_args()
    files = bundle_files(with_results=not args.no_results)
    if not any(f.endswith(".csv.gz") for f in files):
        raise SystemExit("BRFSS cache missing: run `python -m fedauditor.prepare_brfss --year 2023` first")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, os.path.join("Final-Year", os.path.relpath(f, BASE_DIR)).replace(os.sep, "/"))
    size = os.path.getsize(args.out) / 1e6
    print(f"{len(files)} files -> {os.path.relpath(args.out, BASE_DIR)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
