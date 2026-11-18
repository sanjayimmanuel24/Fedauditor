"""Download CDC BRFSS and build the cached, recoded CSV.

    python -m fedauditor.prepare_brfss --year 2023
    python -m fedauditor.prepare_brfss --year 2023 --xpt path/to/LLCP2023.XPT

The download is the CDC's public XPT zip (~100 MB); the cache is written to
data/brfss/brfss_<year>.csv.gz with a JSON sidecar listing the source columns.
"""
from __future__ import annotations

import argparse
import os

from fedauditor import brfss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--xpt", help="use an XPT file already on disk")
    parser.add_argument("--keep-xpt", action="store_true", help="keep the unzipped XPT (~1 GB)")
    args = parser.parse_args()

    xpt = args.xpt or brfss.download_brfss(args.year)
    print(f"converting {xpt} ...")
    out = brfss.convert_xpt(xpt, args.year)
    if not args.xpt and not args.keep_xpt:
        os.remove(xpt)
    frame = brfss.load_brfss(args.year)
    print(f"wrote {out}: {len(frame)} complete cases, "
          f"{frame[brfss.CLIENT_COLUMN].nunique()} states, "
          f"prevalence {frame[brfss.TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
