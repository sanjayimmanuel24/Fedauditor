"""CDC BRFSS: download, recode and cache, with the US state as the client id.

BRFSS is the CDC's annual telephone health survey (~430k adults a year). Each
state runs its own sample, so a state is a natural federated client: real
differences in prevalence, demographics and missingness, and no synthetic split.

Pipeline (run once):  download_brfss(year) -> convert_xpt(...) -> cached CSV.
After that, ``load_brfss(year)`` reads the cached CSV.

Variable names drift between survey years (e.g. ``_RFDRHV7`` -> ``_RFDRHV8``),
so each feature lists candidate names and the first one present is used. The
chosen source column is recorded in the cache's sidecar JSON.
"""
from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR

BRFSS_DIR = os.path.join(DATA_DIR, "brfss")
TARGET = "Outcome"
CLIENT_COLUMN = "State"

STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY", 66: "GU", 72: "PR", 78: "VI",
}


# --- recoders: raw BRFSS codes -> clean values (NaN = unknown/refused) -------

def _map(codes: Dict[float, float], blank: float = np.nan) -> Callable[[pd.Series], pd.Series]:
    def recode(s: pd.Series) -> pd.Series:
        out = s.map(codes).astype(float)
        return out.where(s.notna(), blank)
    return recode


def _ordinal(lo: int, hi: int) -> Callable[[pd.Series], pd.Series]:
    def recode(s: pd.Series) -> pd.Series:
        return s.where((s >= lo) & (s <= hi)).astype(float)
    return recode


def _days(s: pd.Series) -> pd.Series:
    """PHYSHLTH/MENTHLTH: 1-30 days, 88 = none, 77/99 = unknown."""
    out = s.where((s >= 1) & (s <= 30)).astype(float)
    return out.mask(s == 88, 0.0)


def _bmi(s: pd.Series) -> pd.Series:
    """_BMI5 is BMI x 100 with two implied decimals."""
    return (s / 100.0).where((s > 1000) & (s < 9999))


YES1_NO2 = _map({1: 1.0, 2: 0.0})          # 1 = yes, 2 = no
RISK2_NO1 = _map({1: 0.0, 2: 1.0})         # calculated "risk factor" vars: 2 = yes

# (output name, candidate source columns, recoder, required)
FEATURE_SPECS: List[tuple] = [
    ("HighBP", ["_RFHYPE6", "_RFHYPE5"], RISK2_NO1, True),
    # Asked only of people whose cholesterol was ever checked; blank -> never told.
    ("HighChol", ["TOLDHI3", "TOLDHI2"], _map({1: 1.0, 2: 0.0}, blank=0.0), True),
    ("BMI", ["_BMI5"], _bmi, True),
    ("Smoker", ["SMOKE100"], YES1_NO2, True),
    ("Stroke", ["CVDSTRK3"], YES1_NO2, True),
    ("HeartDisease", ["_MICHD"], YES1_NO2, True),
    ("PhysActivity", ["_TOTINDA"], YES1_NO2, True),
    ("HvyAlcohol", ["_RFDRHV9", "_RFDRHV8", "_RFDRHV7"], RISK2_NO1, True),
    ("AnyHealthcare", ["_HLTHPL1", "_HLTHPLN"], YES1_NO2, True),
    ("GenHlth", ["GENHLTH"], _ordinal(1, 5), True),
    ("MentHlth", ["MENTHLTH"], _days, True),
    ("PhysHlth", ["PHYSHLTH"], _days, True),
    ("DiffWalk", ["DIFFWALK"], YES1_NO2, True),
    ("Sex", ["SEXVAR", "_SEX"], _map({1: 1.0, 2: 0.0}), True),   # 1 = male
    ("AgeGroup", ["_AGEG5YR"], _ordinal(1, 13), True),            # 1 = 18-24 ... 13 = 80+
    ("Education", ["EDUCA"], _ordinal(1, 6), True),
    ("Income", ["_INCOMG1", "_INCOMG"], _ordinal(1, 7), True),
    # Fruit/vegetable module runs in odd years only.
    ("Fruits", ["_FRTLT1A", "_FRTLT1"], YES1_NO2, False),
    ("Veggies", ["_VEGLT1A", "_VEGLT1"], YES1_NO2, False),
]

# DIABETE4: 1 yes; 2 yes, only in pregnancy; 3 no; 4 pre-diabetes; 7/9 unknown.
TARGET_SPEC = ("Outcome", ["DIABETE4", "DIABETE3"], _map({1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}))
STATE_SPEC = ("_STATE",)

# Attributes kept for subgroup audits (fairness, subgroup-bias faults).
SENSITIVE = ["Sex", "AgeGroup", "Income"]


def xpt_url(year: int) -> str:
    return f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"


def cache_path(year: int) -> str:
    return os.path.join(BRFSS_DIR, f"brfss_{year}.csv.gz")


def download_brfss(year: int = 2023, dest_dir: str = BRFSS_DIR) -> str:
    """Download and unzip the CDC XPT file. Returns the .XPT path."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"LLCP{year}XPT.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(xpt_url(year), zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.strip().upper().endswith(".XPT")]
        if not names:
            raise ValueError(f"{zip_path} contains no .XPT file")
        archive.extract(names[0], dest_dir)
    return os.path.join(dest_dir, names[0])


def _pick(columns: set, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def recode_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Map raw BRFSS columns to the FedAuditor schema.

    Returns the recoded frame (unknowns left as NaN) and the source column
    used for each output column.
    """
    columns = set(raw.columns)
    used: Dict[str, str] = {}
    out = pd.DataFrame(index=raw.index)

    if "_STATE" not in columns:
        raise KeyError("BRFSS frame has no _STATE column")
    out[CLIENT_COLUMN] = raw["_STATE"].map(lambda v: STATE_FIPS.get(int(v)) if pd.notna(v) else None)
    used[CLIENT_COLUMN] = "_STATE"

    name, candidates, recoder = TARGET_SPEC
    source = _pick(columns, candidates)
    if source is None:
        raise KeyError(f"no diabetes outcome column among {candidates}")
    out[name] = recoder(raw[source])
    used[name] = source

    for name, candidates, recoder, required in FEATURE_SPECS:
        source = _pick(columns, candidates)
        if source is None:
            if required:
                raise KeyError(f"required feature {name!r}: none of {candidates} present")
            continue
        out[name] = recoder(raw[source].astype(float))
        used[name] = source
    return out, used


def source_columns() -> List[str]:
    names = ["_STATE", *TARGET_SPEC[1]]
    for _, candidates, _, _ in FEATURE_SPECS:
        names.extend(candidates)
    return names


def convert_xpt(xpt_path: str, year: int, chunksize: int = 50_000) -> str:
    """Read the (large) XPT in chunks, recode, and cache as gzipped CSV."""
    wanted = set(source_columns())
    parts = []
    used: Dict[str, str] = {}
    with pd.read_sas(xpt_path, format="xport", chunksize=chunksize) as reader:
        for chunk in reader:
            keep = [c for c in chunk.columns if c in wanted]
            recoded, used = recode_frame(chunk[keep])
            parts.append(recoded)
    frame = pd.concat(parts, ignore_index=True)
    os.makedirs(BRFSS_DIR, exist_ok=True)
    out = cache_path(year)
    frame.to_csv(out, index=False, compression="gzip")
    with open(out.replace(".csv.gz", ".json"), "w", encoding="utf-8") as fh:
        json.dump({"year": year, "rows": len(frame), "source_columns": used}, fh, indent=2)
    return out


def load_brfss(year: int = 2023, path: Optional[str] = None, dropna: bool = True) -> pd.DataFrame:
    """Load the cached recoded BRFSS frame.

    ``dropna`` keeps complete cases only (the convention of the widely used
    CDC Diabetes Health Indicators extract). Missingness is studied in FedFault
    as a *planted* fault, so the clean baseline starts from complete cases.
    """
    path = path or cache_path(year)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it once with:\n"
            f"  python -m fedauditor.prepare_brfss --year {year}\n"
            f"(downloads {xpt_url(year)})"
        )
    frame = pd.read_csv(path)
    frame = frame[frame[CLIENT_COLUMN].notna()]
    if dropna:
        frame = frame.dropna()
    frame[TARGET] = frame[TARGET].astype(int)
    return frame.reset_index(drop=True)


def feature_names(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in (TARGET, CLIENT_COLUMN)]
