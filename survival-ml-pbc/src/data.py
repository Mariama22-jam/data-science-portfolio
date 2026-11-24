"""Load and prepare the Mayo Clinic PBC data (R package `survival`, dataset `pbc`).

Source: Therneau & Grambsch (2000), *Modeling Survival Data*, Appendix D; the
file ``data/raw/pbc.rda`` is taken unchanged from
https://github.com/therneau/survival (``data/pbc.rda``).

Event definition: death (status == 2). Transplant (status == 1) is treated as
censored at the transplant time, the standard choice in the literature.

Analysis set: the 312 patients in the randomised trial (``trt`` not missing).
The remaining 106 patients only have a few baseline measurements.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbc.rda"
PROCESSED = ROOT / "data" / "processed" / "pbc_trial.csv"
SOURCE_URL = "https://raw.githubusercontent.com/therneau/survival/master/data/pbc.rda"
SHA256 = "86ed4d908bccc801a01053b4e619a23e92d1b1bf8000a8429c7ced7632de44e2"

NUMERIC = ["age", "bili", "chol", "albumin", "copper", "alk.phos", "ast", "trig", "platelet", "protime"]
BINARY = ["sex_female", "ascites", "hepato", "spiders", "trt_dpca"]
ORDINAL = ["edema", "stage"]
LOG_VARS = ["bili", "chol", "copper", "alk.phos", "ast", "trig", "protime"]


def download(force: bool = False) -> Path:
    """Fetch the raw file if missing and verify its checksum."""
    if force or not RAW.exists():
        import urllib.request

        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != SHA256:
        print(f"warning: checksum of {RAW.name} differs from the pinned version "
              f"(upstream may have changed): {digest}")
    return RAW


def load_raw() -> pd.DataFrame:
    import pyreadr

    download()
    return pyreadr.read_r(str(RAW))["pbc"]


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d = d[d["trt"].notna()].copy()  # randomised trial patients
    for c in ["chol", "copper", "trig", "platelet", "stage", "ascites", "hepato", "spiders", "trt"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame(
        {
            "id": d["id"].astype(int),
            "time_days": d["time"].astype(float),
            "death": (d["status"] == 2).astype(bool),
            "transplant": (d["status"] == 1).astype(bool),
            "age": d["age"].astype(float),
            "sex_female": (d["sex"].astype(str) == "f").astype(float),
            "ascites": d["ascites"].astype(float),
            "hepato": d["hepato"].astype(float),
            "spiders": d["spiders"].astype(float),
            "trt_dpca": (d["trt"] == 1).astype(float),  # 1 = D-penicillamine, 2 = placebo
            "edema": d["edema"].astype(float),  # 0, 0.5, 1
            "stage": d["stage"].astype(float),
        }
    )
    for c in ["bili", "chol", "albumin", "copper", "alk.phos", "ast", "trig", "platelet", "protime"]:
        out[c] = d[c].astype(float).values
    return out.reset_index(drop=True)


def build_processed() -> pd.DataFrame:
    df = tidy(load_raw())
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    return df


def load_processed() -> pd.DataFrame:
    if not PROCESSED.exists():
        return build_processed()
    return pd.read_csv(PROCESSED)


def features(df: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    X = df[NUMERIC + BINARY + ORDINAL].copy()
    if log_transform:
        for c in LOG_VARS:
            X[c] = np.log(X[c])
    return X


def target(df: pd.DataFrame):
    from sksurv.util import Surv

    return Surv.from_arrays(event=df["death"].values, time=df["time_days"].values)


if __name__ == "__main__":
    d = build_processed()
    print(d.shape, "deaths:", int(d.death.sum()), "transplants:", int(d.transplant.sum()))
    print(d.isna().sum()[d.isna().sum() > 0])
