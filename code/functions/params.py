from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_param_values(param_path: str | Path) -> Dict[str, Any]:
    """
    Read *_param.csv (2-column: parameter name, value) and return a dict.

    Returns:
      params: dict with all raw parameters as keys, plus:
        - 'Area_km2' derived from 'area_m2' if present
        - 'HI' derived from heightMin/height10/height90 if present
        - 'df' the raw DataFrame (par, Value)

    Notes:
      - Numeric coercion is applied to Value (strings -> float, invalid -> NaN).
      - Parameter keys preserve original case from the file.
    """
    param_path = Path(param_path)

    par_df = pd.read_csv(param_path, sep=";", header=None, names=["par", "Value"])
    par_df["par"] = par_df["par"].astype(str).str.strip()

    if par_df["Value"].dtype == "object":
        par_df["Value"] = par_df["Value"].astype(str).str.replace(",", ".", regex=False)

    par_df["Value"] = pd.to_numeric(par_df["Value"], errors="coerce")

    params: Dict[str, Any] = {}

    for par, val in zip(par_df["par"], par_df["Value"]):
        if pd.notna(val):
            params[par] = float(val)

    if "area_m2" in params:
        params["Area_km2"] = params["area_m2"] / 1_000_000.0

    if all(k in params for k in ("heightMin", "height10", "height90")):
        h_min, h_10, h_90 = params["heightMin"], params["height10"], params["height90"]
        if (h_90 - h_min) != 0:
            params["HI"] = (h_10 - h_min) / (h_90 - h_min)

    params["df"] = par_df
    return params