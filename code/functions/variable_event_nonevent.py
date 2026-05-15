
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .params import load_param_values



NUMERIC_COLS = [
    "Prec_mm", "Temp_C", "Q_m3s", "Q_OF_m3s",
    "SubsurfaceDef_P_mm", "GrwPlus_P_mm", "SWE_P_mm", "Soilmoist_P_mm",
]


VAR_CODE_TO_COL = {
    "gw": None,
    "q": "Q_m3s",
    "qmm": None,
    "sm": "Soilmoist_P_mm",
    "of": "Q_OF_m3s",
    "pr": "Prec_mm",
    "tmp": "Temp_C",
    "swe": "SWE_P_mm",
}


DEFAULT_PRE_ANCHOR_FEATURES = {
    "gw":  {"anchor": True, "quantile": True, "pctcap": True,  "pctmax": True,  "rates_before": True, "peak_pre": True},
    "q":   {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": True, "peak_pre": True},
    "qmm": {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": True, "peak_pre": True},
    "sm":  {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": True, "peak_pre": True},
    "of":  {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": True, "peak_pre": True},
    "pr":  {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": True, "peak_pre": True},
    "swe": {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": True, "peak_pre": True},
    "tmp": {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True,  "rates_before": False, "peak_pre": False},
}


def _force_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _series_from_df(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = df[["Datetime", col]].dropna().sort_values("Datetime")
    if s.empty:
        return pd.Series(dtype=float)
    out = s.set_index("Datetime")[col]
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = pd.to_numeric(out, errors="coerce").dropna().sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).mean().sort_index()
    return out


def _nearest_at(s: pd.Series, t: pd.Timestamp, tol: pd.Timedelta) -> float:
    if s.empty:
        return np.nan
    s = s.sort_index()
    pos = s.index.get_indexer([pd.Timestamp(t)], method="nearest")[0]
    if pos == -1:
        return np.nan
    tn = s.index[pos]
    if abs(tn - pd.Timestamp(t)) > tol:
        return np.nan
    v = s.iloc[pos]
    return float(v) if np.isfinite(v) else np.nan


def _ecdf_sorted(series: pd.Series | np.ndarray) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals.sort()
    return vals


def _ecdf_u(sorted_vals: np.ndarray, x: float) -> float:
    if sorted_vals.size == 0 or not np.isfinite(x):
        return np.nan
    return float(np.searchsorted(sorted_vals, x, side="right") / sorted_vals.size)


def _ecdf_vec(sorted_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if sorted_vals.size == 0:
        return out
    m = np.isfinite(x)
    out[m] = np.searchsorted(sorted_vals, x[m], side="right") / sorted_vals.size
    return out


def _pct_of_max(x: float, ref: pd.Series | np.ndarray) -> float:
    if not np.isfinite(x):
        return np.nan
    arr = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    vmax = np.nanmax(arr)
    if not np.isfinite(vmax) or vmax == 0:
        return np.nan
    return float((x / vmax) * 100.0)


def _pct_of_absmax(x: float, ref: pd.Series | np.ndarray) -> float:
    if not np.isfinite(x):
        return np.nan
    arr = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax == 0:
        return np.nan
    return float((abs(x) / vmax) * 100.0)


def _rate_before(x0: float, xb: float, H: int) -> float:
    if not (np.isfinite(x0) and np.isfinite(xb) and H > 0):
        return np.nan
    return float((x0 - xb) / H)



def _safe_div(num: float, den: float) -> float:
    """Safe scalar division used for derived features."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return np.nan
    return float(num / den)


def _zscore_value(x: float, ref: pd.Series | np.ndarray) -> float:
    """Z-score of x relative to a full-series reference."""
    if not np.isfinite(x):
        return np.nan
    arr = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float((x - mu) / sd)


def _window_count_and_excess(
    s: pd.Series,
    threshold: float,
    ts0: pd.Timestamp,
    H: int,
) -> tuple[float, float]:
    """Count hours/samples above threshold and sum positive exceedance in [t0-H, t0]."""
    if s.empty or not np.isfinite(threshold) or H <= 0:
        return np.nan, np.nan
    w = s.loc[(s.index > ts0 - pd.Timedelta(hours=int(H))) & (s.index <= ts0)].dropna()
    if w.empty:
        return np.nan, np.nan
    vals = w.to_numpy(dtype=float)
    excess = np.maximum(vals - float(threshold), 0.0)
    return float(np.sum(vals > float(threshold))), float(np.sum(excess))


def _add_derived_features(
    row: Dict[str, Any],
    cache: "CatchmentCache",
    ts0: pd.Timestamp,
    lags_h: list[int],
) -> Dict[str, Any]:
    """
    Add derived hydrological features built from the base anchor/pre-anchor features.

    These are intentionally computed after the base row is created so they are
    available for both events and controls with identical column names.
    """
    gw0 = row.get("gw0", np.nan)
    sm0 = row.get("sm0", np.nan)
    total_storage = float(gw0 + sm0) if (np.isfinite(gw0) and np.isfinite(sm0)) else np.nan
    row["total_storage"] = total_storage
    row["storage_fraction"] = _safe_div(total_storage, cache.cap)

    gw_s = cache.series_map.get("gw", pd.Series(dtype=float))
    sm_s = cache.series_map.get("sm", pd.Series(dtype=float))
    if not gw_s.empty and not sm_s.empty:
        idx = gw_s.index.union(sm_s.index)
        total_s = (gw_s.reindex(idx) + sm_s.reindex(idx)).dropna().sort_index()
    else:
        total_s = pd.Series(dtype=float)

    if not total_s.empty:
        row["relative_storage"] = _safe_div(total_storage, float(total_s.max()))
        row["storage_anomaly"] = _zscore_value(total_storage, total_s)
    else:
        row["relative_storage"] = np.nan
        row["storage_anomaly"] = np.nan

    prpk_t = row.get("prpk_t", np.nan)
    for v in ["gw", "q", "qmm", "sm", "swe"]:
        delay_name = f"{v}_delay_vs_pr"
        abs_name = f"{v}_abs_delay_vs_pr"
        vpk_t = row.get(f"{v}pk_t", np.nan)
        delay = float(vpk_t - prpk_t) if (np.isfinite(vpk_t) and np.isfinite(prpk_t)) else np.nan
        row[delay_name] = delay
        row[abs_name] = abs(delay) if np.isfinite(delay) else np.nan

    for v in ["gw", "q", "qmm", "sm", "of", "pr", "swe"]:
        pk_t = row.get(f"{v}pk_t", np.nan)
        row[f"{v}_peak_before_flag"] = int(pk_t > 0) if np.isfinite(pk_t) else np.nan
        row[f"{v}_peak_after_flag"] = int(pk_t < 0) if np.isfinite(pk_t) else np.nan
        row[f"{v}_distance_to_peak"] = abs(float(pk_t)) if np.isfinite(pk_t) else np.nan

    gw_thr90 = float(gw_s.quantile(0.90)) if not gw_s.empty else np.nan
    pr_s = cache.series_map.get("pr", pd.Series(dtype=float))
    pr_thr90 = float(pr_s.quantile(0.90)) if not pr_s.empty else np.nan
    qmm_s = cache.series_map.get("qmm", pd.Series(dtype=float))
    qmm_thr90 = float(qmm_s.quantile(0.90)) if not qmm_s.empty else np.nan

    for H in lags_h:
        pr = row.get(f"prS{H}b", np.nan)
        qmm = row.get(f"qmmS{H}b", np.nan)
        swe_loss = row.get(f"sweLossS{H}b", np.nan)
        netS = row.get(f"netS{H}b", np.nan)

        row[f"runoff_ratio_{H}b"] = _safe_div(qmm, pr)
        row[f"infiltration_proxy_{H}b"] = (1.0 - row[f"runoff_ratio_{H}b"]) if np.isfinite(row[f"runoff_ratio_{H}b"]) else np.nan

        total_input = float(pr + swe_loss) if (np.isfinite(pr) and np.isfinite(swe_loss)) else np.nan
        row[f"rain_plus_melt_{H}b"] = total_input
        row[f"response_efficiency_{H}b"] = _safe_div(qmm, total_input)
        row[f"melt_fraction_{H}b"] = _safe_div(swe_loss, total_input)
        row[f"rain_fraction_{H}b"] = _safe_div(pr, total_input)

        row[f"precip_x_gw_{H}b"] = float(pr * gw0) if (np.isfinite(pr) and np.isfinite(gw0)) else np.nan
        row[f"precip_x_sm_{H}b"] = float(pr * sm0) if (np.isfinite(pr) and np.isfinite(sm0)) else np.nan
        row[f"netS_x_gw_{H}b"] = float(netS * gw0) if (np.isfinite(netS) and np.isfinite(gw0)) else np.nan

        row[f"gwr_norm_{H}b"] = _safe_div(row.get(f"gwr{H}b", np.nan), gw0)
        row[f"smr_norm_{H}b"] = _safe_div(row.get(f"smr{H}b", np.nan), sm0)
        row[f"qmmr_norm_{H}b"] = _safe_div(row.get(f"qmmr{H}b", np.nan), row.get("qmm0", np.nan))
        row[f"qr_norm_{H}b"] = _safe_div(row.get(f"qr{H}b", np.nan), row.get("q0", np.nan))

        for v in ["gw", "q", "qmm", "sm"]:
            r = row.get(f"{v}r{H}b", np.nan)
            before_flag = row.get(f"{v}_peak_before_flag", np.nan)
            after_flag = row.get(f"{v}_peak_after_flag", np.nan)
            row[f"{v}r{H}b_x_peak_before"] = float(r * before_flag) if (np.isfinite(r) and np.isfinite(before_flag)) else np.nan
            row[f"{v}r{H}b_x_peak_after"] = float(r * after_flag) if (np.isfinite(r) and np.isfinite(after_flag)) else np.nan

        row[f"netS_ratio_{H}b"] = _safe_div(netS, pr)
        row[f"netS_norm_storage_{H}b"] = _safe_div(netS, total_storage)
        row[f"netS_gradient_{H}b"] = _safe_div(netS, float(H))

        c, e = _window_count_and_excess(gw_s, gw_thr90, ts0, H)
        row[f"hours_gw_high_q90_{H}b"] = c
        row[f"gw_excess_q90_sum_{H}b"] = e

        c, e = _window_count_and_excess(pr_s, pr_thr90, ts0, H)
        row[f"hours_pr_high_q90_{H}b"] = c
        row[f"pr_excess_q90_sum_{H}b"] = e

        c, e = _window_count_and_excess(qmm_s, qmm_thr90, ts0, H)
        row[f"hours_qmm_high_q90_{H}b"] = c
        row[f"qmm_excess_q90_sum_{H}b"] = e

    lag_pairs = [(3, 24), (6, 24), (12, 24), (24, 72)]
    for a, b in lag_pairs:
        if a not in lags_h or b not in lags_h:
            continue
        row[f"pr_ratio_{a}_{b}b"] = _safe_div(row.get(f"prS{a}b", np.nan), row.get(f"prS{b}b", np.nan))
        row[f"netS_ratio_{a}_{b}b"] = _safe_div(row.get(f"netS{a}b", np.nan), row.get(f"netS{b}b", np.nan))
        row[f"gw_rate_ratio_{a}_{b}b"] = _safe_div(row.get(f"gwr{a}b", np.nan), row.get(f"gwr{b}b", np.nan))
        row[f"sm_rate_ratio_{a}_{b}b"] = _safe_div(row.get(f"smr{a}b", np.nan), row.get(f"smr{b}b", np.nan))
        row[f"qmm_rate_ratio_{a}_{b}b"] = _safe_div(row.get(f"qmmr{a}b", np.nan), row.get(f"qmmr{b}b", np.nan))

    return row


def _rolling_rate_series(s: pd.Series, H: int, tol: pd.Timedelta) -> pd.Series:
    if s.empty or H <= 0:
        return pd.Series(dtype=float)
    s = s.dropna().sort_index()
    idx = s.index
    prev = s.reindex(idx - pd.Timedelta(hours=int(H)), method="nearest", tolerance=tol)
    out = (s.to_numpy(dtype=float) - prev.to_numpy(dtype=float)) / float(H)
    r = pd.Series(out, index=idx)
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def _rolling_sum_series(s: pd.Series, H: int, tol: pd.Timedelta) -> pd.Series:
    if s.empty or H <= 0:
        return pd.Series(dtype=float)
    s = s.dropna().sort_index()
    idx = pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h"), freq="1h")
    s_reg = s.reindex(idx, method="nearest", tolerance=tol)
    roll = pd.to_numeric(s_reg, errors="coerce").rolling(window=int(H), min_periods=int(H)).sum()
    return roll.replace([np.inf, -np.inf], np.nan).dropna()


def _sum_before_from_roll(roll: pd.Series, t0: pd.Timestamp, tol: pd.Timedelta) -> float:
    return _nearest_at(roll, pd.Timestamp(t0).floor("h"), tol)


def _mask_season(idx: pd.DatetimeIndex, target_doys: list[int], season_days: int) -> np.ndarray:
    if len(idx) == 0 or not target_doys:
        return np.zeros(len(idx), dtype=bool)
    doys = idx.dayofyear.to_numpy()
    out = np.zeros(len(idx), dtype=bool)
    for td in target_doys:
        dist = np.minimum(np.abs(doys - td), 366 - np.abs(doys - td))
        out |= dist <= int(season_days)
    return out


def _build_gw_signal(df: pd.DataFrame, gw_def: str = "gw", gw_sm_weight: float = 1.0) -> tuple[pd.Series, str]:
    gw_def = (gw_def or "gw").lower().strip()
    if gw_def == "gw":
        return _series_from_df(df, "GrwPlus_P_mm"), "GrwPlus_P_mm"
    if gw_def == "gw_plus_sm":
        s_gw = _series_from_df(df, "GrwPlus_P_mm")
        s_sm = _series_from_df(df, "Soilmoist_P_mm")
        idx = s_gw.index.union(s_sm.index)
        sig = s_gw.reindex(idx) + float(gw_sm_weight) * s_sm.reindex(idx)
        sig = pd.to_numeric(sig, errors="coerce").dropna().sort_index()
        sig.name = "gw_signal"
        return sig, f"GrwPlus_P_mm + {gw_sm_weight}*Soilmoist_P_mm"
    raise ValueError("gw_def must be 'gw' or 'gw_plus_sm'")


def _build_qmm_series(q: pd.Series, area_km2: Optional[float]) -> tuple[pd.Series, str]:
    if q.empty:
        return pd.Series(dtype=float), "Q_m3s -> qmm unavailable"
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        return pd.Series(dtype=float), "Q_m3s -> qmm unavailable: invalid Area_km2"
    qmm = q * (3.6 / float(area_km2))
    qmm.name = "Q_mm"
    return qmm.dropna().sort_index(), f"Q_m3s * (3.6 / Area_km2), Area_km2={area_km2}"


def _nearest_peak_in_window(
    s: pd.Series,
    ts0: pd.Timestamp,
    window_before_h: int,
    window_after_h: int,
) -> tuple[pd.Timestamp | None, float]:
    """
    Find nearest peak in [ts0 - window_before_h, ts0 + window_after_h].

    This mirrors the older peak logic:
      - prefer local maxima inside the combined window
      - if no local maxima exist, fall back to the window maximum

    Returned timing in the feature row is computed as (ts0 - peak_time) in hours:
      + positive value = peak was before anchor
      0 = peak at anchor
      - negative value = peak is after anchor
    """
    if s.empty:
        return None, np.nan

    t_start = ts0 - pd.Timedelta(hours=int(window_before_h))
    t_end = ts0 + pd.Timedelta(hours=int(window_after_h))
    w = s.loc[(s.index >= t_start) & (s.index <= t_end)].dropna().sort_index()
    if w.empty:
        return None, np.nan

    times = list(w.index)
    vals = w.to_numpy(dtype=float)
    n = len(vals)

    cand_idx: list[int] = []
    if n == 1:
        cand_idx = [0]
    else:
        for i in range(n):
            if i == 0:
                if vals[i] >= vals[i + 1]:
                    cand_idx.append(i)
            elif i == n - 1:
                if vals[i] >= vals[i - 1]:
                    cand_idx.append(i)
            else:
                if (vals[i] >= vals[i - 1]) and (vals[i] >= vals[i + 1]) and (
                    (vals[i] > vals[i - 1]) or (vals[i] > vals[i + 1])
                ):
                    cand_idx.append(i)

    if not cand_idx:
        j = int(np.nanargmax(vals))
        return pd.Timestamp(times[j]), float(vals[j])

    best_i = None
    best_abs_h = None
    best_val = None
    for i in cand_idx:
        tpk = pd.Timestamp(times[i])
        abs_h = abs((ts0 - tpk).total_seconds()) / 3600.0
        vpk = float(vals[i])

        if (best_i is None) or (abs_h < best_abs_h) or (abs_h == best_abs_h and vpk > best_val):
            best_i = i
            best_abs_h = abs_h
            best_val = vpk

    return pd.Timestamp(times[best_i]), float(vals[best_i])


def _first_cross_time_in_pre_window(s: pd.Series, thr: float, ts0: pd.Timestamp, window_before_h: int) -> pd.Timestamp | None:
    if s.empty or not np.isfinite(thr):
        return None
    w = s.loc[(s.index >= ts0 - pd.Timedelta(hours=int(window_before_h))) & (s.index <= ts0)]
    hit = w[w >= thr]
    if hit.empty:
        return None
    return pd.Timestamp(hit.index[0])



@dataclass
class CatchmentCache:
    guid: str
    df: pd.DataFrame
    param_values: Dict[str, Any]
    area_km2: Optional[float]
    cap: float
    gw_def: str
    gw_sm_weight: float
    gw_sig_name: str
    qmm_sig_name: str
    tol: pd.Timedelta
    series_map: Dict[str, pd.Series]
    sorted_map: Dict[str, np.ndarray]
    rolling_rates: Dict[tuple[str, int], pd.Series]
    rolling_sums_b: Dict[tuple[str, int], pd.Series]


def build_catchment_cache(
    guid: str,
    sim_path: str | Path,
    param_path: str | Path,
    gw_def: str = "gw",
    gw_sm_weight: float = 1.0,
    lags_h: Optional[list[int]] = None,
    tol_minutes: int = 30,
) -> CatchmentCache:
    if lags_h is None:
        lags_h = [3, 6, 12, 24, 48, 72]
    lags_h = sorted(set(int(x) for x in lags_h))

    df = pd.read_csv(sim_path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    df = _force_numeric(df, NUMERIC_COLS)

    pv = load_param_values(param_path)
    area_km2 = pv.get("Area_km2")
    cap = float(df["SubsurfaceDef_P_mm"].iloc[0]) if "SubsurfaceDef_P_mm" in df.columns and len(df) else np.nan

    tol = pd.Timedelta(minutes=int(tol_minutes))

    gw, gw_name = _build_gw_signal(df, gw_def=gw_def, gw_sm_weight=gw_sm_weight)
    q = _series_from_df(df, "Q_m3s")
    qmm, qmm_name = _build_qmm_series(q, area_km2=area_km2)

    series_map: Dict[str, pd.Series] = {"gw": gw, "q": q, "qmm": qmm}
    for v, col in VAR_CODE_TO_COL.items():
        if v in {"gw", "q", "qmm"} or col is None:
            continue
        series_map[v] = _series_from_df(df, col)

    sorted_map = {k: _ecdf_sorted(v) for k, v in series_map.items()}

    rolling_rates: Dict[tuple[str, int], pd.Series] = {}
    rolling_sums_b: Dict[tuple[str, int], pd.Series] = {}

    for v, s in series_map.items():
        if not s.empty:
            for H in lags_h:
                rolling_rates[(v, H)] = _rolling_rate_series(s, H, tol=tol)

    for v in ["pr", "qmm"]:
        s = series_map.get(v, pd.Series(dtype=float))
        if not s.empty:
            for H in lags_h:
                rolling_sums_b[(v, H)] = _rolling_sum_series(s, H, tol=tol)

    return CatchmentCache(
        guid=guid,
        df=df,
        param_values=pv,
        area_km2=area_km2,
        cap=cap,
        gw_def=gw_def,
        gw_sm_weight=gw_sm_weight,
        gw_sig_name=gw_name,
        qmm_sig_name=qmm_name,
        tol=tol,
        series_map=series_map,
        sorted_map=sorted_map,
        rolling_rates=rolling_rates,
        rolling_sums_b=rolling_sums_b,
    )



def _candidate_table(
    cache: CatchmentCache,
    event_times: Iterable[pd.Timestamp],
    season_days: int,
    exclude_event_buffer_days: int,
    prsum_lag_h: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evts = [pd.to_datetime(t, errors="coerce") for t in event_times]
    evts = [t.floor("h") for t in evts if pd.notna(t)]
    if not evts:
        return pd.DataFrame(), pd.DataFrame()

    gw = cache.series_map["gw"]
    q = cache.series_map.get("q", pd.Series(dtype=float))
    pr_roll = _rolling_sum_series(cache.series_map.get("pr", pd.Series(dtype=float)), int(prsum_lag_h), cache.tol)

    if gw.empty:
        return pd.DataFrame(), pd.DataFrame()

    target_rows = []
    for et in evts:
        gw0 = _nearest_at(gw, et, cache.tol)
        q0 = _nearest_at(q, et, cache.tol) if not q.empty else np.nan
        prsum = _sum_before_from_roll(pr_roll, et, cache.tol) if not pr_roll.empty else np.nan
        target_rows.append({
            "event_time": et,
            "event_year": et.year,
            "event_doy": et.dayofyear,
            "gw0_event": gw0,
            "gwq_event": _ecdf_u(cache.sorted_map["gw"], gw0),
            "q0_event": q0,
            "qq_event": _ecdf_u(cache.sorted_map.get("q", np.array([])), q0),
            "prsum_event": prsum,
            "prsum_q_event": _ecdf_u(_ecdf_sorted(pr_roll), prsum),
        })
    targets = pd.DataFrame(target_rows)

    idx = gw.index
    season_mask = _mask_season(idx, [int(t.dayofyear) for t in evts], int(season_days))
    ok = pd.Series(season_mask, index=idx)

    buf = pd.Timedelta(days=int(exclude_event_buffer_days))
    for et in evts:
        ok.loc[(idx >= et - buf) & (idx <= et + buf)] = False

    pool_idx = idx[ok.values]
    if len(pool_idx) == 0:
        return targets, pd.DataFrame()

    cand = pd.DataFrame({"t_anchor": pool_idx})
    cand["year"] = cand["t_anchor"].dt.year
    cand["doy"] = cand["t_anchor"].dt.dayofyear
    cand["gw0_anchor"] = gw.reindex(pool_idx).to_numpy(dtype=float)
    cand["gwq_anchor"] = _ecdf_vec(cache.sorted_map["gw"], cand["gw0_anchor"].to_numpy(dtype=float))

    if not q.empty:
        q_pool = q.reindex(pool_idx)
        cand["q0_anchor"] = q_pool.to_numpy(dtype=float)
        cand["qq_anchor"] = _ecdf_vec(cache.sorted_map.get("q", np.array([])), cand["q0_anchor"].to_numpy(dtype=float))
    else:
        cand["q0_anchor"] = np.nan
        cand["qq_anchor"] = np.nan

    if not pr_roll.empty:
        pr_pool = pr_roll.reindex(pool_idx)
        cand["prsum_anchor"] = pr_pool.to_numpy(dtype=float)
        cand["prsum_q_anchor"] = _ecdf_vec(_ecdf_sorted(pr_roll), cand["prsum_anchor"].to_numpy(dtype=float))
    else:
        cand["prsum_anchor"] = np.nan
        cand["prsum_q_anchor"] = np.nan

    cand = cand.dropna(subset=["gw0_anchor"]).reset_index(drop=True)
    return targets, cand


def _candidate_errors(cand: pd.DataFrame, targets: pd.DataFrame, match_mode: str) -> pd.DataFrame:
    out = cand.copy()
    best_dist = np.full(len(out), np.inf)
    best_i = np.full(len(out), -1, dtype=int)

    for i, t in targets.reset_index(drop=True).iterrows():
        d_parts = {}

        if match_mode == "quantile":
            d_parts["gw"] = np.abs(out["gwq_anchor"].to_numpy(dtype=float) - float(t["gwq_event"]))
            d_parts["q"] = np.abs(out["qq_anchor"].to_numpy(dtype=float) - float(t["qq_event"]))
            d_parts["prsum"] = np.abs(out["prsum_q_anchor"].to_numpy(dtype=float) - float(t["prsum_q_event"]))
        else:
            def rel(a, b):
                a = np.asarray(a, dtype=float)
                b = float(b)
                denom = abs(b)
                if not np.isfinite(denom) or denom == 0:
                    return np.full(len(a), np.inf)
                return np.abs(a - b) / denom

            d_parts["gw"] = rel(out["gw0_anchor"], t["gw0_event"])
            d_parts["q"] = rel(out["q0_anchor"], t["q0_event"])
            d_parts["prsum"] = rel(out["prsum_anchor"], t["prsum_event"])

        d_ref = d_parts["gw"]
        m = np.isfinite(d_ref) & (d_ref < best_dist)
        best_dist[m] = d_ref[m]
        best_i[m] = i

        for key, d in d_parts.items():
            col = f"{key}_dist_to_event_{i}"
            out[col] = d

    for key in ["gw", "q", "prsum"]:
        dist_cols = [c for c in out.columns if c.startswith(f"{key}_dist_to_event_")]
        if dist_cols:
            arr = out[dist_cols].to_numpy(dtype=float)
            out[f"{key}_dist"] = np.nanmin(arr, axis=1)
            arg = np.nanargmin(np.where(np.isfinite(arr), arr, np.inf), axis=1)
            out[f"{key}_best_event_i"] = arg
        else:
            out[f"{key}_dist"] = np.inf
            out[f"{key}_best_event_i"] = -1

    out["both_dist"] = np.maximum(out["gw_dist"].to_numpy(dtype=float), out["prsum_dist"].to_numpy(dtype=float))
    out["both_best_event_i"] = out["gw_best_event_i"]

    out["all3_dist"] = np.maximum.reduce([
        out["gw_dist"].to_numpy(dtype=float),
        out["prsum_dist"].to_numpy(dtype=float),
        out["q_dist"].to_numpy(dtype=float),
    ])
    out["all3_best_event_i"] = out["gw_best_event_i"]

    out["both_qweighted_best_event_i"] = out["gw_best_event_i"]

    return out


def _decluster_from_ranked(
    ranked: pd.DataFrame,
    k: int,
    min_sep: pd.Timedelta,
    already: list[pd.Timestamp],
    rng: np.random.Generator,
    randomize_within_rank: bool = True,
) -> pd.DataFrame:
    if k <= 0 or ranked.empty:
        return ranked.iloc[0:0].copy()

    df = ranked.copy()
    if randomize_within_rank:
        df["_rand"] = rng.random(len(df))
        sort_cols = [c for c in ["match_tolerance_used", "match_distance", "_rand"] if c in df.columns]
        df = df.sort_values(sort_cols)
    else:
        sort_cols = [c for c in ["match_tolerance_used", "match_distance", "t_anchor"] if c in df.columns]
        df = df.sort_values(sort_cols)

    picked_rows = []
    selected = [pd.Timestamp(t).floor("h") for t in already]

    for _, r in df.iterrows():
        t = pd.Timestamp(r["t_anchor"]).floor("h")
        if all(abs(t - s) >= min_sep for s in selected):
            picked_rows.append(r)
            selected.append(t)
        if len(picked_rows) >= int(k):
            break

    if not picked_rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(picked_rows).drop(columns=["_rand"], errors="ignore")


def select_controls_progressive(
    cache: CatchmentCache,
    event_times: Iterable[pd.Timestamp],
    season_days: int = 30,
    exclude_event_buffer_days: int = 14,
    n_controls: int = 25,
    match_target: str = "gw",
    match_mode: str = "value_pct",
    value_tolerance_levels: Optional[list[float]] = None,
    quantile_tolerance_levels: Optional[list[float]] = None,
    prsum_lag_h: int = 72,
    q_weight: float = 0.10,
    decluster_days: int = 7,
    min_controls_per_year: int = 2,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Progressive matched non-event selection.

    Important behavior:
    - Starts at strictest tolerance.
    - Adds declustered controls tier by tier.
    - Does not use random fallback.
    - If still short after max tolerance, it keeps drawing the best remaining matched controls
      from the same already-searched candidate pool until n_controls is reached, while respecting declustering.
    """
    rng = np.random.default_rng(int(random_seed))
    match_target = (match_target or "gw").lower().strip()
    match_mode = (match_mode or "value_pct").lower().strip()

    if match_target not in {"gw", "prsum", "q", "both", "all3", "both_qweighted"}:
        raise ValueError("match_target must be one of: 'gw', 'prsum', 'q', 'both', 'all3', 'both_qweighted'")
    if match_mode not in {"value_pct", "quantile"}:
        raise ValueError("match_mode must be one of: 'value_pct', 'quantile'")

    if value_tolerance_levels is None:
        value_tolerance_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    if quantile_tolerance_levels is None:
        quantile_tolerance_levels = [0.005, 0.01, 0.02, 0.05, 0.10]

    tol_levels = value_tolerance_levels if match_mode == "value_pct" else quantile_tolerance_levels
    tol_levels = sorted(set(float(x) for x in tol_levels))

    targets, cand = _candidate_table(
        cache=cache,
        event_times=event_times,
        season_days=season_days,
        exclude_event_buffer_days=exclude_event_buffer_days,
        prsum_lag_h=prsum_lag_h,
    )

    empty_cols = [
        "GUID", "t_anchor", "rank", "group", "match_source", "match_tolerance_used",
        "match_tier", "match_distance", "gw0_anchor", "gwq_anchor", "prsum_anchor",
        "prsum_q_anchor", "q0_anchor", "qq_anchor", "matched_event_time",
    ]

    if targets.empty or cand.empty:
        return pd.DataFrame(columns=empty_cols)

    cand = _candidate_errors(cand, targets, match_mode=match_mode)

    q_weight = float(q_weight)
    if q_weight < 0:
        raise ValueError("q_weight must be >= 0")

    cand["both_qweighted_dist"] = (
        cand["both_dist"].to_numpy(dtype=float)
        + q_weight * cand["q_dist"].to_numpy(dtype=float)
    )
    cand["both_qweighted_best_event_i"] = cand["both_best_event_i"]

    dist_col = f"{match_target}_dist"
    cand["match_distance"] = cand[dist_col].to_numpy(dtype=float)
    cand = cand[np.isfinite(cand["match_distance"])].copy()

    if cand.empty:
        return pd.DataFrame(columns=empty_cols)

    min_sep = pd.Timedelta(days=int(decluster_days))
    selected = pd.DataFrame()
    selected_times: list[pd.Timestamp] = []

    def add_picks(pool: pd.DataFrame, k: int) -> None:
        nonlocal selected, selected_times
        if k <= 0 or pool.empty:
            return
        pool = pool[~pool["t_anchor"].isin(selected_times)].copy()
        picks = _decluster_from_ranked(pool, k=k, min_sep=min_sep, already=selected_times, rng=rng)
        if picks.empty:
            return
        selected = pd.concat([selected, picks], ignore_index=True)
        selected_times.extend([pd.Timestamp(t).floor("h") for t in picks["t_anchor"]])

    years = sorted(cand["year"].dropna().unique())
    for tier, tol in enumerate(tol_levels, start=1):
        if len(selected) >= int(n_controls):
            break
        tier_pool_all = cand[cand["match_distance"] <= tol].copy()
        tier_pool_all["match_tolerance_used"] = tol
        tier_pool_all["match_tier"] = tier
        tier_pool_all["match_source"] = match_target

        for year in years:
            if len(selected) >= int(n_controls):
                break
            n_this_year = int((selected["year"] == year).sum()) if not selected.empty else 0
            need_y = max(0, int(min_controls_per_year) - n_this_year)
            if need_y <= 0:
                continue
            pool_y = tier_pool_all[tier_pool_all["year"] == year]
            add_picks(pool_y, need_y)

    for tier, tol in enumerate(tol_levels, start=1):
        if len(selected) >= int(n_controls):
            break
        tier_pool = cand[cand["match_distance"] <= tol].copy()
        tier_pool["match_tolerance_used"] = tol
        tier_pool["match_tier"] = tier
        tier_pool["match_source"] = match_target
        add_picks(tier_pool, int(n_controls) - len(selected))

    if len(selected) < int(n_controls):
        extra = cand.copy()
        extra["match_tolerance_used"] = np.nan
        extra["match_tier"] = len(tol_levels) + 1
        extra["match_source"] = f"{match_target}_best_extra"
        add_picks(extra, int(n_controls) - len(selected))

    if selected.empty:
        return pd.DataFrame(columns=empty_cols)

    selected = selected.sort_values(["match_tier", "match_distance", "t_anchor"]).head(int(n_controls)).copy()
    selected = selected.reset_index(drop=True)
    selected["rank"] = np.arange(1, len(selected) + 1)
    selected["GUID"] = cache.guid
    selected["group"] = "matched"

    best_i_col = f"{match_target}_best_event_i"
    best_indices = selected[best_i_col].fillna(0).astype(int).clip(lower=0, upper=len(targets) - 1).to_numpy()
    targ = targets.reset_index(drop=True).iloc[best_indices].reset_index(drop=True)

    selected["matched_event_time"] = targ["event_time"].to_numpy()
    for c in ["gw0_event", "gwq_event", "prsum_event", "prsum_q_event", "q0_event", "qq_event"]:
        selected[c] = targ[c].to_numpy()

    selected["season_days"] = int(season_days)
    selected["exclude_event_buffer_days"] = int(exclude_event_buffer_days)
    selected["n_controls"] = int(n_controls)
    selected["match_target"] = match_target
    selected["match_mode"] = match_mode
    selected["value_tolerance_levels"] = ",".join(str(x) for x in tol_levels) if match_mode == "value_pct" else ""
    selected["quantile_tolerance_levels"] = ",".join(str(x) for x in tol_levels) if match_mode == "quantile" else ""
    selected["prsum_lag_h"] = int(prsum_lag_h)
    selected["q_weight"] = float(q_weight)
    selected["decluster_days"] = int(decluster_days)

    keep = [
        "GUID", "t_anchor", "rank", "group", "match_source", "match_tolerance_used",
        "match_tier", "match_distance", "year", "doy",
        "gw0_anchor", "gwq_anchor", "prsum_anchor", "prsum_q_anchor",
        "q0_anchor", "qq_anchor",
        "matched_event_time", "gw0_event", "gwq_event", "prsum_event",
        "prsum_q_event", "q0_event", "qq_event",
        "season_days", "exclude_event_buffer_days", "n_controls",
        "match_target", "match_mode", "value_tolerance_levels",
        "quantile_tolerance_levels", "prsum_lag_h", "q_weight", "decluster_days",
    ]
    return selected[keep].sort_values(["GUID", "rank"]).reset_index(drop=True)



def compute_episode_stats_pre_anchor(
    cache: CatchmentCache,
    t_anchor,
    is_event: int,
    window_before_h: int = 72,
    window_after_h: int = 72,
    lags_h: Optional[list[int]] = None,
    var_features: Optional[Dict[str, Dict[str, bool]]] = None,
    gw_act_q: float = 0.99,
    q_act_q: float = 0.99,
) -> Dict[str, Any]:
    if lags_h is None:
        lags_h = [3, 6, 12, 24, 48, 72]
    lags_h = sorted(set(int(x) for x in lags_h))
    feats = var_features or DEFAULT_PRE_ANCHOR_FEATURES

    t_anchor = pd.to_datetime(t_anchor).floor("h")
    dt = cache.df["Datetime"]
    if (dt == t_anchor).any():
        ts0 = t_anchor
    else:
        i = (dt - t_anchor).abs().idxmin()
        ts0 = pd.Timestamp(cache.df.loc[i, "Datetime"]).floor("h")

    row: Dict[str, Any] = {
        "GUID": cache.guid,
        "t0": ts0,
        "y": int(is_event),
        "wb": int(window_before_h),
        "wa": int(window_after_h),
        "lags": ",".join(str(x) for x in lags_h),
        "cap": cache.cap,
        "gwdef": cache.gw_def,
        "gwsig": cache.gw_sig_name,
        "qmm_sig": cache.qmm_sig_name,
        "A_km2": cache.area_km2,
        "HI": cache.param_values.get("HI"),
        "MRT": cache.param_values.get("MRT"),
        "meandailyT": cache.param_values.get("meandailyT"),
        "gw_act_q": float(gw_act_q),
        "q_act_q": float(q_act_q),
    }

    B = max([h for h in lags_h if h <= int(window_before_h)], default=None)

    for v, s in cache.series_map.items():
        cfg = feats.get(v, {"anchor": True, "quantile": True, "pctcap": False, "pctmax": True, "rates_before": False, "peak_pre": False})
        sorted_full = cache.sorted_map.get(v, np.array([]))

        if s.empty:
            continue

        x0 = _nearest_at(s, ts0, cache.tol)

        if cfg.get("anchor", False):
            row[f"{v}0"] = x0
        if cfg.get("quantile", False):
            row[f"{v}q"] = _ecdf_u(sorted_full, x0)
        if cfg.get("pctcap", False):
            row[f"{v}p"] = float((x0 / cache.cap) * 100.0) if (np.isfinite(x0) and np.isfinite(cache.cap) and cache.cap > 0) else np.nan
            vmax = float(s.max()) if not s.empty else np.nan
            row[f"{v}mxcap"] = float((vmax / cache.cap) * 100.0) if (np.isfinite(vmax) and np.isfinite(cache.cap) and cache.cap > 0) else np.nan
        if cfg.get("pctmax", False):
            row[f"{v}pmx"] = _pct_of_max(x0, s)

        if cfg.get("peak_pre", False):
            tpk, vpk = _nearest_peak_in_window(
                s,
                ts0,
                window_before_h=int(window_before_h),
                window_after_h=int(window_after_h),
            )
            row[f"{v}pk_t"] = float((ts0 - tpk).total_seconds() / 3600.0) if tpk is not None else np.nan
            row[f"{v}pk_0"] = vpk
            row[f"{v}pk_q"] = _ecdf_u(sorted_full, vpk)

        if cfg.get("rates_before", False):
            for H in lags_h:
                xb = _nearest_at(s, ts0 - pd.Timedelta(hours=int(H)), cache.tol)
                r_val_b = _rate_before(x0, xb, int(H))
                row[f"{v}r{H}b"] = r_val_b

                r_series = cache.rolling_rates.get((v, H), pd.Series(dtype=float))
                if np.isfinite(r_val_b) and not r_series.empty:
                    row[f"{v}rq{H}b"] = _ecdf_u(_ecdf_sorted(r_series), r_val_b)
                    r_pos = r_series[r_series > 0]
                    r_neg = r_series[r_series < 0]
                    row[f"{v}rqp{H}b"] = _ecdf_u(_ecdf_sorted(r_pos), r_val_b) if not r_pos.empty else np.nan
                    row[f"{v}rqn{H}b"] = _ecdf_u(_ecdf_sorted(r_neg), r_val_b) if not r_neg.empty else np.nan
                    row[f"{v}rpmx{H}b"] = _pct_of_absmax(r_val_b, r_series)
                else:
                    row[f"{v}rq{H}b"] = np.nan
                    row[f"{v}rqp{H}b"] = np.nan
                    row[f"{v}rqn{H}b"] = np.nan
                    row[f"{v}rpmx{H}b"] = np.nan

            if B is not None:
                rB = row.get(f"{v}r{B}b", np.nan)
                for H in lags_h:
                    rH = row.get(f"{v}r{H}b", np.nan)
                    row[f"{v}x{H}"] = float(rH - rB) if (np.isfinite(rH) and np.isfinite(rB)) else np.nan

    pr_s = cache.series_map.get("pr", pd.Series(dtype=float))
    qmm_s = cache.series_map.get("qmm", pd.Series(dtype=float))

    for H in lags_h:
        pr_roll = cache.rolling_sums_b.get(("pr", H), pd.Series(dtype=float))
        qmm_roll = cache.rolling_sums_b.get(("qmm", H), pd.Series(dtype=float))

        prSb = _sum_before_from_roll(pr_roll, ts0, cache.tol) if not pr_roll.empty else np.nan
        qSb = _sum_before_from_roll(qmm_roll, ts0, cache.tol) if not qmm_roll.empty else np.nan

        row[f"prS{H}b"] = prSb
        row[f"qmmS{H}b"] = qSb
        row[f"netS{H}b"] = float(prSb - qSb) if (np.isfinite(prSb) and np.isfinite(qSb)) else np.nan

        row[f"prSq{H}b"] = _ecdf_u(_ecdf_sorted(pr_roll), prSb) if (np.isfinite(prSb) and not pr_roll.empty) else np.nan
        row[f"prSpmx{H}b"] = _pct_of_max(prSb, pr_roll) if not pr_roll.empty else np.nan

        row[f"qmmSq{H}b"] = _ecdf_u(_ecdf_sorted(qmm_roll), qSb) if (np.isfinite(qSb) and not qmm_roll.empty) else np.nan
        row[f"qmmSpmx{H}b"] = _pct_of_max(qSb, qmm_roll) if not qmm_roll.empty else np.nan

        if not pr_roll.empty:
            w = pr_roll.loc[
                (pr_roll.index >= ts0 - pd.Timedelta(hours=int(window_before_h)) + pd.Timedelta(hours=max(0, int(H) - 1)))
                & (pr_roll.index <= ts0)
            ]
            row[f"prmr{H}b"] = float(w.max()) if not w.empty else np.nan
            row[f"prmrpmx{H}b"] = _pct_of_max(row[f"prmr{H}b"], pr_roll) if np.isfinite(row[f"prmr{H}b"]) else np.nan
        else:
            row[f"prmr{H}b"] = np.nan
            row[f"prmrpmx{H}b"] = np.nan

    swe = cache.series_map.get("swe", pd.Series(dtype=float))
    swe0 = _nearest_at(swe, ts0, cache.tol) if not swe.empty else np.nan
    for H in lags_h:
        row[f"sweLossS{H}b"] = np.nan
        row[f"sweLossSq{H}b"] = np.nan
        row[f"sweLossSpmx{H}b"] = np.nan
        if swe.empty or not np.isfinite(swe0):
            continue
        swe_b = _nearest_at(swe, ts0 - pd.Timedelta(hours=int(H)), cache.tol)
        if np.isfinite(swe_b):
            loss = float(max(0.0, float(swe_b) - float(swe0)))
            row[f"sweLossS{H}b"] = loss
            swer = cache.rolling_rates.get(("swe", H), pd.Series(dtype=float))
            if not swer.empty:
                loss_dist = pd.Series(np.maximum(0.0, -swer.to_numpy(dtype=float) * float(H)))
                row[f"sweLossSq{H}b"] = _ecdf_u(_ecdf_sorted(loss_dist), loss)
                row[f"sweLossSpmx{H}b"] = _pct_of_max(loss, loss_dist)

    row["hxl_pre"] = np.nan
    row["hxl_pre_gwt"] = np.nan
    row["hxl_pre_qt"] = np.nan
    row["hxl_pre_gwthr"] = np.nan
    row["hxl_pre_qthr"] = np.nan

    gw_s = cache.series_map.get("gw", pd.Series(dtype=float))
    q_s = cache.series_map.get("q", pd.Series(dtype=float))
    if not gw_s.empty and not q_s.empty:
        gw_thr = float(gw_s.quantile(float(gw_act_q)))
        q_thr = float(q_s.quantile(float(q_act_q)))
        row["hxl_pre_gwthr"] = gw_thr
        row["hxl_pre_qthr"] = q_thr

        t_gw = _first_cross_time_in_pre_window(gw_s, gw_thr, ts0, int(window_before_h))
        t_q = _first_cross_time_in_pre_window(q_s, q_thr, ts0, int(window_before_h))

        row["hxl_pre_gwt"] = float((ts0 - t_gw).total_seconds() / 3600.0) if t_gw is not None else np.nan
        row["hxl_pre_qt"] = float((ts0 - t_q).total_seconds() / 3600.0) if t_q is not None else np.nan
        if t_gw is not None and t_q is not None:
            row["hxl_pre"] = float((t_gw - t_q).total_seconds() / 3600.0)

    row = _add_derived_features(row=row, cache=cache, ts0=ts0, lags_h=lags_h)
    return row


def build_episodes_fast(
    valid: pd.DataFrame,
    simres_paths: Dict[str, Path],
    param_paths: Dict[str, Path],
    gw_def: str = "gw",
    gw_sm_weight: float = 1.0,
    window_before_h: int = 72,
    window_after_h: int = 72,
    lags_h: Optional[list[int]] = None,
    tol_minutes: int = 30,
    season_days: int = 30,
    exclude_event_buffer_days: int = 14,
    n_controls: int = 25,
    match_target: str = "gw",
    match_mode: str = "value_pct",
    value_tolerance_levels: Optional[list[float]] = None,
    quantile_tolerance_levels: Optional[list[float]] = None,
    prsum_lag_h: int = 72,
    q_weight: float = 0.10,
    decluster_days: int = 7,
    min_controls_per_year: int = 2,
    random_seed: int = 123,
    gw_act_q: float = 0.99,
    q_act_q: float = 0.99,
    var_features: Optional[Dict[str, Dict[str, bool]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """
    Main high-level runner.

    Returns:
      episodes_df, all_non_event_anchors_df, list_of_non_event_anchor_dfs
    """
    if lags_h is None:
        lags_h = [3, 6, 12, 24, 48, 72]

    event_times_by_guid = (
        valid.groupby("GUID")["event_time"]
        .apply(lambda s: pd.to_datetime(s, errors="coerce").dropna().tolist())
        .to_dict()
    )

    rows = []
    anchor_dfs = []

    for i, guid in enumerate(sorted(valid["GUID"].unique()), start=1):
        print(f"{i}/{valid['GUID'].nunique()} GUID={guid}")

        cache = build_catchment_cache(
            guid=guid,
            sim_path=simres_paths[guid],
            param_path=param_paths[guid],
            gw_def=gw_def,
            gw_sm_weight=gw_sm_weight,
            lags_h=lags_h,
            tol_minutes=tol_minutes,
        )

        valid_g = valid[valid["GUID"] == guid].copy()

        for _, ev in valid_g.iterrows():
            row = compute_episode_stats_pre_anchor(
                cache=cache,
                t_anchor=ev["event_time"],
                is_event=1,
                window_before_h=window_before_h,
                window_after_h=window_after_h,
                lags_h=lags_h,
                var_features=var_features,
                gw_act_q=gw_act_q,
                q_act_q=q_act_q,
            )
            row["SkredID"] = ev.get("SkredID", None)
            row["No"] = ev.get("No", None)
            rows.append(row)

        anchors = select_controls_progressive(
            cache=cache,
            event_times=event_times_by_guid.get(guid, []),
            season_days=season_days,
            exclude_event_buffer_days=exclude_event_buffer_days,
            n_controls=n_controls,
            match_target=match_target,
            match_mode=match_mode,
            value_tolerance_levels=value_tolerance_levels,
            quantile_tolerance_levels=quantile_tolerance_levels,
            prsum_lag_h=prsum_lag_h,
            q_weight=q_weight,
            decluster_days=decluster_days,
            min_controls_per_year=min_controls_per_year,
            random_seed=random_seed,
        )

        print(
            "  controls:", len(anchors),
            " max_tier:", anchors["match_tier"].max() if len(anchors) else None,
            " max_dist:", round(float(anchors["match_distance"].max()), 5) if len(anchors) else None,
        )

        if not anchors.empty:
            anchor_dfs.append(anchors)
            for _, a in anchors.iterrows():
                row = compute_episode_stats_pre_anchor(
                    cache=cache,
                    t_anchor=a["t_anchor"],
                    is_event=0,
                    window_before_h=window_before_h,
                    window_after_h=window_after_h,
                    lags_h=lags_h,
                    var_features=var_features,
                    gw_act_q=gw_act_q,
                    q_act_q=q_act_q,
                )
                for c in anchors.columns:
                    if c not in {"GUID"}:
                        row[f"ne_{c}"] = a[c]
                rows.append(row)

    episodes_df = pd.DataFrame(rows)
    non_event_anchors_df = pd.concat(anchor_dfs, ignore_index=True) if anchor_dfs else pd.DataFrame()
    return episodes_df, non_event_anchors_df, anchor_dfs




def feature_name_table(lags_h: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Compact feature description table for the fast progressive dataset.

    This focuses on the derived features added in this module plus the most
    important existing base columns. It is intentionally compact rather than
    listing every old rate/quantile column exhaustively.
    """
    if lags_h is None:
        lags_h = [3, 6, 12, 24, 48, 72]
    lags_h = sorted(set(int(x) for x in lags_h))

    rows: list[tuple[str, str]] = []
    add = rows.append

    add(("GUID", "Catchment identifier."))
    add(("t0", "Anchor timestamp for event or non-event episode."))
    add(("y", "Episode label: 1=event, 0=non-event."))
    add(("gw0, sm0, q0, qmm0, pr0, swe0", "Variable values at the anchor timestamp."))
    add(("gwq, smq, qq, qmmq, prq, sweq", "Full-series empirical quantile of anchor value."))
    add(("{var}pk_t", "Hours from variable peak to anchor in [t0-wb, t0+wa]. Positive=peak before anchor, negative=peak after anchor."))
    add(("{var}pk_0", "Value of selected variable peak in the combined peak window."))
    add(("{var}pk_q", "Full-series empirical quantile of selected peak value."))
    add(("{var}r{H}b", "Rate of change before anchor over H hours: (x0 - x(t0-H))/H."))
    add(("prS{H}b", "Precipitation sum over H hours ending at anchor."))
    add(("qmmS{H}b", "Runoff depth sum over H hours ending at anchor."))
    add(("netS{H}b", "Input-output imbalance over H hours: prS{H}b - qmmS{H}b."))
    add(("sweLossS{H}b", "Positive SWE loss over H hours before anchor; used as melt proxy."))

    add(("total_storage", "Combined storage state at anchor: gw0 + sm0."))
    add(("storage_fraction", "total_storage divided by catchment capacity cap."))
    add(("relative_storage", "total_storage divided by full-series max(gw+sm)."))
    add(("storage_anomaly", "Z-score of total_storage relative to full-series gw+sm."))

    add(("{var}_delay_vs_pr", "Peak timing delay relative to precipitation peak: {var}pk_t - prpk_t."))
    add(("{var}_abs_delay_vs_pr", "Absolute value of {var}_delay_vs_pr."))
    add(("{var}_peak_before_flag", "1 if {var} peak occurs before anchor, 0 otherwise."))
    add(("{var}_peak_after_flag", "1 if {var} peak occurs after anchor, 0 otherwise."))
    add(("{var}_distance_to_peak", "Absolute hours between anchor and selected {var} peak."))

    for H in lags_h:
        add((f"runoff_ratio_{H}b", f"qmmS{H}b / prS{H}b. Low values indicate retained water; high values indicate efficient runoff response."))
        add((f"infiltration_proxy_{H}b", f"1 - runoff_ratio_{H}b."))
        add((f"rain_plus_melt_{H}b", f"prS{H}b + sweLossS{H}b."))
        add((f"response_efficiency_{H}b", f"qmmS{H}b / (prS{H}b + sweLossS{H}b)."))
        add((f"melt_fraction_{H}b", f"sweLossS{H}b / (prS{H}b + sweLossS{H}b)."))
        add((f"rain_fraction_{H}b", f"prS{H}b / (prS{H}b + sweLossS{H}b)."))
        add((f"precip_x_gw_{H}b", f"Rain-on-wetness interaction: prS{H}b * gw0."))
        add((f"precip_x_sm_{H}b", f"Rain-on-soil-moisture interaction: prS{H}b * sm0."))
        add((f"netS_x_gw_{H}b", f"Imbalance-on-wetness interaction: netS{H}b * gw0."))
        add((f"gwr_norm_{H}b", f"Groundwater rate normalized by anchor groundwater: gwr{H}b / gw0."))
        add((f"smr_norm_{H}b", f"Soil-moisture rate normalized by anchor soil moisture: smr{H}b / sm0."))
        add((f"qmmr_norm_{H}b", f"Runoff-depth rate normalized by anchor runoff depth: qmmr{H}b / qmm0."))
        add((f"qr_norm_{H}b", f"Runoff rate normalized by anchor runoff: qr{H}b / q0."))
        add((f"netS_ratio_{H}b", f"netS{H}b / prS{H}b."))
        add((f"netS_norm_storage_{H}b", f"netS{H}b / total_storage."))
        add((f"netS_gradient_{H}b", f"Approximate imbalance rate: netS{H}b / H."))
        add((f"hours_gw_high_q90_{H}b", f"Number of samples/hours in the previous {H} h where gw > full-series q90."))
        add((f"gw_excess_q90_sum_{H}b", f"Cumulative positive exceedance above GW q90 in the previous {H} h."))
        add((f"hours_pr_high_q90_{H}b", f"Number of samples/hours in the previous {H} h where precip > full-series q90."))
        add((f"pr_excess_q90_sum_{H}b", f"Cumulative positive precip exceedance above precip q90 in the previous {H} h."))
        add((f"hours_qmm_high_q90_{H}b", f"Number of samples/hours in the previous {H} h where qmm > full-series q90."))
        add((f"qmm_excess_q90_sum_{H}b", f"Cumulative positive qmm exceedance above qmm q90 in the previous {H} h."))
        for v in ["gw", "q", "qmm", "sm"]:
            add((f"{v}r{H}b_x_peak_before", f"{v}r{H}b multiplied by peak-before-anchor flag."))
            add((f"{v}r{H}b_x_peak_after", f"{v}r{H}b multiplied by peak-after-anchor flag."))

    for a, b in [(3, 24), (6, 24), (12, 24), (24, 72)]:
        if a in lags_h and b in lags_h:
            add((f"pr_ratio_{a}_{b}b", f"Short/long precipitation ratio: prS{a}b / prS{b}b."))
            add((f"netS_ratio_{a}_{b}b", f"Short/long imbalance ratio: netS{a}b / netS{b}b."))
            add((f"gw_rate_ratio_{a}_{b}b", f"Short/long groundwater-rate ratio: gwr{a}b / gwr{b}b."))
            add((f"sm_rate_ratio_{a}_{b}b", f"Short/long soil-moisture-rate ratio: smr{a}b / smr{b}b."))
            add((f"qmm_rate_ratio_{a}_{b}b", f"Short/long runoff-depth-rate ratio: qmmr{a}b / qmmr{b}b."))

    add(("ne_match_tier", "For non-events: tolerance tier used during control selection."))
    add(("ne_match_distance", "For non-events: distance used to rank/filter selected controls."))
    add(("ne_match_source", "For non-events: matching source/target used, e.g. gw, prsum, both, all3, both_qweighted."))
    add(("ne_match_tolerance_used", "For non-events: tolerance level used for selected control; NaN can indicate best-extra fallback."))

    return pd.DataFrame(rows, columns=["feature", "description"])


def save_feature_name_table(out_path: str | Path, lags_h: Optional[list[int]] = None) -> Path:
    """Save the compact feature description table to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_name_table(lags_h=lags_h).to_csv(out_path, index=False)
    return out_path