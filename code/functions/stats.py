# functions/stats.py
from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, List
import numpy as np
import pandas as pd

from .params import load_param_values


# =========================
# Helpers
# =========================
def _force_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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
    return out


def _ecdf_sorted(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    vals.sort()
    return vals


def _ecdf_u(sorted_vals: np.ndarray, x: float) -> float:
    if sorted_vals.size == 0 or not np.isfinite(x):
        return np.nan
    k = np.searchsorted(sorted_vals, x, side="right")
    return float(k / sorted_vals.size)


def _nearest_at(s: pd.Series, t: pd.Timestamp, tol: pd.Timedelta) -> float:
    """
    Return nearest value in s at timestamp t, within tolerance tol.
    If none within tol -> np.nan
    """
    if s.empty:
        return np.nan
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()

    pos = s.index.get_indexer([t], method="nearest")[0]
    if pos == -1:
        return np.nan

    t_near = s.index[pos]
    if abs(t_near - t) > tol:
        return np.nan

    v = s.iloc[pos]
    return float(v) if np.isfinite(v) else np.nan


def _rate_before(x0: float, x_before: float, H: int) -> Optional[float]:
    if not (np.isfinite(x0) and np.isfinite(x_before) and H > 0):
        return None
    return float((x0 - x_before) / H)


def _rate_after(x0: float, x_after: float, H: int) -> Optional[float]:
    if not (np.isfinite(x0) and np.isfinite(x_after) and H > 0):
        return None
    return float((x_after - x0) / H)


def _first_cross_time_in_window(
    s: pd.Series,
    thr: float,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.Timestamp | None:
    """
    Return first timestamp in [t_start, t_end] where s >= thr.
    If never crosses, return None.
    """
    if s.empty or not np.isfinite(thr):
        return None
    w = s.loc[(s.index >= t_start) & (s.index <= t_end)]
    if w.empty:
        return None
    hit = w[w >= thr]
    if hit.empty:
        return None
    return pd.Timestamp(hit.index[0])


def _max_slope_in_window(
    times: np.ndarray,
    values: np.ndarray,
    max_dt_h: float,
    mode: str,
) -> tuple[Optional[float], Optional[int]]:
    """
    mode: "rise" -> max slope
          "fall" -> min slope
    returns (best_slope, best_j) where j is endpoint index
    """
    n = len(values)
    if n < 2:
        return None, None

    best: Optional[float] = None
    best_j: Optional[int] = None

    for i in range(n - 1):
        for j in range(i + 1, n):
            dt_h = (times[j] - times[i]).astype("timedelta64[s]").astype(float) / 3600.0
            if dt_h <= 0:
                continue
            if dt_h > max_dt_h:
                break

            slope = (values[j] - values[i]) / dt_h

            if best is None:
                best = float(slope)
                best_j = j
                continue

            if mode == "rise":
                if slope > best:
                    best = float(slope)
                    best_j = j
            elif mode == "fall":
                if slope < best:
                    best = float(slope)
                    best_j = j
            else:
                raise ValueError("mode must be 'rise' or 'fall'")

    return best, best_j


# =========================
# Variable config (compact codes)
# =========================
VAR_CODE_TO_COL = {
    "gw": None,              # built signal (gw or gw+sm)
    "q": "Q_m3s",
    "sm": "Soilmoist_P_mm",
    "of": "Q_OF_m3s",
    "pr": "Prec_mm",
    "tmp": "Temp_C",
    "swe": "SWE_P_mm",
}

# Which feature families we compute per variable by default
DEFAULT_VAR_FEATURES = {
    "gw": {"anchor": True, "quantile": True, "pctcap": True,  "rates": True,  "max": True},
    "q":  {"anchor": True, "quantile": True, "pctcap": False, "rates": True,  "max": True},
    "sm": {"anchor": True, "quantile": True, "pctcap": False, "rates": True,  "max": True},
    "of": {"anchor": True, "quantile": True, "pctcap": False, "rates": True,  "max": True},
    "pr": {"anchor": True, "quantile": True, "pctcap": False, "rates": True,  "max": True},
    "swe":{"anchor": True, "quantile": True, "pctcap": False, "rates": True,  "max": True},

    # Temperature: keep only level (and quantile) by default
    "tmp":{"anchor": True, "quantile": True, "pctcap": False, "rates": False, "max": False},
}


# =========================
# GW signal definition
# =========================
def build_gw_signal(df: pd.DataFrame, gw_def: str = "gw", gw_sm_weight: float = 1.0) -> tuple[pd.Series, str]:
    """
    gw_def:
      - "gw"         -> GrwPlus_P_mm
      - "gw_plus_sm" -> GrwPlus_P_mm + gw_sm_weight * Soilmoist_P_mm
    """
    gw_def = (gw_def or "gw").lower().strip()

    if gw_def == "gw":
        return _series_from_df(df, "GrwPlus_P_mm"), "GrwPlus_P_mm"

    if gw_def == "gw_plus_sm":
        s_gw = _series_from_df(df, "GrwPlus_P_mm")
        s_sm = _series_from_df(df, "Soilmoist_P_mm")
        if s_gw.empty and s_sm.empty:
            return pd.Series(dtype=float), f"GrwPlus_P_mm + {gw_sm_weight}*Soilmoist_P_mm"
        idx = s_gw.index.union(s_sm.index)
        sig = s_gw.reindex(idx) + (gw_sm_weight * s_sm.reindex(idx))
        sig = pd.to_numeric(sig, errors="coerce").dropna().sort_index()
        sig.name = "gw_signal"
        return sig, f"GrwPlus_P_mm + {gw_sm_weight}*Soilmoist_P_mm"

    raise ValueError(f"Unknown gw_def={gw_def!r}. Use 'gw' or 'gw_plus_sm'.")


# =========================
# Episode stats (events + non-events)
# =========================
def compute_episode_stats(
    guid: str,
    sim_path: str,
    param_path: str,
    t_anchor,
    is_event: int,
    window_before_h: int = 72,
    window_after_h: int = 72,
    lags_h: Optional[list[int]] = None,
    gw_def: str = "gw",
    gw_sm_weight: float = 1.0,
    var_features: Optional[Dict[str, Dict[str, bool]]] = None,
    tol_minutes: int = 90,

    # Hysteresis activation thresholds (full-series quantiles)
    gw_act_q: float = 0.90,
    q_act_q: float = 0.90,
) -> Dict[str, Any]:
    """
    Compute one row of episode features centered on t_anchor.

    Window length choice:
      - window_before_h controls the PRE window [t0-window_before_h, t0]
      - window_after_h  controls the POST window [t0, t0+window_after_h]
    Works identically for:
      - events: t_anchor = landslide time
      - non-events: t_anchor = selected GW peak time (or other anchor)

    Feature naming:
      Anchor:
        {v}0  : value at t0
        {v}q  : full-series quantile at t0
        {v}p  : % capacity at t0 (only if enabled; meaningful mainly for gw)

      Rates (if enabled for v):
        {v}r{H}b : (v(t0)-v(t0-H))/H
        {v}r{H}a : (v(t0+H)-v(t0))/H

      "Extremeness" (if enabled):
        {v}x{H}  : {v}r{H}b - {v}r{B}b, where B = max lag <= window_before_h

      Max rise/fall within PRE/POST window using dt<=H (if enabled):
        {v}m{H}b, {v}n{H}b, {v}m{H}a, {v}n{H}a
        plus endpoint metadata: _t, _0, _q for each max stat

      Hysteresis activation lag (pre-window, full-series thresholds):
        hxl_pre : hours( t_gw_cross - t_q_cross )
          + => GW crosses earlier (antecedent storage leads)
          - => Q crosses earlier (flashy runoff leads)
        Also stores:
          hxl_pre_gwt (hours BEFORE t0 when GW crosses), hxl_pre_qt (same for Q)
          hxl_pre_gwthr, hxl_pre_qthr (threshold values used)
    """
    if lags_h is None:
        lags_h = [3, 6, 12, 24, 48, 72]
    lags_h = sorted(list(set(int(x) for x in lags_h)))

    feats = var_features or DEFAULT_VAR_FEATURES

    t_anchor = pd.to_datetime(t_anchor).floor("h")

    df = pd.read_csv(sim_path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    df = _force_numeric(
        df,
        ["Prec_mm", "Temp_C", "Q_m3s", "Q_OF_m3s", "SubsurfaceDef_P_mm", "GrwPlus_P_mm", "SWE_P_mm", "Soilmoist_P_mm"],
    )

    # anchor time: choose closest in df (robust)
    if (df["Datetime"] == t_anchor).any():
        ts0 = t_anchor
    else:
        ts0 = pd.to_datetime(df.loc[(df["Datetime"] - t_anchor).abs().idxmin(), "Datetime"]).floor("h")

    # capacity
    M = df["SubsurfaceDef_P_mm"].iloc[0] if "SubsurfaceDef_P_mm" in df.columns else np.nan
    tol = pd.Timedelta(minutes=int(tol_minutes))

    # build series
    gw_sig, gw_sig_name = build_gw_signal(df, gw_def=gw_def, gw_sm_weight=gw_sm_weight)

    series_map: Dict[str, pd.Series] = {"gw": gw_sig}
    for v, col in VAR_CODE_TO_COL.items():
        if v == "gw" or col is None:
            continue
        series_map[v] = _series_from_df(df, col)

    # baseline lag B for extremeness (pre)
    eligible_B = [h for h in lags_h if h <= int(window_before_h)]
    B = max(eligible_B) if eligible_B else None

    # row base
    row: Dict[str, Any] = {
        "GUID": guid,
        "t0": ts0,
        "y": int(is_event),      # 1 event, 0 non-event
        "wb": int(window_before_h),
        "wa": int(window_after_h),
        "lags": ",".join(str(x) for x in lags_h),
        "cap": M,
        "gwdef": gw_def,
        "gwsig": gw_sig_name,
        "gw_act_q": float(gw_act_q),
        "q_act_q": float(q_act_q),
    }

    # params
    pv = load_param_values(param_path)
    row["A_km2"] = pv.get("Area_km2")
    row["HI"] = pv.get("HI")
    row["MRT"] = pv.get("MRT")
    row["meandailyT"] = pv.get("meandailyT")

    # =========================
    # Per-variable features
    # =========================
    for v, s in series_map.items():
        cfg = feats.get(v, {"anchor": True, "quantile": True, "pctcap": False, "rates": False, "max": False})

        if s.empty:
            if cfg.get("anchor", False):
                row[f"{v}0"] = np.nan
            if cfg.get("quantile", False):
                row[f"{v}q"] = np.nan
            if cfg.get("pctcap", False):
                row[f"{v}p"] = np.nan
            continue

        sorted_full = _ecdf_sorted(s)

        # anchor
        x0 = _nearest_at(s, ts0, tol)
        if cfg.get("anchor", False):
            row[f"{v}0"] = x0
        if cfg.get("quantile", False):
            row[f"{v}q"] = _ecdf_u(sorted_full, x0)
        if cfg.get("pctcap", False):
            row[f"{v}p"] = float((x0 / M) * 100.0) if (pd.notna(M) and M > 0 and np.isfinite(x0)) else np.nan

        # rates
        if cfg.get("rates", False):
            for H in lags_h:
                xb = _nearest_at(s, ts0 - pd.Timedelta(hours=H), tol)
                xa = _nearest_at(s, ts0 + pd.Timedelta(hours=H), tol)
                row[f"{v}r{H}b"] = _rate_before(x0, xb, H)
                row[f"{v}r{H}a"] = _rate_after(x0, xa, H)

            # extremeness vs long-lag rate
            if B is not None:
                rB = row.get(f"{v}r{B}b", None)
                for H in lags_h:
                    rH = row.get(f"{v}r{H}b", None)
                    row[f"{v}x{H}"] = float(rH - rB) if (rH is not None and rB is not None) else None

        # max rise/fall (pre/post)
        if cfg.get("max", False):
            pre = s.loc[(s.index > ts0 - pd.Timedelta(hours=window_before_h)) & (s.index <= ts0)].dropna().sort_index()
            post = s.loc[(s.index >= ts0) & (s.index < ts0 + pd.Timedelta(hours=window_after_h))].dropna().sort_index()

            def _do_window(win: pd.Series, where: str, win_h: int):
                if win.empty or len(win) < 2:
                    return

                sorted_win = _ecdf_sorted(win)
                times = win.index.to_numpy()
                vals = win.to_numpy()

                for H in lags_h:
                    if H > win_h:
                        continue

                    # max rise
                    mkey = f"{v}m{H}{where}"
                    best, j = _max_slope_in_window(times, vals, max_dt_h=float(H), mode="rise")
                    row[mkey] = best
                    if j is not None:
                        t_end = pd.Timestamp(times[j])
                        end_val = float(vals[j]) if np.isfinite(vals[j]) else np.nan
                        dt = (ts0 - t_end).total_seconds() / 3600.0 if where == "b" else (t_end - ts0).total_seconds() / 3600.0
                        row[f"{mkey}_t"] = float(dt)
                        row[f"{mkey}_0"] = end_val
                        row[f"{mkey}_q"] = _ecdf_u(sorted_win, end_val)

                    # max fall
                    nkey = f"{v}n{H}{where}"
                    best, j = _max_slope_in_window(times, vals, max_dt_h=float(H), mode="fall")
                    row[nkey] = best
                    if j is not None:
                        t_end = pd.Timestamp(times[j])
                        end_val = float(vals[j]) if np.isfinite(vals[j]) else np.nan
                        dt = (ts0 - t_end).total_seconds() / 3600.0 if where == "b" else (t_end - ts0).total_seconds() / 3600.0
                        row[f"{nkey}_t"] = float(dt)
                        row[f"{nkey}_0"] = end_val
                        row[f"{nkey}_q"] = _ecdf_u(sorted_win, end_val)

            _do_window(pre, "b", int(window_before_h))
            _do_window(post, "a", int(window_after_h))

    # =========================
    # Hysteresis timing: activation crossing lag (PRE-window)
    # =========================
    # Thresholds are FULL-series quantiles (unit-free, comparable across GW vs Q).
    # We find first crossing in [t0-wb, t0] for both, then compute:
    # hxl_pre = t_gw_cross - t_q_cross (hours)
    # + => GW crosses earlier, - => Q crosses earlier.
    row["hxl_pre"] = np.nan
    row["hxl_pre_gwt"] = np.nan
    row["hxl_pre_qt"] = np.nan
    row["hxl_pre_gwthr"] = np.nan
    row["hxl_pre_qthr"] = np.nan

    gw_s = series_map.get("gw", pd.Series(dtype=float)).dropna().sort_index()
    q_s = series_map.get("q", pd.Series(dtype=float)).dropna().sort_index()

    if (not gw_s.empty) and (not q_s.empty):
        # thresholds from FULL series
        try:
            gw_thr = float(gw_s.quantile(float(gw_act_q)))
        except Exception:
            gw_thr = np.nan
        try:
            q_thr = float(q_s.quantile(float(q_act_q)))
        except Exception:
            q_thr = np.nan

        row["hxl_pre_gwthr"] = gw_thr
        row["hxl_pre_qthr"] = q_thr

        t_start = ts0 - pd.Timedelta(hours=int(window_before_h))
        t_end = ts0

        t_gw = _first_cross_time_in_window(gw_s, gw_thr, t_start, t_end) if np.isfinite(gw_thr) else None
        t_q = _first_cross_time_in_window(q_s, q_thr, t_start, t_end) if np.isfinite(q_thr) else None

        row["hxl_pre_gwt"] = float((ts0 - t_gw).total_seconds() / 3600.0) if t_gw is not None else np.nan
        row["hxl_pre_qt"] = float((ts0 - t_q).total_seconds() / 3600.0) if t_q is not None else np.nan

        if (t_gw is not None) and (t_q is not None):
            row["hxl_pre"] = float((t_gw - t_q).total_seconds() / 3600.0)

    return row


# =========================
# Non-event selection (POT-like) anchored on GW signal
# =========================
def select_non_event_gw_peaks(
    df: pd.DataFrame,
    guid: str,
    event_times: Iterable[pd.Timestamp],
    gw_def: str = "gw",
    gw_sm_weight: float = 1.0,
    threshold_mode: str = "quantile",  # "quantile" or "capacity_frac"
    threshold_q: float = 0.95,
    capacity_frac: float = 0.90,
    decluster_days: int = 7,
    peaks_per_year: int = 2,
    exclude_event_buffer_days: int = 3,
) -> pd.DataFrame:
    """
    POT-like GW peak selection (non-event anchors):
      - compute GW signal (gw or gw_plus_sm)
      - apply threshold (quantile or capacity fraction)
      - find local maxima above threshold
      - exclude those within ±buffer of any event time
      - decluster within each year, then pick top N per year

    Returns columns:
      GUID, t_anchor, peak_value, year, rank_in_year
    """
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    df = _force_numeric(df, ["GrwPlus_P_mm", "Soilmoist_P_mm", "SubsurfaceDef_P_mm"])

    gw_sig, _ = build_gw_signal(df, gw_def=gw_def, gw_sm_weight=gw_sm_weight)
    gw_sig = pd.to_numeric(gw_sig, errors="coerce").dropna().sort_index()
    if gw_sig.empty:
        return pd.DataFrame(columns=["GUID", "t_anchor", "peak_value", "year", "rank_in_year"])

    M = df["SubsurfaceDef_P_mm"].iloc[0] if "SubsurfaceDef_P_mm" in df.columns else np.nan

    threshold_mode = (threshold_mode or "quantile").lower().strip()
    if threshold_mode == "quantile":
        thr = float(gw_sig.quantile(float(threshold_q)))
    elif threshold_mode == "capacity_frac":
        if pd.notna(M) and M > 0:
            thr = float(float(capacity_frac) * float(M))
        else:
            thr = float(gw_sig.quantile(float(threshold_q)))
    else:
        raise ValueError("threshold_mode must be 'quantile' or 'capacity_frac'")

    s = gw_sig
    prev = s.shift(1)
    nxt = s.shift(-1)
    is_peak = (s > prev) & (s >= nxt) & (s > thr)
    cand = s[is_peak].dropna()
    if cand.empty:
        return pd.DataFrame(columns=["GUID", "t_anchor", "peak_value", "year", "rank_in_year"])

    cand_df = cand.reset_index()
    cand_df.columns = ["t_anchor", "peak_value"]
    cand_df["t_anchor"] = pd.to_datetime(cand_df["t_anchor"], errors="coerce").dt.floor("h")
    cand_df = cand_df.dropna(subset=["t_anchor"])
    cand_df["year"] = cand_df["t_anchor"].dt.year

    # exclude peaks near known events
    evts = [pd.to_datetime(t, errors="coerce") for t in event_times]
    evts = [t.floor("h") for t in evts if pd.notna(t)]
    if evts:
        buf = pd.Timedelta(days=int(exclude_event_buffer_days))
        keep = []
        for t in cand_df["t_anchor"]:
            ok = True
            for et in evts:
                if abs(pd.Timestamp(t) - et) <= buf:
                    ok = False
                    break
            keep.append(ok)
        cand_df = cand_df.loc[keep].copy()

    if cand_df.empty:
        return pd.DataFrame(columns=["GUID", "t_anchor", "peak_value", "year", "rank_in_year"])

    min_sep = pd.Timedelta(days=int(decluster_days))

    out_rows = []
    for year, g in cand_df.groupby("year"):
        g = g.sort_values("peak_value", ascending=False).reset_index(drop=True)

        selected: List[pd.Timestamp] = []
        picked = []
        for _, r in g.iterrows():
            t = pd.Timestamp(r["t_anchor"])
            if all(abs(t - tsel) >= min_sep for tsel in selected):
                selected.append(t)
                picked.append({"t_anchor": t, "peak_value": float(r["peak_value"]), "year": int(year)})
            if len(picked) >= int(peaks_per_year):
                break

        for rank, rr in enumerate(picked, start=1):
            rr["GUID"] = guid
            rr["rank_in_year"] = rank
            out_rows.append(rr)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return pd.DataFrame(columns=["GUID", "t_anchor", "peak_value", "year", "rank_in_year"])

    return out[["GUID", "t_anchor", "peak_value", "year", "rank_in_year"]].sort_values(["year", "rank_in_year"]).reset_index(drop=True)


# =========================
# Feature name table (for sanity / documentation)
# =========================
def feature_name_table(lags_h: list[int]) -> pd.DataFrame:
    var_desc = {
        "gw": "groundwater signal (gw or gw_plus_sm)",
        "q": "runoff (Q_m3s)",
        "sm": "soil moisture (Soilmoist_P_mm)",
        "of": "overland flow (Q_OF_m3s)",
        "pr": "precipitation (Prec_mm)",
        "tmp": "temperature (Temp_C)",
        "swe": "snow water equivalent (SWE_P_mm)",
    }

    rows = []

    # meta
    rows.append(("GUID", "catchment identifier"))
    rows.append(("t0", "anchor timestamp for the episode (event time or non-event peak time)"))
    rows.append(("y", "episode label: 1=event, 0=non-event"))
    rows.append(("wb", "pre-window length (hours)"))
    rows.append(("wa", "post-window length (hours)"))
    rows.append(("cap", "capacity (SubsurfaceDef_P_mm first value)"))
    rows.append(("gwdef", "definition of GW signal used (gw or gw_plus_sm)"))
    rows.append(("gwsig", "text description of the GW signal formula"))
    rows.append(("A_km2", "catchment area from param file"))
    rows.append(("HI", "HI from param file"))
    rows.append(("MRT", "MRT from param file"))
    rows.append(("meandailyT", "meandailyT from param file"))
    rows.append(("gw_act_q", "GW activation quantile used for hysteresis crossing"))
    rows.append(("q_act_q", "Q activation quantile used for hysteresis crossing"))

    # anchor features
    for v, desc in var_desc.items():
        rows.append((f"{v}0", f"{desc}: value at anchor t0"))
        rows.append((f"{v}q", f"{desc}: full-series empirical quantile at anchor t0"))
        rows.append((f"{v}p", f"{desc}: percent of capacity at t0 (mostly meaningful for gw)"))

    # lag-based features
    for H in sorted(set(int(x) for x in lags_h)):
        for v, desc in var_desc.items():
            rows.append((f"{v}r{H}b", f"{desc}: rate BEFORE anchor: (v(t0)-v(t0-{H}h))/{H}"))
            rows.append((f"{v}r{H}a", f"{desc}: rate AFTER  anchor: (v(t0+{H}h)-v(t0))/{H}"))
            rows.append((f"{v}x{H}", f"{desc}: extremeness: v_r{H}b - v_rLONGb (LONG=max lag<=wb)"))
            rows.append((f"{v}m{H}b", f"{desc}: max RISE rate in PRE window with dt<= {H}h"))
            rows.append((f"{v}n{H}b", f"{desc}: max FALL rate in PRE window with dt<= {H}h"))
            rows.append((f"{v}m{H}a", f"{desc}: max RISE rate in POST window with dt<= {H}h"))
            rows.append((f"{v}n{H}a", f"{desc}: max FALL rate in POST window with dt<= {H}h"))
            rows.append((f"{v}m{H}b_t", f"{desc}: endpoint hours BEFORE anchor for max-rise PRE"))
            rows.append((f"{v}m{H}b_0", f"{desc}: endpoint value for max-rise PRE"))
            rows.append((f"{v}m{H}b_q", f"{desc}: endpoint quantile (within PRE series)"))
            rows.append((f"{v}n{H}b_t", f"{desc}: endpoint hours BEFORE anchor for max-fall PRE"))
            rows.append((f"{v}n{H}b_0", f"{desc}: endpoint value for max-fall PRE"))
            rows.append((f"{v}n{H}b_q", f"{desc}: endpoint quantile (within PRE series)"))

    # hysteresis crossing lag
    rows.append(("hxl_pre", "hysteresis activation lag (hours): t_gw_cross - t_q_cross within PRE window"))
    rows.append(("hxl_pre_gwt", "hours BEFORE t0 when GW first crosses its activation threshold in PRE window"))
    rows.append(("hxl_pre_qt", "hours BEFORE t0 when Q first crosses its activation threshold in PRE window"))
    rows.append(("hxl_pre_gwthr", "GW activation threshold value (full-series quantile)"))
    rows.append(("hxl_pre_qthr", "Q activation threshold value (full-series quantile)"))

    return pd.DataFrame(rows, columns=["feature", "meaning"])
