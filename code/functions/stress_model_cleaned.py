from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from functions.io_search import find_sim_param_files
from functions.metadata import load_beskrivelser_events
from functions.params import load_param_values


AccumMode = Literal["rolling", "exp"]
ForcingThresholdMode = Literal["fraction_of_max", "quantile"]
FinalCombineMode = Literal["weighted_mean", "weighted_sum_norm", "weighted_sum_power_norm"]


# ============================================================
# Config
# ============================================================
@dataclass
class StressConfig:
    # ----------------------------
    # OUTPUT
    # ----------------------------
    export_format: Literal["csv", "xlsx"] = "csv"
    export_after_plots: bool = True

    # ----------------------------
    # EVENT WINDOWS (in HOURS)
    # ----------------------------
    hours_before: float = 48
    hours_after: float = 12
    post_hyst_hours: float = 72

    # ----------------------------
    # PER-GROUP MEMORY SETTINGS
    # ----------------------------
    base_accum_mode: AccumMode = "exp"
    base_window_days: float = 14
    base_tau_days: float = 1.0

    grad_accum_mode: AccumMode = "exp"
    grad_window_days: float = 14
    grad_tau_days: float = 1.0

    forc_accum_mode: AccumMode = "exp"
    forc_window_days: float = 14
    forc_tau_days: float = 1.0

    # ----------------------------
    # BASE THRESHOLDS (GW/SM)
    # ----------------------------
    use_capacity_fraction: bool = False
    capacity_frac: float = 0.90

    use_quantile_threshold: bool = True
    gw_q_threshold: float = 0.99

    sm_use_quantile_threshold: bool = True
    sm_q_threshold: float = 0.99

    # ----------------------------
    # OF THRESHOLDS (OF component)
    # ----------------------------
    of_use_quantile_threshold: bool = False
    of_q_threshold: float = 0.90
    of_use_zero_threshold: bool = True

    # ----------------------------
    # GRADIENT THRESHOLDS
    # ----------------------------
    grad_threshold_mode: ForcingThresholdMode = "quantile"
    pos_grad_q: float = 0.99
    neg_grad_q: float = 0.90
    pos_grad_frac_of_max: float = 0.70
    neg_grad_frac_of_max: float = 0.70

    # ----------------------------
    # FORCING THRESHOLDS
    # ----------------------------
    prec_threshold_mode: ForcingThresholdMode = "fraction_of_max"
    prec_frac_of_max: float = 0.70
    prec_q: float = 0.99

    swe_threshold_mode: ForcingThresholdMode = "fraction_of_max"
    swe_frac_of_max: float = 0.70
    swe_q: float = 0.99

    # ----------------------------
    # COMPONENT TOGGLES + WEIGHTS
    # ----------------------------
    include_gw_base: bool = True
    include_sm_base: bool = True

    # kept for backward compatibility; now means:
    # include OF as a separate component
    include_of_base: bool = False

    gw_base_mult: float = 1.0
    sm_base_mult: float = 1.0
    of_base_mult: float = 1.0  # internal OF scaling before memory

    include_pos_grad: bool = True
    include_neg_grad: bool = False
    grad_hours: float = 6

    pos_grad_mult: float = 1.0
    neg_grad_mult: float = 1.0

    include_prec: bool = True
    prec_hours: float = 6
    prec_mult: float = 1.0  # final PREC weight

    include_swe: bool = True
    swe_hours: float = 6
    swe_mult: float = 1.0  # final SWE weight

    of_mult: float = 1.0  # final OF weight

    split_prec_by_temp: bool = True
    temp_rain_threshold_c: float = 0.5
    use_only_rain: bool = True

    # internal combination of GW and SM within BASE
    combine_mode: Literal["sum", "avg"] = "sum"

    # final combination weights
    base_mult: float = 1.0
    grad_mult: float = 1.0

    # ----------------------------
    # FINAL TOTAL STRESS COMBINATION
    # ----------------------------
    final_combine_mode: FinalCombineMode = "weighted_mean"
    final_combine_power: float = 2.0

    # ----------------------------
    # NORMALIZATION
    # ----------------------------
    norm_q_low: float = 0.01
    norm_q_high: float = 0.99

    # ----------------------------
    # IO
    # ----------------------------
    usecols: Tuple[str, ...] = (
        "Datetime",
        "Prec_mm", "Temp_C",
        "Q_m3s", "Q_OF_m3s",
        "SubsurfaceDef_P_mm", "GrwPlus_P_mm",
        "SWE_P_mm", "Soilmoist_P_mm",
    )

    # ----------------------------
    # PLOTTING
    # ----------------------------
    figsize: Tuple[int, int] = (12, 18)
    dpi: int = 200
    zoom_hour_tick_interval: int = 6

    # ----------------------------
    # EXPORT DEBUG
    # ----------------------------
    export_debug_intensities: bool = True
    export_debug_thresholds: bool = True


# ============================================================
# Discovery
# ============================================================
def build_valid_events(
    *,
    beskrivelser_file: Path,
    input_dir: Path,
    tr: str,
) -> tuple[pd.DataFrame, dict, dict]:
    simres_paths, param_paths = find_sim_param_files(input_dir, tr=tr)
    meta_events = load_beskrivelser_events([Path(beskrivelser_file)])

    valid = meta_events[
        meta_events["GUID"].isin(simres_paths.keys())
        & meta_events["GUID"].isin(param_paths.keys())
    ].copy()

    return valid, simres_paths, param_paths


# ============================================================
# Helpers
# ============================================================
def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col]
    if s.dtype == "object":
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _median_dt_hours(t: pd.Series) -> float:
    dt = pd.to_datetime(t, errors="coerce").diff().dt.total_seconds()
    dt_med = dt.median()
    if pd.isna(dt_med) or dt_med <= 0:
        return 1.0
    return float(dt_med / 3600.0)


def choose_threshold(
    series: pd.Series,
    *,
    M: Optional[float] = None,
    use_capacity_fraction: bool = False,
    cap_frac: float = 0.9,
    use_quantile: bool = True,
    q: float = 0.9,
) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if use_capacity_fraction and (M is not None) and pd.notna(M) and M > 0:
        return float(cap_frac * M)
    if use_quantile:
        if s.notna().sum() == 0:
            return np.nan
        return float(s.quantile(q))
    return np.nan


def compute_rate_over_hours(series: pd.Series, dt_hours: float, hours: float = 1.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    hours = float(hours) if (hours is not None and hours > 0) else float(dt_hours)
    steps = max(int(round(hours / dt_hours)), 1)
    delta = s - s.shift(steps)
    return delta / (hours / 24.0)


def rolling_sum_over_hours(series: pd.Series, dt_hours: float, hours: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    hours = float(hours) if (hours is not None and hours > 0) else float(dt_hours)
    steps = max(int(round(hours / dt_hours)), 1)
    return s.rolling(steps, min_periods=1).sum()


def rolling_positive_drop_over_hours(series: pd.Series, dt_hours: float, hours: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    hours = float(hours) if (hours is not None and hours > 0) else float(dt_hours)
    steps = max(int(round(hours / dt_hours)), 1)
    delta = s.shift(steps) - s
    return delta.clip(lower=0)


def robust_minmax(s: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    lo = s.quantile(q_low) if (q_low is not None) else s.min(skipna=True)
    hi = s.quantile(q_high) if (q_high is not None) else s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or (hi - lo) == 0:
        return pd.Series(0.0, index=s.index)
    out = (s - lo) / (hi - lo)
    return out.clip(0, 1)


def accumulate_inst(inst: pd.Series, dt_hours: float, *, mode: AccumMode, window_days: float, tau_days: float) -> pd.Series:
    dt_days = dt_hours / 24.0
    inst = pd.to_numeric(inst, errors="coerce").fillna(0.0)

    if mode == "rolling":
        steps = max(int(round((window_days * 24.0) / dt_hours)), 1)
        return inst.rolling(steps, min_periods=1).sum() * dt_days

    if mode == "exp":
        if tau_days is None or tau_days <= 0:
            tau_days = dt_days
        r = float(np.exp(-dt_days / tau_days))
        out = np.zeros(len(inst), dtype=float)
        x = inst.to_numpy()
        for i in range(len(x)):
            out[i] = (r * out[i - 1] if i > 0 else 0.0) + x[i] * dt_days
        return pd.Series(out, index=inst.index)

    raise ValueError("mode must be 'rolling' or 'exp'")


def combine_final_stress(
    *,
    base_stress: pd.Series,
    grad_stress: pd.Series,
    prec_stress: pd.Series,
    swe_stress: pd.Series,
    of_stress: pd.Series,
    w_base: float,
    w_grad: float,
    w_prec: float,
    w_swe: float,
    w_of: float,
    mode: FinalCombineMode,
    power: float,
    q_low: float,
    q_high: float,
) -> tuple[pd.Series, pd.Series]:
    if mode == "weighted_mean":
        denom_w = w_base + w_grad + w_prec + w_swe + w_of
        if denom_w <= 0:
            raw = pd.Series(0.0, index=base_stress.index)
            total = raw.copy()
        else:
            raw = (
                w_base * base_stress
                + w_grad * grad_stress
                + w_prec * prec_stress
                + w_swe * swe_stress
                + w_of * of_stress
            )
            total = raw / denom_w
        return total, raw

    if mode == "weighted_sum_norm":
        raw = (
            w_base * base_stress
            + w_grad * grad_stress
            + w_prec * prec_stress
            + w_swe * swe_stress
            + w_of * of_stress
        )
        total = robust_minmax(raw, q_low, q_high).fillna(0.0)
        return total, raw

    if mode == "weighted_sum_power_norm":
        p = float(power) if power is not None and power > 0 else 2.0
        raw = (
            w_base * (base_stress ** p)
            + w_grad * (grad_stress ** p)
            + w_prec * (prec_stress ** p)
            + w_swe * (swe_stress ** p)
            + w_of * (of_stress ** p)
        )
        total = robust_minmax(raw, q_low, q_high).fillna(0.0)
        return total, raw

    raise ValueError("Unknown final_combine_mode")


# ============================================================
# Component builders
# ============================================================
def build_base_excess(x: pd.Series, thr: float, clip_nonneg=True) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if pd.notna(thr) and thr > 0:
        out = (x - thr).clip(lower=0)
    else:
        out = x
    return out.clip(lower=0) if clip_nonneg else out


def build_gradient_component(
    x: pd.Series,
    dt_hours: float,
    *,
    grad_hours: float,
    include_pos: bool,
    include_neg: bool,
    threshold_mode: ForcingThresholdMode,
    pos_q: float,
    neg_q: float,
    pos_frac_of_max: float,
    neg_frac_of_max: float,
) -> tuple[pd.Series, pd.Series, pd.Series, float, float]:
    """
    Returns:
      pos_sig, neg_sig, dXdt, thr_pos, thr_neg
    """
    dXdt = compute_rate_over_hours(x, dt_hours, hours=grad_hours)

    pos_raw = dXdt.clip(lower=0)
    neg_raw = (-dXdt).clip(lower=0)

    pos_sig = pd.Series(0.0, index=x.index)
    neg_sig = pd.Series(0.0, index=x.index)

    thr_pos = np.nan
    thr_neg = np.nan

    if include_pos and pos_raw.notna().any():
        if threshold_mode == "fraction_of_max":
            ref = float(pos_raw.max(skipna=True))
            thr_pos = pos_frac_of_max * ref if np.isfinite(ref) else np.nan
        else:
            thr_pos = float(pos_raw.quantile(pos_q)) if (pos_q is not None) else 0.0
        pos_sig = (pos_raw - thr_pos).clip(lower=0)

    if include_neg and neg_raw.notna().any():
        if threshold_mode == "fraction_of_max":
            ref = float(neg_raw.max(skipna=True))
            thr_neg = neg_frac_of_max * ref if np.isfinite(ref) else np.nan
        else:
            thr_neg = float(neg_raw.quantile(neg_q)) if (neg_q is not None) else 0.0
        neg_sig = (neg_raw - thr_neg).clip(lower=0)

    return pos_sig, neg_sig, dXdt, thr_pos, thr_neg


def build_prec_component(df: pd.DataFrame, dt_hours: float, cfg: StressConfig) -> tuple[pd.Series, pd.Series]:
    if "Prec_mm" not in df.columns:
        z = pd.Series(0.0, index=df.index)
        return z, z

    P = _to_numeric_series(df, "Prec_mm").fillna(0.0)

    if cfg.split_prec_by_temp and ("Temp_C" in df.columns):
        T = _to_numeric_series(df, "Temp_C")
        rain_mask = T >= cfg.temp_rain_threshold_c
        if cfg.use_only_rain:
            P = P.where(rain_mask, 0.0)

    Pwin = rolling_sum_over_hours(P, dt_hours, hours=cfg.prec_hours)

    if Pwin.notna().any():
        if cfg.prec_threshold_mode == "fraction_of_max":
            ref = float(Pwin.max(skipna=True))
            thr = cfg.prec_frac_of_max * ref if np.isfinite(ref) else np.nan
        else:
            thr = float(Pwin.quantile(cfg.prec_q)) if (cfg.prec_q is not None) else 0.0
        prec_excess = (Pwin - thr).clip(lower=0)
    else:
        prec_excess = pd.Series(0.0, index=df.index)

    return prec_excess, Pwin


def build_swe_melt_component(df: pd.DataFrame, dt_hours: float, cfg: StressConfig) -> tuple[pd.Series, pd.Series]:
    if "SWE_P_mm" not in df.columns:
        z = pd.Series(0.0, index=df.index)
        return z, z

    SWE = _to_numeric_series(df, "SWE_P_mm")
    melt_win = rolling_positive_drop_over_hours(SWE, dt_hours, hours=cfg.swe_hours)

    if melt_win.notna().any():
        if cfg.swe_threshold_mode == "fraction_of_max":
            ref = float(melt_win.max(skipna=True))
            thr = cfg.swe_frac_of_max * ref if np.isfinite(ref) else np.nan
        else:
            thr = float(melt_win.quantile(cfg.swe_q)) if (cfg.swe_q is not None) else 0.0
        melt_excess = (melt_win - thr).clip(lower=0)
    else:
        melt_excess = pd.Series(0.0, index=df.index)

    return melt_excess, melt_win


# ============================================================
# Runner
# ============================================================
def run_stress(
    *,
    beskrivelser_file: Path,
    input_dir: Path,
    tr: str,
    out_dir: Path,
    cfg: StressConfig,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid, simres_paths, param_paths = build_valid_events(
        beskrivelser_file=Path(beskrivelser_file),
        input_dir=input_dir,
        tr=tr,
    )
    print("Stress outputs ->", out_dir)
    print(f"Using beskrivelser file -> {Path(beskrivelser_file)}")
    print(f"Valid events: {len(valid)}")

    C_SM = "tab:blue"
    C_GW = "tab:orange"
    C_DEF = "darkgreen"
    C_STRESS_COMBO = "black"

    C_BASE = "tab:orange"
    C_GRAD = "tab:green"
    C_PREC = "tab:red"
    C_SWE = "tab:cyan"
    C_OF = "tab:purple"

    def add_stress_axis(ax, t, stress_total, base_s, grad_s, prec_s, swe_s, of_s, show_legend=False):
        ax2 = ax.twinx()
        l_total, = ax2.plot(t, stress_total, linestyle="-", lw=2.0, color=C_STRESS_COMBO, label="Stress (total)")
        l_base, = ax2.plot(t, base_s, linestyle=":", lw=1.2, color=C_BASE, alpha=1, label="Base stress")
        l_grad, = ax2.plot(t, grad_s, linestyle=":", lw=1.2, color=C_GRAD, alpha=1, label="Grad stress")
        l_prec, = ax2.plot(t, prec_s, linestyle=":", lw=1.2, color=C_PREC, alpha=1, label="Prec stress")
        l_swe, = ax2.plot(t, swe_s, linestyle=":", lw=1.2, color=C_SWE, alpha=1, label="SWE stress")
        l_of, = ax2.plot(t, of_s, linestyle=":", lw=1.2, color=C_OF, alpha=1, label="OF stress")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Component stress (0–1)")

        if show_legend:
            h1, l1 = ax.get_legend_handles_labels()
            h2 = [l_total, l_base, l_grad, l_prec, l_swe, l_of]
            l2 = [h.get_label() for h in h2]
            ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)

        return ax2

    for _, ev in valid.iterrows():
        guid = ev["GUID"]
        no = ev.get("No", None)
        skredid = ev.get("SkredID", None)
        event_time = ev.get("event_time", pd.NaT)
        if pd.isna(event_time):
            continue
        event_time = pd.to_datetime(event_time).floor("h")

        sim_path = simres_paths.get(guid)
        param_path = param_paths.get(guid)
        if sim_path is None or param_path is None:
            continue

        try:
            df = pd.read_csv(sim_path, sep=";", low_memory=False, usecols=list(cfg.usecols))
        except Exception:
            df = pd.read_csv(sim_path, sep=";", low_memory=False)

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

        for col in ["Prec_mm", "Temp_C", "Q_m3s", "Q_OF_m3s", "SubsurfaceDef_P_mm", "GrwPlus_P_mm", "SWE_P_mm", "Soilmoist_P_mm"]:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
        if df.empty:
            continue

        dt_hours = _median_dt_hours(df["Datetime"])

        if (df["Datetime"] == event_time).any():
            event_idx = df.index[df["Datetime"] == event_time][0]
        else:
            event_idx = (df["Datetime"] - event_time).abs().idxmin()
        ts_event = df.loc[event_idx, "Datetime"]

        before_steps = max(int(round(cfg.hours_before / dt_hours)), 1)
        after_steps = max(int(round(cfg.hours_after / dt_hours)), 1)
        start_idx = max(event_idx - before_steps, 0)
        end_idx = min(event_idx + after_steps, len(df) - 1)
        win = df.iloc[start_idx:end_idx + 1].copy()
        win_idx = win.index

        _ = load_param_values(param_path)
        M = df["SubsurfaceDef_P_mm"].iloc[0] if "SubsurfaceDef_P_mm" in df.columns else np.nan

        gw_series = df["GrwPlus_P_mm"] if "GrwPlus_P_mm" in df.columns else pd.Series(np.nan, index=df.index)
        sm_series = df["Soilmoist_P_mm"] if "Soilmoist_P_mm" in df.columns else pd.Series(np.nan, index=df.index)
        of_series = df["Q_OF_m3s"] if "Q_OF_m3s" in df.columns else pd.Series(np.nan, index=df.index)

        # ============================================================
        # THRESHOLDS + RAW EXCESS
        # ============================================================
        if cfg.of_use_zero_threshold:
            of_threshold = 0.0
        else:
            of_threshold = choose_threshold(
                of_series,
                M=None,
                use_capacity_fraction=False,
                cap_frac=1.0,
                use_quantile=cfg.of_use_quantile_threshold,
                q=cfg.of_q_threshold,
            )

        gw_threshold = np.nan
        sm_threshold = np.nan
        base_storage_threshold = np.nan

        if cfg.use_capacity_fraction and cfg.include_gw_base and cfg.include_sm_base:
            storage = pd.to_numeric(gw_series, errors="coerce").fillna(0.0) + pd.to_numeric(sm_series, errors="coerce").fillna(0.0)
            base_storage_threshold = choose_threshold(
                storage,
                M=M,
                use_capacity_fraction=True,
                cap_frac=cfg.capacity_frac,
                use_quantile=False,
                q=cfg.gw_q_threshold,
            )
            base_raw = build_base_excess(storage, base_storage_threshold)

            gw_base_raw = base_raw
            sm_base_raw = pd.Series(0.0, index=df.index)

            gw_threshold = base_storage_threshold
            sm_threshold = np.nan
        else:
            gw_threshold = choose_threshold(
                gw_series,
                M=M,
                use_capacity_fraction=cfg.use_capacity_fraction,
                cap_frac=cfg.capacity_frac,
                use_quantile=cfg.use_quantile_threshold,
                q=cfg.gw_q_threshold,
            )
            sm_threshold = choose_threshold(
                sm_series,
                M=None,
                use_capacity_fraction=False,
                cap_frac=1.0,
                use_quantile=cfg.sm_use_quantile_threshold,
                q=cfg.sm_q_threshold,
            )

            gw_base_raw = build_base_excess(gw_series, gw_threshold) if cfg.include_gw_base else pd.Series(0.0, index=df.index)
            sm_base_raw = build_base_excess(sm_series, sm_threshold) if cfg.include_sm_base else pd.Series(0.0, index=df.index)

        of_raw = build_base_excess(of_series, of_threshold) if cfg.include_of_base else pd.Series(0.0, index=df.index)

        # gradients
        gw_pos_raw, gw_neg_raw, gw_dXdt, gw_thr_pos, gw_thr_neg = build_gradient_component(
            gw_series, dt_hours,
            grad_hours=cfg.grad_hours,
            include_pos=cfg.include_pos_grad,
            include_neg=cfg.include_neg_grad,
            threshold_mode=cfg.grad_threshold_mode,
            pos_q=cfg.pos_grad_q,
            neg_q=cfg.neg_grad_q,
            pos_frac_of_max=cfg.pos_grad_frac_of_max,
            neg_frac_of_max=cfg.neg_grad_frac_of_max,
        )
        sm_pos_raw, sm_neg_raw, sm_dXdt, sm_thr_pos, sm_thr_neg = build_gradient_component(
            sm_series, dt_hours,
            grad_hours=cfg.grad_hours,
            include_pos=cfg.include_pos_grad,
            include_neg=cfg.include_neg_grad,
            threshold_mode=cfg.grad_threshold_mode,
            pos_q=cfg.pos_grad_q,
            neg_q=cfg.neg_grad_q,
            pos_frac_of_max=cfg.pos_grad_frac_of_max,
            neg_frac_of_max=cfg.neg_grad_frac_of_max,
        )

        # forcing
        prec_raw, Pwin = build_prec_component(df, dt_hours, cfg) if cfg.include_prec else (pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index))
        swe_raw, melt_win = build_swe_melt_component(df, dt_hours, cfg) if cfg.include_swe else (pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index))

        # ------------------------------------------------------------
        # 1) Instantaneous exceedance normalization
        # ------------------------------------------------------------
        gw_n = robust_minmax(gw_base_raw, cfg.norm_q_low, cfg.norm_q_high)
        sm_n = robust_minmax(sm_base_raw, cfg.norm_q_low, cfg.norm_q_high)

        grad_raw = (cfg.pos_grad_mult * (gw_pos_raw + sm_pos_raw)) + (cfg.neg_grad_mult * (gw_neg_raw + sm_neg_raw))
        grad_n = robust_minmax(grad_raw, cfg.norm_q_low, cfg.norm_q_high)

        prec_n = robust_minmax(prec_raw, cfg.norm_q_low, cfg.norm_q_high)
        swe_n = robust_minmax(swe_raw, cfg.norm_q_low, cfg.norm_q_high)
        of_n = robust_minmax(of_raw, cfg.norm_q_low, cfg.norm_q_high)

        # ------------------------------------------------------------
        # 2) Build instantaneous component signals BEFORE memory
        # ------------------------------------------------------------
        gw_inst = cfg.gw_base_mult * gw_n
        sm_inst = cfg.sm_base_mult * sm_n

        base_sum = gw_inst + sm_inst
        if cfg.combine_mode.lower() == "avg":
            n_base = int(cfg.include_gw_base) + int(cfg.include_sm_base)
            n_base = max(n_base, 1)
            base_inst = base_sum / float(n_base)
        else:
            base_inst = base_sum

        grad_inst = grad_n
        prec_inst = prec_n
        swe_inst = swe_n
        of_inst = cfg.of_base_mult * of_n

        # ------------------------------------------------------------
        # 3) Apply memory per component
        # ------------------------------------------------------------
        base_mem = accumulate_inst(base_inst, dt_hours, mode=cfg.base_accum_mode, window_days=cfg.base_window_days, tau_days=cfg.base_tau_days)
        grad_mem = accumulate_inst(grad_inst, dt_hours, mode=cfg.grad_accum_mode, window_days=cfg.grad_window_days, tau_days=cfg.grad_tau_days)

        prec_mem = accumulate_inst(prec_inst, dt_hours, mode=cfg.forc_accum_mode, window_days=cfg.forc_window_days, tau_days=cfg.forc_tau_days)
        swe_mem = accumulate_inst(swe_inst, dt_hours, mode=cfg.forc_accum_mode, window_days=cfg.forc_window_days, tau_days=cfg.forc_tau_days)
        of_mem = accumulate_inst(of_inst, dt_hours, mode=cfg.forc_accum_mode, window_days=cfg.forc_window_days, tau_days=cfg.forc_tau_days)

        # ------------------------------------------------------------
        # 4) Normalize each accumulated component to 0–1
        # ------------------------------------------------------------
        base_stress = robust_minmax(base_mem, cfg.norm_q_low, cfg.norm_q_high).fillna(0.0)
        grad_stress = robust_minmax(grad_mem, cfg.norm_q_low, cfg.norm_q_high).fillna(0.0)
        prec_stress = robust_minmax(prec_mem, cfg.norm_q_low, cfg.norm_q_high).fillna(0.0)
        swe_stress = robust_minmax(swe_mem, cfg.norm_q_low, cfg.norm_q_high).fillna(0.0)
        of_stress = robust_minmax(of_mem, cfg.norm_q_low, cfg.norm_q_high).fillna(0.0)

        if not (cfg.include_gw_base or cfg.include_sm_base):
            base_stress[:] = 0.0
        if not (cfg.include_pos_grad or cfg.include_neg_grad):
            grad_stress[:] = 0.0
        if not cfg.include_prec:
            prec_stress[:] = 0.0
        if not cfg.include_swe:
            swe_stress[:] = 0.0
        if not cfg.include_of_base:
            of_stress[:] = 0.0

        # ------------------------------------------------------------
        # 5) Final combination
        # ------------------------------------------------------------
        w_base = float(cfg.base_mult) if (cfg.include_gw_base or cfg.include_sm_base) else 0.0
        w_grad = float(cfg.grad_mult) if (cfg.include_pos_grad or cfg.include_neg_grad) else 0.0
        w_prec = float(cfg.prec_mult) if cfg.include_prec else 0.0
        w_swe = float(cfg.swe_mult) if cfg.include_swe else 0.0
        w_of = float(cfg.of_mult) if cfg.include_of_base else 0.0

        stress, stress_raw_final = combine_final_stress(
            base_stress=base_stress,
            grad_stress=grad_stress,
            prec_stress=prec_stress,
            swe_stress=swe_stress,
            of_stress=of_stress,
            w_base=w_base,
            w_grad=w_grad,
            w_prec=w_prec,
            w_swe=w_swe,
            w_of=w_of,
            mode=cfg.final_combine_mode,
            power=cfg.final_combine_power,
            q_low=cfg.norm_q_low,
            q_high=cfg.norm_q_high,
        )

        # ============================================================
        # PLOTS
        # ============================================================
        fig = plt.figure(figsize=cfg.figsize)
        gs = gridspec.GridSpec(5, 2, height_ratios=[1.3, 1, 1, 1, 1])

        ax_top_full = fig.add_subplot(gs[0, :])
        ax_prec = fig.add_subplot(gs[1, 0])
        ax_temp = fig.add_subplot(gs[1, 1])
        ax_runoff = fig.add_subplot(gs[2, 0])
        ax_grw = fig.add_subplot(gs[2, 1])
        ax_sm = fig.add_subplot(gs[3, 0])
        ax_hyst = fig.add_subplot(gs[3, 1])
        ax_swe = fig.add_subplot(gs[4, 0])
        ax_qof = fig.add_subplot(gs[4, 1])

        locator_full = mdates.AutoDateLocator()
        formatter_full = mdates.AutoDateFormatter(locator_full)

        locator_win = mdates.HourLocator(interval=cfg.zoom_hour_tick_interval)
        formatter_win = mdates.DateFormatter("%d %H")

        ax_top_full.plot(df["Datetime"], stress, lw=1.2, color=C_STRESS_COMBO, label="Stress (Full time series)")
        ax_top_full.axvline(ts_event, color="red", linestyle=":", lw=1.5)
        stress_evt_val = stress.iloc[event_idx] if event_idx < len(stress) else np.nan
        if pd.notna(stress_evt_val):
            ax_top_full.plot(ts_event, stress_evt_val, "o", color="red", ms=6, zorder=10)

        ax_top_full.set_ylabel("Stress (0–1)")
        ax_top_full.set_ylim(0, 1.05)
        ax_top_full.xaxis.set_major_locator(locator_full)
        ax_top_full.xaxis.set_major_formatter(formatter_full)
        plt.setp(ax_top_full.get_xticklabels(), rotation=30, ha="right")
        ax_top_full.legend(loc="upper left")

        ax_top_full.set_title(
            f"Stress | GUID={guid} | No={no} | SkredID={skredid}\n"
        )

        t_win = win["Datetime"]

        if "Prec_mm" in win.columns:
            ax_prec.vlines(t_win, [0], win["Prec_mm"], lw=2, label="Precip")
            ax_prec.axvline(ts_event, color="red", linestyle=":", lw=1.5)
            p_evt = df.loc[event_idx, "Prec_mm"] if "Prec_mm" in df.columns else np.nan
            if pd.notna(p_evt):
                ax_prec.plot(ts_event, p_evt, "o", color="red", ms=6, zorder=10)
        ax_prec.set_title("Precip")
        ax_prec.set_ylabel("Precip [mm]")
        ax_prec.xaxis.set_major_locator(locator_win)
        ax_prec.xaxis.set_major_formatter(formatter_win)
        ax_prec.tick_params(axis="x", rotation=45)

        if "Temp_C" in win.columns:
            ax_temp.plot(t_win, win["Temp_C"], lw=2, label="Temp")
            ax_temp.axhline(0.0, lw=1.2)
            ax_temp.axvline(ts_event, color="red", linestyle=":", lw=1.5)
            t_evt = df.loc[event_idx, "Temp_C"] if "Temp_C" in df.columns else np.nan
            if pd.notna(t_evt):
                ax_temp.plot(ts_event, t_evt, "o", color="red", ms=6, zorder=10)
        ax_temp.set_title("Temperature")
        ax_temp.set_ylabel("Temp [°C]")
        ax_temp.xaxis.set_major_locator(locator_win)
        ax_temp.xaxis.set_major_formatter(formatter_win)
        ax_temp.tick_params(axis="x", rotation=45)

        add_stress_axis(
            ax_temp, t_win,
            stress.loc[win_idx], base_stress.loc[win_idx], grad_stress.loc[win_idx], prec_stress.loc[win_idx], swe_stress.loc[win_idx], of_stress.loc[win_idx],
            show_legend=True
        )

        if "Q_m3s" in win.columns:
            ax_runoff.plot(t_win, win["Q_m3s"], lw=2, label="Runoff")
            ax_runoff.axvline(ts_event, color="red", linestyle=":", lw=1.5)
            q_evt = df.loc[event_idx, "Q_m3s"] if "Q_m3s" in df.columns else np.nan
            if pd.notna(q_evt):
                ax_runoff.plot(ts_event, q_evt, "o", color="red", ms=6, zorder=10)
        ax_runoff.set_title("Runoff (solid blue)")
        ax_runoff.set_ylabel("Runoff [m³/s]")
        ax_runoff.xaxis.set_major_locator(locator_win)
        ax_runoff.xaxis.set_major_formatter(formatter_win)
        ax_runoff.tick_params(axis="x", rotation=45)

        add_stress_axis(
            ax_runoff, t_win,
            stress.loc[win_idx], base_stress.loc[win_idx], grad_stress.loc[win_idx], prec_stress.loc[win_idx], swe_stress.loc[win_idx], of_stress.loc[win_idx],
            show_legend=False
        )

        if "SubsurfaceDef_P_mm" in win.columns:
            ax_grw.plot(t_win, win["SubsurfaceDef_P_mm"], lw=2, color=C_DEF, label="GW deficit")
        if "GrwPlus_P_mm" in win.columns:
            ax_grw.plot(t_win, win["GrwPlus_P_mm"], lw=2, color=C_GW, label="GW")
        if pd.notna(M):
            ax_grw.axhline(M, linestyle="--", lw=1.2, color="grey", label="Capacity M")
        ax_grw.axvline(ts_event, color="red", linestyle=":", lw=1.5)
        gw_evt_val = df.loc[event_idx, "GrwPlus_P_mm"] if "GrwPlus_P_mm" in df.columns else np.nan
        if pd.notna(gw_evt_val):
            ax_grw.plot(ts_event, gw_evt_val, "o", color="red", ms=6, zorder=10)

        ax_grw.set_title("GW deficit (solid green) + GW (solid orange)")
        ax_grw.set_ylabel("mm")
        ax_grw.xaxis.set_major_locator(locator_win)
        ax_grw.xaxis.set_major_formatter(formatter_win)
        ax_grw.tick_params(axis="x", rotation=45)

        add_stress_axis(
            ax_grw, t_win,
            stress.loc[win_idx], base_stress.loc[win_idx], grad_stress.loc[win_idx], prec_stress.loc[win_idx], swe_stress.loc[win_idx], of_stress.loc[win_idx],
            show_legend=False
        )

        if "Soilmoist_P_mm" in win.columns:
            ax_sm.plot(t_win, win["Soilmoist_P_mm"], lw=2, color=C_SM, label="SM")
        ax_sm.axvline(ts_event, color="red", linestyle=":", lw=1.5)
        sm_evt_val = df.loc[event_idx, "Soilmoist_P_mm"] if "Soilmoist_P_mm" in df.columns else np.nan
        if pd.notna(sm_evt_val):
            ax_sm.plot(ts_event, sm_evt_val, "o", color="red", ms=6, zorder=10)
        ax_sm.set_title("Soil moisture (solid blue)")
        ax_sm.set_ylabel("SM [mm]")
        ax_sm.xaxis.set_major_locator(locator_win)
        ax_sm.xaxis.set_major_formatter(formatter_win)
        ax_sm.tick_params(axis="x", rotation=45)

        add_stress_axis(
            ax_sm, t_win,
            stress.loc[win_idx], base_stress.loc[win_idx], grad_stress.loc[win_idx], prec_stress.loc[win_idx], swe_stress.loc[win_idx], of_stress.loc[win_idx],
            show_legend=False
        )

        if ("Q_m3s" in df.columns) and ("GrwPlus_P_mm" in df.columns):
            post_steps = max(int(round(cfg.post_hyst_hours / dt_hours)), 1)
            post_end = min(event_idx + post_steps, len(df) - 1)
            pre = df.iloc[start_idx:event_idx + 1]
            post = df.iloc[event_idx:post_end + 1]
            ax_hyst.scatter(pre["Q_m3s"], pre["GrwPlus_P_mm"], s=15, label="Before / up to event")
            ax_hyst.scatter(post["Q_m3s"], post["GrwPlus_P_mm"], s=15, color="red", label="After event")
            ax_hyst.scatter(
                df.loc[event_idx, "Q_m3s"],
                df.loc[event_idx, "GrwPlus_P_mm"],
                s=120,
                color="green",
                edgecolor="black",
                zorder=10,
                label="Event time",
            )
            ax_hyst.set_xlabel("Runoff [m³/s]")
            ax_hyst.set_ylabel("GW [mm]")
            ax_hyst.set_title("Hysteresis GW vs Q")
            ax_hyst.legend(loc="best")
        else:
            ax_hyst.axis("off")

        if "SWE_P_mm" in win.columns:
            ax_swe.plot(t_win, win["SWE_P_mm"], lw=2, label="SWE")
        ax_swe.axvline(ts_event, color="red", linestyle=":", lw=1.5)
        swe_evt = df.loc[event_idx, "SWE_P_mm"] if "SWE_P_mm" in df.columns else np.nan
        if pd.notna(swe_evt):
            ax_swe.plot(ts_event, swe_evt, "o", color="red", ms=6, zorder=10)
        ax_swe.set_title("SWE")
        ax_swe.set_ylabel("SWE [mm]")
        ax_swe.xaxis.set_major_locator(locator_win)
        ax_swe.xaxis.set_major_formatter(formatter_win)
        ax_swe.tick_params(axis="x", rotation=45)

        if "Q_OF_m3s" in win.columns:
            ax_qof.plot(t_win, win["Q_OF_m3s"], lw=2, label="OF")
        ax_qof.axvline(ts_event, color="red", linestyle=":", lw=1.5)
        of_evt = df.loc[event_idx, "Q_OF_m3s"] if "Q_OF_m3s" in df.columns else np.nan
        if pd.notna(of_evt):
            ax_qof.plot(ts_event, of_evt, "o", color="red", ms=6, zorder=10)
        ax_qof.set_title("Overland flow (solid blue)")
        ax_qof.set_ylabel("OF [m³/s]")
        ax_qof.xaxis.set_major_locator(locator_win)
        ax_qof.xaxis.set_major_formatter(formatter_win)
        ax_qof.tick_params(axis="x", rotation=45)

        add_stress_axis(
            ax_qof, t_win,
            stress.loc[win_idx], base_stress.loc[win_idx], grad_stress.loc[win_idx], prec_stress.loc[win_idx], swe_stress.loc[win_idx], of_stress.loc[win_idx],
            show_legend=False
        )

        fig.tight_layout()

        no_str = f"No{no}" if (no is not None and not pd.isna(no)) else "NoNA"
        out_png = out_dir / f"{guid}_{no_str}_stress.png"
        fig.savefig(out_png, dpi=cfg.dpi)
        plt.close(fig)

        # ============================================================
        # EXPORT
        # ============================================================
        out_df = pd.DataFrame({
            "Datetime": df["Datetime"],
            "is_event_time": (df["Datetime"] == ts_event),
            "event_time": ts_event,

            "GUID": guid,
            "No": no,
            "SkredID": skredid,

            "GW_mm": df.get("GrwPlus_P_mm"),
            "SM_mm": df.get("Soilmoist_P_mm"),
            "OF_m3s": df.get("Q_OF_m3s"),
            "SWE_mm": df.get("SWE_P_mm"),
            "Prec_mm": df.get("Prec_mm"),
            "Temp_C": df.get("Temp_C"),
            "Q_m3s": df.get("Q_m3s"),
            "GW_def_mm": df.get("SubsurfaceDef_P_mm"),

            "stress_total": stress,
            "stress_total_raw_pre_final_norm": stress_raw_final,
            "stress_base": base_stress,
            "stress_grad": grad_stress,
            "stress_prec": prec_stress,
            "stress_swe": swe_stress,
            "stress_of": of_stress,

            "gw_norm": gw_n,
            "sm_norm": sm_n,
            "grad_norm": grad_n,
            "prec_norm": prec_n,
            "swe_norm": swe_n,
            "of_norm": of_n,

            "base_mem": base_mem,
            "grad_mem": grad_mem,
            "prec_mem": prec_mem,
            "swe_mem": swe_mem,
            "of_mem": of_mem,
        })

        if cfg.export_debug_intensities:
            out_df["prec_window_mm"] = Pwin
            out_df["swe_melt_window_mm"] = melt_win
            out_df["GW_dXdt_unitsday"] = gw_dXdt
            out_df["SM_dXdt_unitsday"] = sm_dXdt
            out_df["GW_pos_excess"] = gw_pos_raw
            out_df["GW_neg_excess"] = gw_neg_raw
            out_df["SM_pos_excess"] = sm_pos_raw
            out_df["SM_neg_excess"] = sm_neg_raw
            out_df["prec_excess_mm"] = prec_raw
            out_df["swe_excess_mm"] = swe_raw
            out_df["of_excess"] = of_raw

        if cfg.export_debug_thresholds:
            out_df["thr_GW_base"] = gw_threshold
            out_df["thr_SM_base"] = sm_threshold
            out_df["thr_BASE_storage"] = base_storage_threshold
            out_df["thr_OF"] = of_threshold

            if cfg.include_prec and Pwin.notna().any():
                if cfg.prec_threshold_mode == "fraction_of_max":
                    out_df["thr_prec_window"] = float(Pwin.max(skipna=True)) * cfg.prec_frac_of_max
                else:
                    out_df["thr_prec_window"] = float(Pwin.quantile(cfg.prec_q))
            else:
                out_df["thr_prec_window"] = np.nan

            if cfg.include_swe and melt_win.notna().any():
                if cfg.swe_threshold_mode == "fraction_of_max":
                    out_df["thr_swe_window"] = float(melt_win.max(skipna=True)) * cfg.swe_frac_of_max
                else:
                    out_df["thr_swe_window"] = float(melt_win.quantile(cfg.swe_q))
            else:
                out_df["thr_swe_window"] = np.nan

            out_df["thr_GW_posgrad"] = gw_thr_pos
            out_df["thr_GW_neggrad"] = gw_thr_neg
            out_df["thr_SM_posgrad"] = sm_thr_pos
            out_df["thr_SM_neggrad"] = sm_thr_neg

        base_name = f"{guid}_{no_str}_StressComponents"
        if cfg.export_format.lower() == "xlsx":
            out_path = out_dir / f"{base_name}.xlsx"
            out_df.to_excel(out_path, index=False)
        else:
            out_path = out_dir / f"{base_name}.csv"
            out_df.to_csv(out_path, index=False)

        del out_df, df, win
        gc.collect()

    print("Done. Saved stress plots + exports in:", out_dir)
