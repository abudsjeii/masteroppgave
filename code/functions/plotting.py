from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from .params import load_param_values


def _force_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_event_panels(
    guid: str,
    simres_path: str | Path,
    param_path: str | Path,
    event_info: Dict[str, Any] | pd.Series,
    tr: str,
    out_dir: str | Path,
    hours_before: int = 48,
    hours_after: int = 12,
    post_hyst_hours: int = 72,
) -> Path:
    """
    Replicate the R-script style multi-panel plot for ONE event.

    Panels:
      - Full time series SS (GrwPlus_P_mm) + capacity line + vertical event line
      - 4x2 grid (prec, temp, runoff, deficit+SS, soilmoist, hysteresis, SWE, overland flow)

    event_info must include:
      - event_time (datetime)
      - optional: No, SkredID, MeanAnnualFlood

    Returns:
      output path to saved PNG.
    """
    ev = dict(event_info)
    no = ev.get("No")
    skredid = ev.get("SkredID")
    maf = ev.get("MeanAnnualFlood")
    event_time = ev.get("event_time")

    if pd.isna(event_time):
        raise ValueError(f"Missing event_time for GUID={guid}")

    event_time = pd.to_datetime(event_time).floor("h")

    df = pd.read_csv(simres_path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    num_cols = ["Prec_mm","Temp_C","Q_m3s","Q_OF_m3s","SubsurfaceDef_P_mm","GrwPlus_P_mm","SWE_P_mm","Soilmoist_P_mm"]
    df = _force_numeric(df, num_cols)

    if (df["Datetime"] == event_time).any():
        event_idx = df.index[df["Datetime"] == event_time][0]
    else:
        event_idx = (df["Datetime"] - event_time).abs().idxmin()

    start_idx = max(event_idx - hours_before, 0)
    slutt_idx = min(event_idx + hours_after, len(df) - 1)

    land_idx = max(slutt_idx - hours_after, start_idx)
    event_time_marker = df.loc[land_idx, "Datetime"]

    window = df.iloc[start_idx:slutt_idx + 1].copy()
    t = window["Datetime"]

    prec = window.get("Prec_mm")
    temp = window.get("Temp_C")
    Qtot = window.get("Q_m3s")
    QOF = window.get("Q_OF_m3s")
    SSdef = window.get("SubsurfaceDef_P_mm")
    SS = window.get("GrwPlus_P_mm")
    SWE = window.get("SWE_P_mm")
    SM = window.get("Soilmoist_P_mm")

    M = df["SubsurfaceDef_P_mm"].iloc[0] if "SubsurfaceDef_P_mm" in df.columns else np.nan

    param_vals = load_param_values(param_path)
    area = param_vals.get("Area_km2")
    HI = param_vals.get("HI")
    area_str = f"{area:.2f}" if isinstance(area, (int, float)) else "NA"
    HI_str = f"{HI:.2f}" if isinstance(HI, (int, float)) else "NA"

    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(5, 2, height_ratios=[1.3, 1, 1, 1, 1])

    ax_gw_full = fig.add_subplot(gs[0, :])
    ax_prec = fig.add_subplot(gs[1, 0])
    ax_temp = fig.add_subplot(gs[1, 1])
    ax_runoff = fig.add_subplot(gs[2, 0])
    ax_grw = fig.add_subplot(gs[2, 1])
    ax_sm = fig.add_subplot(gs[3, 0])
    ax_hyst = fig.add_subplot(gs[3, 1])
    ax_swe = fig.add_subplot(gs[4, 0])
    ax_qof = fig.add_subplot(gs[4, 1])

    ax_gw_full.plot(df["Datetime"], df.get("GrwPlus_P_mm"), "-", lw=1.5)
    ax_gw_full.set_ylabel("Groundwater SS [mm]")

    full_SS = df.get("GrwPlus_P_mm")
    if full_SS is not None and full_SS.notna().any():
        ymax_full = float(full_SS.max()) * 1.1
    else:
        ymax_full = 1.0
    ax_gw_full.set_ylim(0, ymax_full)

    if pd.notna(M):
        ax_gw_full.axhline(M, linestyle="--", lw=1.5, color="grey")

    ax_gw_full.axvline(event_time_marker, linestyle=":", lw=1.5, color="red")
    gw_event_val = df.loc[land_idx, "GrwPlus_P_mm"] if "GrwPlus_P_mm" in df.columns else np.nan
    if pd.notna(gw_event_val):
        ax_gw_full.plot(event_time_marker, gw_event_val, "o", ms=6, color="red")

    ax_gw_full.set_title(
        f"Full groundwater time series (SS)\n"
        f"GUID={guid}, No={no}, SkredID={skredid}, Capacity M={M:.1f} mm" if pd.notna(M)
        else f"Full groundwater time series (SS)\nGUID={guid}, No={no}, SkredID={skredid}"
    )

    loc_full = mdates.AutoDateLocator()
    fmt_full = mdates.AutoDateFormatter(loc_full)
    ax_gw_full.xaxis.set_major_locator(loc_full)
    ax_gw_full.xaxis.set_major_formatter(fmt_full)
    plt.setp(ax_gw_full.get_xticklabels(), rotation=30, ha="right")

    locator = mdates.HourLocator(interval=5)
    formatter = mdates.DateFormatter("%d%H")

    if prec is not None:
        ax_prec.vlines(t, [0], prec, lw=2)
        ax_prec.set_ylim(0, (float(prec.max()) * 1.1) if prec.notna().any() else 1.0)
    ax_prec.set_ylabel("Precip [mm]")
    ax_prec.set_title(f"{guid} – No {no}\nEvent: {event_time_marker:%Y-%m-%d %H:%M}")
    ax_prec.xaxis.set_major_locator(locator)
    ax_prec.xaxis.set_major_formatter(formatter)
    ax_prec.tick_params(axis="x", rotation=45)
    ax_prec.plot(event_time_marker, 3, "ro", ms=6)

    if temp is not None:
        ax_temp.plot(t, temp, "-", lw=2)
        tmin = float(temp.min()) if temp.notna().any() else -5.0
        tmax = float(temp.max()) if temp.notna().any() else 5.0
        ax_temp.set_ylim(min(-5.0, tmin), tmax * 1.1)
    ax_temp.set_ylabel("Temperature [°C]")
    ax_temp.set_title(f"Area={area_str} km², HI={HI_str}\nRed dot = landslide time")
    ax_temp.xaxis.set_major_locator(locator)
    ax_temp.xaxis.set_major_formatter(formatter)
    ax_temp.tick_params(axis="x", rotation=45)
    ax_temp.plot(event_time_marker, 3, "ro", ms=6)
    ax_temp.axhline(0.0, color="k", lw=1.5)

    if Qtot is not None:
        ax_runoff.plot(t, Qtot, "-", lw=2)
        qmax = float(Qtot.max()) if Qtot.notna().any() else 1.0
    else:
        qmax = 1.0
    if maf is not None and not pd.isna(maf):
        ymax = max(float(maf) + 0.1, qmax)
        ax_runoff.axhline(float(maf), linestyle="--", lw=1.5)
        ax_runoff.set_title(f"Runoff. MeanAnnualFlood (dashed)={maf} m³/s")
    else:
        ymax = qmax
        ax_runoff.set_title("Runoff (MeanAnnualFlood not available)")
    ax_runoff.set_ylabel("Runoff [m³/s]")
    ax_runoff.set_ylim(0, ymax * 1.05 if np.isfinite(ymax) and ymax > 0 else 1.0)
    ax_runoff.xaxis.set_major_locator(locator)
    ax_runoff.xaxis.set_major_formatter(formatter)
    ax_runoff.tick_params(axis="x", rotation=45)
    if Qtot is not None and Qtot.notna().any():
        ax_runoff.plot(event_time_marker, float(Qtot.max()) - 0.1, "ro", ms=6)

    if SSdef is not None:
        ax_grw.plot(t, SSdef, "-", lw=2, color="darkgreen", label="Groundwater deficit")
    if SS is not None:
        ax_grw.plot(t, SS, "-", lw=2, color="darkorange", label="Subsurface waterplus")
    ax_grw.set_ylabel("GrwDef [mm]")
    if pd.notna(M) and np.isfinite(M) and M > 0:
        ax_grw.set_ylim(0, 1.2 * M)
        ax_grw.axhline(M, linestyle="--", lw=1.5)
    ax_grw.set_title("Groundwater deficit (green) & Groundwater (orange)\nDashed line = ss capacity")
    ax_grw.xaxis.set_major_locator(locator)
    ax_grw.xaxis.set_major_formatter(formatter)
    ax_grw.tick_params(axis="x", rotation=45)
    ax_grw.plot(event_time_marker, 3, "ro", ms=6)

    if SM is not None:
        ax_sm.plot(t, SM, "-", lw=2)
        sm_max = float(SM.max()) if SM.notna().any() else 1.0
        ax_sm.set_ylim(0, sm_max * 1.1 if sm_max > 0 else 1.0)
    ax_sm.set_ylabel("Soil moisture [mm]")
    ax_sm.set_title("Soil moisture")
    ax_sm.xaxis.set_major_locator(locator)
    ax_sm.xaxis.set_major_formatter(formatter)
    ax_sm.tick_params(axis="x", rotation=45)
    if "Soilmoist_P_mm" in df.columns:
        sm_land = df.loc[land_idx, "Soilmoist_P_mm"]
        if pd.notna(sm_land):
            ax_sm.plot(event_time_marker, sm_land, "ro", ms=6)

    pre = df.iloc[start_idx:land_idx + 1]
    post_end = min(land_idx + post_hyst_hours, len(df) - 1)
    post = df.iloc[land_idx:post_end + 1]

    ax_hyst.scatter(pre["Q_m3s"], pre["GrwPlus_P_mm"], s=15, label="Before / up to event")
    ax_hyst.scatter(post["Q_m3s"], post["GrwPlus_P_mm"], s=15, color="red", label="After event")

    event_Q = df.loc[land_idx, "Q_m3s"]
    event_SS = df.loc[land_idx, "GrwPlus_P_mm"]
    if pd.notna(event_Q) and pd.notna(event_SS):
        ax_hyst.scatter(event_Q, event_SS, s=120, color="green", edgecolor="black", zorder=10, label="Landslide time")
    ax_hyst.set_xlabel("Runoff [m³/s]")
    ax_hyst.set_ylabel("Groundwater [mm]")

    Q_vals = pd.concat([pre["Q_m3s"], post["Q_m3s"]], ignore_index=True)
    SS_vals = pd.concat([pre["GrwPlus_P_mm"], post["GrwPlus_P_mm"]], ignore_index=True)
    x_min, x_max = Q_vals.min(), Q_vals.max()
    y_max_h = max(SS_vals.max(), 1.2 * M) if pd.notna(M) and np.isfinite(M) else SS_vals.max()
    ax_hyst.set_xlim(x_min, x_max)
    ax_hyst.set_ylim(0, y_max_h if np.isfinite(y_max_h) and y_max_h > 0 else 1.0)
    ax_hyst.set_title("Hysteresis SS vs Q\n(blue=before, red=after, green=landslide)")
    if pd.notna(M) and np.isfinite(M):
        ax_hyst.axhline(M, linestyle="--", lw=1.5)

    if SWE is not None:
        ax_swe.plot(t, SWE, "-", lw=2)
        swe_max = float(SWE.max()) if SWE.notna().any() else 1.0
        ax_swe.set_ylim(0, swe_max * 1.1 if swe_max > 0 else 1.0)
    ax_swe.set_ylabel("SWE [mm]")
    ax_swe.set_title("SWE")
    ax_swe.xaxis.set_major_locator(locator)
    ax_swe.xaxis.set_major_formatter(formatter)
    ax_swe.tick_params(axis="x", rotation=45)
    ax_swe.plot(event_time_marker, 0.5, "ro", ms=6)

    if QOF is not None:
        ax_qof.plot(t, QOF, "-", lw=2)
        qof_max = float(QOF.max()) if QOF.notna().any() else 1.0
        ax_qof.set_ylim(0, qof_max * 1.1 if qof_max > 0 else 1.0)
    ax_qof.set_ylabel("Overland flow [m³/s]")
    ax_qof.set_title("Overland flow")
    ax_qof.xaxis.set_major_locator(locator)
    ax_qof.xaxis.set_major_formatter(formatter)
    ax_qof.tick_params(axis="x", rotation=45)
    ax_qof.plot(event_time_marker, 0.1, "ro", ms=6)

    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    no_str = f"No{no}" if (no is not None and not pd.isna(no)) else "NoNA"
    out_file = out_dir / f"{no_str}_{guid}_{tr}_Skredanalyse.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

    return out_file