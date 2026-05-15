from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from functions.io_search import find_sim_param_files, find_beskrivelser_files
from functions.metadata import load_beskrivelser_events
from functions.plotting import plot_event_panels
from functions.stats import compute_episode_stats

from functions.run_config import RunConfig


def prepare_inputs(
    cfg: RunConfig,
    beskrivelser_files: list[Path] | None = None
) -> Tuple[Dict[str, Path], Dict[str, Path], pd.DataFrame]:

    simres_paths, param_paths = find_sim_param_files(cfg.input_dir, tr=cfg.tr)

    if beskrivelser_files is None:
        beskrivelser_files = find_beskrivelser_files(cfg.base_dir)

    meta_events = load_beskrivelser_events(beskrivelser_files)

    valid = meta_events[
        meta_events["GUID"].isin(simres_paths.keys())
        & meta_events["GUID"].isin(param_paths.keys())
    ].copy()

    return simres_paths, param_paths, valid

def run_event_plots(
    cfg: RunConfig,
    simres_paths: Dict[str, Path],
    param_paths: Dict[str, Path],
    valid_events: pd.DataFrame,
) -> List[Path]:
    cfg.ensure_dirs()
    saved = []

    for _, ev in valid_events.iterrows():
        guid = ev["GUID"]
        out = plot_event_panels(
            guid=guid,
            simres_path=simres_paths[guid],
            param_path=param_paths[guid],
            event_info=ev,
            tr=cfg.tr,
            out_dir=cfg.plots_dir,
            hours_before=cfg.hours_before,
            hours_after=cfg.hours_after,
            post_hyst_hours=cfg.post_hyst_hours,
        )
        saved.append(out)

    return saved


def run_event_stats(
    cfg: RunConfig,
    simres_paths: Dict[str, Path],
    param_paths: Dict[str, Path],
    valid_events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, ev in valid_events.iterrows():
        guid = ev["GUID"]
        row = compute_event_stats(
            guid=guid,
            sim_path=str(simres_paths[guid]),
            param_path=str(param_paths[guid]),
            ev=ev,
            window_hours=cfg.window_hours,
            fast_window_h=cfg.fast_window_h,
            post_long_h=cfg.post_long_h,
            post_short_h=cfg.post_short_h,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_run_metadata(cfg: RunConfig) -> Path:
    cfg.ensure_dirs()
    out = cfg.results_dir / "run_config.csv"
    pd.Series(asdict(cfg)).to_csv(out, header=False)
    return out
