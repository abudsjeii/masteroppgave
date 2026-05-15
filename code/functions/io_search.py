from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple


def find_sim_param_files(root_dir: str | Path, tr: str = "1H") -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Walk `root_dir` and find:
      - *_{TR}_simres.csv
      - *_{TR}_param.csv

    Returns:
      simres_paths: {GUID: Path}
      param_paths : {GUID: Path}

    Notes:
      - If duplicates exist in the folder structure, the *last one found* will overwrite earlier ones.
        If you want "first wins", change assignment to use `setdefault`.
    """
    root = Path(root_dir)
    simres_paths: Dict[str, Path] = {}
    param_paths: Dict[str, Path] = {}

    tr_lower = tr.lower()

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            lower = fname.lower()
            fpath = Path(dirpath) / fname

            if lower.endswith(f"_{tr_lower}_simres.csv"):
                guid = fname.split(f"_{tr}_simres.csv")[0]
                simres_paths[guid] = fpath

            elif lower.endswith(f"_{tr_lower}_param.csv"):
                guid = fname.split(f"_{tr}_param.csv")[0]
                param_paths[guid] = fpath

    return simres_paths, param_paths


def find_beskrivelser_files(root_dir: str | Path) -> List[Path]:
    """
    Walk `root_dir` and find any Excel file whose name contains 'Beskrivelser' (case-insensitive).
    Returns list of Paths.
    """
    root = Path(root_dir)
    out: List[Path] = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            lower = fname.lower()
            if "beskrivelser" in lower and lower.endswith(".xlsx"):
                out.append(Path(dirpath) / fname)

    return out