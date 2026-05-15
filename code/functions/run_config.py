# functions/run_config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    base_dir: Path
    input_dir: Path
    tr: str = "1H"

    # plotting windows
    hours_before: int = 48
    hours_after: int = 12
    post_hyst_hours: int = 72

    # stats windows
    window_hours: int = 48
    fast_window_h: int = 6
    post_long_h: int = 12
    post_short_h: int = 6

    @property
    def results_dir(self) -> Path:
        return self.input_dir.with_name(self.input_dir.name + "_results")

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
