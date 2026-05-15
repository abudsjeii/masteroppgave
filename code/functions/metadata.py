from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _first_col_containing(df: pd.DataFrame, token: str) -> Optional[str]:
    token = token.lower()
    for c in df.columns:
        if token in str(c).lower():
            return c
    return None


def load_beskrivelser_events(beskrivelser_files: Iterable[str | Path]) -> pd.DataFrame:
    """
    Read all Beskrivelser workbooks and all sheets, extracting per-event metadata.

    Required (to define an event):
      - GUID column (contains 'guid')
      - Tidspunkt column (contains 'tidspunkt')

    Optional:
      - No column (contains 'no')
      - SkredID column (contains 'skredid')
      - Mean annual flood / middelflom column (contains 'middelflom' OR 'mean annual flood')

    Deduplication:
      - dedupe key: (GUID, SkredID)
      - keep the row with the most metadata filled (No, SkredID, MeanAnnualFlood)

    Returns DataFrame with:
      ['GUID', 'No', 'SkredID', 'MeanAnnualFlood', 'event_time']
    """
    rows = []

    for fpath in map(Path, beskrivelser_files):
        try:
            xls = pd.ExcelFile(fpath)
        except Exception as e:
            print(f"Could not read {fpath}: {e}")
            continue

        print(f"\nScanning workbook: {fpath}")

        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet)
            except Exception as e:
                print(f"  Could not read sheet '{sheet}': {e}")
                continue

            if df.empty:
                continue

            guid_col = _first_col_containing(df, "guid")
            time_col = _first_col_containing(df, "tidspunkt")
            no_col = _first_col_containing(df, "no")
            skred_col = _first_col_containing(df, "skredid")

            maf_col = _first_col_containing(df, "middelflom")
            if maf_col is None:
                maf_col = _first_col_containing(df, "mean annual flood")

            if guid_col is None or time_col is None:
                continue

            tmp_cols = [guid_col, time_col]
            col_names = ["GUID", "Tidspunkt"]

            if maf_col is not None:
                tmp_cols.append(maf_col)
                col_names.append("MeanAnnualFlood")

            if no_col is not None:
                tmp_cols.append(no_col)
                col_names.append("No")

            if skred_col is not None:
                tmp_cols.append(skred_col)
                col_names.append("SkredID")

            tmp = df[tmp_cols].copy()
            tmp.columns = col_names

            tmp["Tidspunkt"] = pd.to_datetime(tmp["Tidspunkt"], dayfirst=True, errors="coerce")

            if "MeanAnnualFlood" in tmp.columns:
                tmp["MeanAnnualFlood"] = pd.to_numeric(tmp["MeanAnnualFlood"], errors="coerce")
            else:
                tmp["MeanAnnualFlood"] = pd.NA

            tmp = tmp.dropna(subset=["GUID", "Tidspunkt"])

            for _, r in tmp.iterrows():
                rows.append({
                    "GUID": str(r["GUID"]).strip(),
                    "No": r.get("No", None),
                    "SkredID": r.get("SkredID", None),
                    "MeanAnnualFlood": float(r["MeanAnnualFlood"]) if pd.notna(r["MeanAnnualFlood"]) else None,
                    "event_time": r["Tidspunkt"],
                })

    meta = pd.DataFrame(rows)
    print(f"\nTotal landslide rows (including duplicates): {len(meta)}")

    if meta.empty:
        return meta

    meta["info_score"] = meta[["No", "SkredID", "MeanAnnualFlood"]].notna().sum(axis=1)
    meta = meta.sort_values(by=["GUID", "SkredID", "info_score"], ascending=[True, True, False])

    dupes = meta[meta.duplicated(subset=["GUID", "SkredID"], keep=False)]

    if not dupes.empty:
        print("\nDUPLICATES FOUND (same GUID + SkredID):")
        print(dupes.sort_values(["GUID", "SkredID"]))
        print(f"Total duplicate rows: {len(dupes)}")
        print(f"Unique duplicate groups: {dupes[['GUID', 'SkredID']].drop_duplicates().shape[0]}")
    else:
        print("\nNo duplicates found for GUID + SkredID")

    meta = meta.drop_duplicates(subset=["GUID", "SkredID"], keep="first").drop(columns=["info_score"])

    print(f"Total unique landslide events (after dedupe by GUID+SkredID): {len(meta)}")
    return meta