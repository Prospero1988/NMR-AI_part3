#!/usr/bin/env python
"""
Combine NMR-based CSV inputs (1H & 13C) in three ways:

    1. 1H13C  – concat(1H, 13C)
    2. 13C1H  – concat(13C, 1H)
    3. 1Hx13C – element-wise sum

For each pair of files that share the same stem up to the first “_”
and differ only by “1H/13C”, three new CSV files are written to:
    <input_dir>/<stem>/1H13C/
    <input_dir>/<stem>/13C1H/
    <input_dir>/<stem>/1Hx13C/

Usage
-----
python combine_nmr_inputs.py <directory_with_csv>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s | %(message)s",
)

META_COLS = ["MOLECULE_NAME", "LABEL"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def split_meta_feat(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (meta, features) DataFrames."""
    meta = df[META_COLS].copy()
    features = df.drop(columns=META_COLS).copy()
    return meta, features


def renumber_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename feature columns to FEATURE_001, FEATURE_002, … (continuous)."""
    meta, feats = split_meta_feat(df)
    renamed = {
        f"FEATURE_{i}": feats[col]
        for i, col in enumerate(feats.columns, start=1)
    }
    return pd.concat([meta, pd.DataFrame(renamed)], axis=1)


def concat_features(df_h: pd.DataFrame, df_c: pd.DataFrame, order: str) -> pd.DataFrame:
    """
    Concatenate feature vectors in a given order, keeping meta once.

    Parameters
    ----------
    order : 'H_first' or 'C_first'
    """
    meta, feats_h = split_meta_feat(df_h)
    _, feats_c = split_meta_feat(df_c)

    # Give temporary unique names to avoid duplication problems.
    feats_h.columns = [f"H_{col}" for col in feats_h.columns]
    feats_c.columns = [f"C_{col}" for col in feats_c.columns]

    feats_concat = (
        pd.concat([feats_h, feats_c], axis=1)
        if order == "H_first"
        else pd.concat([feats_c, feats_h], axis=1)
    )
    out = pd.concat([meta, feats_concat], axis=1)
    return renumber_features(out)


def add_features(df_h: pd.DataFrame, df_c: pd.DataFrame) -> pd.DataFrame:
    """Element-wise sum of 1H and 13C feature vectors (meta kept once)."""
    meta, feats_h = split_meta_feat(df_h)
    _, feats_c = split_meta_feat(df_c)

    if len(feats_h.columns) != len(feats_c.columns):
        raise ValueError("1H and 13C have different feature dimensions.")

    summed = feats_h.values + feats_c.values
    summed_df = pd.DataFrame(
        summed,
        columns=[f"FEATURE_{i}" for i in range(1, summed.shape[1] + 1)],
    )
    return pd.concat([meta, summed_df], axis=1)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #


def process_pair(
    h_path: Path,
    c_path: Path,
    out_root: Path,
) -> None:
    """Load a pair of CSVs and write three combined variants."""
    df_h = pd.read_csv(h_path)
    df_c = pd.read_csv(c_path)

    stem = h_path.stem.split("_")[0]

    # Prepare output sub-dirs
    out_1h13c = (out_root / stem / "1H13C")
    out_13c1h = (out_root / stem / "13C1H")
    out_sum = (out_root / stem / "1Hx13C")
    for p in (out_1h13c, out_13c1h, out_sum):
        p.mkdir(parents=True, exist_ok=True)

    # ---- concat 1H + 13C
    df_concat = concat_features(df_h, df_c, order="H_first")
    df_concat.to_csv(out_1h13c / f"{stem}_1H13C.csv", index=False)

    # ---- concat 13C + 1H
    df_concat_rev = concat_features(df_h, df_c, order="C_first")
    df_concat_rev.to_csv(out_13c1h / f"{stem}_13C1H.csv", index=False)

    # ---- element-wise sum
    df_sum = add_features(df_h, df_c)
    df_sum.to_csv(out_sum / f"{stem}_1Hx13C.csv", index=False)

    logging.info("  ↳ saved: %s", stem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine paired 1H / 13C CSV inputs."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Folder containing *_1H_*.csv and *_13C_*.csv files.",
    )
    args = parser.parse_args()
    input_dir: Path = args.directory.resolve()

    if not input_dir.is_dir():
        parser.error(f"{input_dir} is not a directory")

    logging.info("Scanning directory: %s", input_dir)

    files = list(input_dir.glob("*.csv"))
    groups: dict[str, dict[str, Path]] = {}
    for csv in files:
        head, suffix = csv.stem.split("_", maxsplit=1)
        if suffix.startswith("1H"):
            groups.setdefault(head, {})["1H"] = csv
        elif suffix.startswith("13C"):
            groups.setdefault(head, {})["13C"] = csv

    pairs = {k: v for k, v in groups.items() if {"1H", "13C"} <= v.keys()}
    logging.info("Found %d file pair(s).", len(pairs))

    for stem, pair in pairs.items():
        logging.info("  ↳ %s + %s", pair['1H'].name, pair['13C'].name)
        try:
            process_pair(pair["1H"], pair["13C"], input_dir)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to process pair '%s*': %s", stem, exc)


if __name__ == "__main__":
    main()
