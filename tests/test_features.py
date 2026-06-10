"""Tests unitaires pour src/features.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import DROP_COLS, DIFFICULTY_ORDER, build_features, split_and_save


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def test_build_features_adds_derived_columns(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    for col in ("cr_level_delta", "hp_ratio", "ac_gap",
                "log_xp_raw", "log_xp_adjusted",
                "log_monster_avg_hp", "log_party_avg_hp"):
        assert col in df.columns, f"Colonne manquante : {col}"


def test_build_features_drops_raw_columns(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    for col in DROP_COLS:
        assert col not in df.columns, f"Colonne non supprimée : {col}"


def test_build_features_cr_level_delta(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    expected = tiny_encounters["monster_avg_cr"] - tiny_encounters["party_avg_level"]
    pd.testing.assert_series_equal(
        df["cr_level_delta"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_build_features_hp_ratio(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    expected = tiny_encounters["monster_avg_hp"] / tiny_encounters["party_avg_hp"]
    pd.testing.assert_series_equal(
        df["hp_ratio"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_build_features_log_transforms_non_negative(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    for col in ("log_xp_raw", "log_xp_adjusted", "log_monster_avg_hp", "log_party_avg_hp"):
        assert (df[col] >= 0).all(), f"{col} contient des valeurs négatives"


def test_build_features_difficulty_encoded_values(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    assert set(df["difficulty_encoded"].unique()) == {0, 1, 2, 3}


def test_build_features_encoding_order(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    for difficulty, code in DIFFICULTY_ORDER.items():
        mask = df["difficulty"] == difficulty
        assert (df.loc[mask, "difficulty_encoded"] == code).all()


def test_build_features_no_missing_values(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    assert df.isnull().sum().sum() == 0


def test_build_features_preserves_row_count(tiny_encounters: pd.DataFrame) -> None:
    df = build_features(tiny_encounters)
    assert len(df) == len(tiny_encounters)


# ---------------------------------------------------------------------------
# split_and_save
# ---------------------------------------------------------------------------


def test_split_and_save_creates_files(tiny_features: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        split_and_save(tiny_features, output_dir=out, test_size=0.25, random_state=42)
        for fname in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
                      "encounters_features.csv"):
            assert (out / fname).exists(), f"Fichier manquant : {fname}"


def test_split_and_save_shapes(tiny_features: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        X_train, X_test, y_train, y_test = split_and_save(
            tiny_features, output_dir=out, test_size=0.25, random_state=42
        )
        total = len(tiny_features)
        assert len(X_train) + len(X_test) == total
        assert len(y_train) == len(X_train)
        assert len(y_test)  == len(X_test)


def test_split_and_save_no_xp_leakage_check(tiny_features: pd.DataFrame) -> None:
    """Les features XP doivent rester dans les fichiers (c'est models.py qui les retire)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        split_and_save(tiny_features, output_dir=out)
        X_train = pd.read_csv(out / "X_train.csv")
        assert "xp_ratio" in X_train.columns


def test_split_stratified_distribution(tiny_features: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _, _, y_train, y_test = split_and_save(
            tiny_features, output_dir=out, test_size=0.25, random_state=42
        )
        # Chaque classe doit apparaître dans les deux splits
        assert set(y_train.unique()) == {0, 1, 2, 3}
        assert set(y_test.unique())  == {0, 1, 2, 3}
