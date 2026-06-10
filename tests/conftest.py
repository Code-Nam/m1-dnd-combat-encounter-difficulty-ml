"""Fixtures partagées entre les tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import XP_BY_CR, label_encounter
from src.features import build_features


@pytest.fixture()
def tiny_encounters() -> pd.DataFrame:
    """20 rencontres synthétiques minimales (5 par classe) sans dépendance externe."""
    rows = []

    # Easy : 1 gobelin (CR 1/4) contre groupe niveau 5 x4
    for _ in range(5):
        rows.append({
            "party_size": 4, "party_avg_level": 5.0, "party_avg_hp": 30.0,
            "party_avg_ac": 14.0, "party_avg_str": 12.0, "party_avg_dex": 13.0,
            "party_avg_con": 12.0, "monster_count": 1, "monster_avg_cr": 0.25,
            "monster_avg_hp": 7.0, "monster_avg_ac": 13.0,
            "xp_raw": 50, "xp_adjusted": 50, "xp_ratio": 0.05,
            "threshold_easy": 1000, "threshold_medium": 2000,
            "threshold_hard": 3000, "threshold_deadly": 4000,
            "difficulty": "Easy",
        })

    # Medium : 2 orcs (CR 1/2) contre groupe niveau 3 x3
    for _ in range(5):
        rows.append({
            "party_size": 3, "party_avg_level": 3.0, "party_avg_hp": 20.0,
            "party_avg_ac": 13.0, "party_avg_str": 11.0, "party_avg_dex": 11.0,
            "party_avg_con": 11.0, "monster_count": 2, "monster_avg_cr": 0.5,
            "monster_avg_hp": 15.0, "monster_avg_ac": 13.0,
            "xp_raw": 300, "xp_adjusted": 450, "xp_ratio": 0.50,
            "threshold_easy": 225, "threshold_medium": 450,
            "threshold_hard": 675, "threshold_deadly": 900,
            "difficulty": "Medium",
        })

    # Hard : 4 loups (CR 1/4) contre groupe niveau 2 x3
    for _ in range(5):
        rows.append({
            "party_size": 3, "party_avg_level": 2.0, "party_avg_hp": 15.0,
            "party_avg_ac": 13.0, "party_avg_str": 11.0, "party_avg_dex": 11.0,
            "party_avg_con": 11.0, "monster_count": 4, "monster_avg_cr": 0.25,
            "monster_avg_hp": 11.0, "monster_avg_ac": 13.0,
            "xp_raw": 200, "xp_adjusted": 400, "xp_ratio": 0.80,
            "threshold_easy": 150, "threshold_medium": 300,
            "threshold_hard": 450, "threshold_deadly": 600,
            "difficulty": "Hard",
        })

    # Deadly : 6 zombies (CR 1/4) contre groupe niveau 1 x2
    for _ in range(5):
        rows.append({
            "party_size": 2, "party_avg_level": 1.0, "party_avg_hp": 9.0,
            "party_avg_ac": 12.0, "party_avg_str": 10.0, "party_avg_dex": 10.0,
            "party_avg_con": 10.0, "monster_count": 6, "monster_avg_cr": 0.25,
            "monster_avg_hp": 22.0, "monster_avg_ac": 8.0,
            "xp_raw": 300, "xp_adjusted": 600, "xp_ratio": 3.0,
            "threshold_easy": 50, "threshold_medium": 100,
            "threshold_hard": 150, "threshold_deadly": 200,
            "difficulty": "Deadly",
        })

    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_features(tiny_encounters: pd.DataFrame) -> pd.DataFrame:
    """Dataset avec features construites à partir de tiny_encounters."""
    return build_features(tiny_encounters)


@pytest.fixture()
def tiny_splits(tiny_features: pd.DataFrame):
    """Train/test split (75/25) sur tiny_features — retourne (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split

    feature_cols = [c for c in tiny_features.columns if c not in ("difficulty", "difficulty_encoded")]
    X = tiny_features[feature_cols]
    y = tiny_features["difficulty_encoded"]

    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
