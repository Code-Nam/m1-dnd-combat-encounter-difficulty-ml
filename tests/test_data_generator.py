"""Tests unitaires pour src/data_generator.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    XP_BY_CR,
    XP_THRESHOLDS_BY_LEVEL,
    get_xp_multiplier,
    label_encounter,
    party_thresholds,
    sample_encounter,
)

# ---------------------------------------------------------------------------
# get_xp_multiplier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_monsters,expected", [
    (1,  1.0),
    (2,  1.5),
    (3,  2.0),
    (6,  2.0),
    (7,  2.5),
    (10, 2.5),
    (11, 3.0),
    (14, 3.0),
    (15, 4.0),
    (50, 4.0),
])
def test_get_xp_multiplier(n_monsters: int, expected: float) -> None:
    assert get_xp_multiplier(n_monsters) == expected


# ---------------------------------------------------------------------------
# party_thresholds
# ---------------------------------------------------------------------------


def test_party_thresholds_single_level() -> None:
    easy, medium, hard, deadly = party_thresholds([5])
    assert (easy, medium, hard, deadly) == XP_THRESHOLDS_BY_LEVEL[5]


def test_party_thresholds_cumulates_correctly() -> None:
    easy1, medium1, hard1, deadly1 = party_thresholds([3])
    easy2, medium2, hard2, deadly2 = party_thresholds([3, 3])
    assert easy2   == easy1   * 2
    assert medium2 == medium1 * 2
    assert hard2   == hard1   * 2
    assert deadly2 == deadly1 * 2


def test_party_thresholds_mixed_levels() -> None:
    easy, _, _, deadly = party_thresholds([1, 20])
    e1, _, _, d1 = XP_THRESHOLDS_BY_LEVEL[1]
    e20, _, _, d20 = XP_THRESHOLDS_BY_LEVEL[20]
    assert easy   == e1  + e20
    assert deadly == d1  + d20


# ---------------------------------------------------------------------------
# label_encounter
# ---------------------------------------------------------------------------


def test_label_encounter_easy() -> None:
    # 1 gobelin (CR 1/4) contre 4 personnages niveau 10 = très facile
    assert label_encounter([10, 10, 10, 10], [0.25]) == "Easy"


def test_label_encounter_deadly() -> None:
    # 6 zombies (CR 1/4) contre 2 personnages niveau 1
    assert label_encounter([1, 1], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) == "Deadly"


def test_label_encounter_returns_valid_class() -> None:
    valid = {"Easy", "Medium", "Hard", "Deadly"}
    result = label_encounter([5, 5, 5], [1.0, 1.0])
    assert result in valid


@pytest.mark.parametrize("levels,crs,expected", [
    # 4 × niveau 5 vs 1 gobelin CR 1/4 → Easy
    ([5, 5, 5, 5], [0.25],                   "Easy"),
    # 2 × niveau 1 vs 6 zombies CR 1/4 → Deadly
    ([1, 1],       [0.25] * 6,               "Deadly"),
])
def test_label_encounter_known_cases(
    levels: list[int], crs: list[float], expected: str
) -> None:
    assert label_encounter(levels, crs) == expected


def test_label_encounter_consistency_with_xp() -> None:
    """Le label doit être cohérent avec le calcul XP manuel."""
    levels = [4, 4, 4]
    crs    = [0.5, 0.5]

    easy, medium, hard, deadly = party_thresholds(levels)
    raw_xp      = sum(XP_BY_CR[cr] for cr in crs)
    adjusted_xp = raw_xp * get_xp_multiplier(len(crs))
    label = label_encounter(levels, crs)

    if adjusted_xp >= deadly:
        assert label == "Deadly"
    elif adjusted_xp >= hard:
        assert label == "Hard"
    elif adjusted_xp >= medium:
        assert label == "Medium"
    else:
        assert label == "Easy"


# ---------------------------------------------------------------------------
# sample_encounter
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_chars() -> pd.DataFrame:
    """DataFrame personnages minimal (10 lignes niveau 5)."""
    return pd.DataFrame({
        "level": [5] * 10,
        "HP":    [35] * 10,
        "AC":    [14] * 10,
        "Str":   [12] * 10,
        "Dex":   [13] * 10,
        "Con":   [12] * 10,
    })


@pytest.fixture()
def minimal_monsters() -> pd.DataFrame:
    """DataFrame monstres minimal avec CR valides."""
    return pd.DataFrame({
        "cr":          [0.25, 0.5, 1.0, 2.0, 3.0] * 4,
        "hit_points":  [7,    15,  30,  45,  60]  * 4,
        "armor_class": [13,   13,  13,  13,  13]  * 4,
    })


def test_sample_encounter_schema(
    minimal_chars: pd.DataFrame, minimal_monsters: pd.DataFrame
) -> None:
    rng = np.random.default_rng(0)
    row = sample_encounter(minimal_chars, minimal_monsters, rng)

    expected_keys = {
        "party_size", "party_avg_level", "party_avg_hp", "party_avg_ac",
        "party_avg_str", "party_avg_dex", "party_avg_con",
        "monster_count", "monster_avg_cr", "monster_avg_hp", "monster_avg_ac",
        "xp_raw", "xp_adjusted", "xp_ratio", "difficulty",
        "threshold_easy", "threshold_medium", "threshold_hard", "threshold_deadly",
    }
    assert expected_keys == set(row.keys())


def test_sample_encounter_valid_difficulty(
    minimal_chars: pd.DataFrame, minimal_monsters: pd.DataFrame
) -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        row = sample_encounter(minimal_chars, minimal_monsters, rng)
        assert row["difficulty"] in {"Easy", "Medium", "Hard", "Deadly"}


def test_sample_encounter_party_size_range(
    minimal_chars: pd.DataFrame, minimal_monsters: pd.DataFrame
) -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        row = sample_encounter(minimal_chars, minimal_monsters, rng)
        assert 2 <= row["party_size"] <= 6


def test_sample_encounter_monster_count_range(
    minimal_chars: pd.DataFrame, minimal_monsters: pd.DataFrame
) -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        row = sample_encounter(minimal_chars, minimal_monsters, rng)
        assert 1 <= row["monster_count"] <= 6


def test_sample_encounter_xp_ratio_positive(
    minimal_chars: pd.DataFrame, minimal_monsters: pd.DataFrame
) -> None:
    rng = np.random.default_rng(4)
    row = sample_encounter(minimal_chars, minimal_monsters, rng)
    assert row["xp_ratio"] > 0
