"""Génération de rencontres D&D 5e synthétiques.

Ce module implémente la méthode de difficulté officielle du Dungeon Master's Guide
(D&D 5e, p. 82) pour labeller chaque rencontre générée (Easy / Medium / Hard / Deadly).

La génération est stratifiée : on remplit des buckets de TARGET_PER_CLASS exemples par
classe jusqu'à atteindre l'équilibre, évitant la sur-représentation naturelle de "Deadly".

Usage :
    python src/data_generator.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tables DMG
# ---------------------------------------------------------------------------

# Seuils XP par niveau de personnage (Easy, Medium, Hard, Deadly).
# Source : DMG p. 82 — D&D 5e (2014).
XP_THRESHOLDS_BY_LEVEL: dict[int, tuple[int, int, int, int]] = {
    1:  (25,   50,   75,   100),
    2:  (50,   100,  150,  200),
    3:  (75,   150,  225,  400),
    4:  (125,  250,  375,  500),
    5:  (250,  500,  750,  1100),
    6:  (300,  600,  900,  1400),
    7:  (350,  750,  1100, 1700),
    8:  (450,  900,  1400, 2100),
    9:  (550,  1100, 1600, 2400),
    10: (600,  1200, 1900, 2800),
    11: (800,  1600, 2400, 3600),
    12: (1000, 2000, 3000, 4500),
    13: (1100, 2200, 3400, 5100),
    14: (1250, 2500, 3800, 5700),
    15: (1400, 2800, 4300, 6400),
    16: (1600, 3200, 4800, 7200),
    17: (2000, 3900, 5900, 8800),
    18: (2100, 4200, 6300, 9500),
    19: (2400, 4900, 7300, 10900),
    20: (2800, 5700, 8500, 12700),
}

# XP accordé par CR.
# Source : DMG p. 274 — D&D 5e (2014).
XP_BY_CR: dict[float, int] = {
    0: 10, 0.125: 25, 0.25: 50, 0.5: 100,
    1: 200, 2: 450, 3: 700, 4: 1100, 5: 1800,
    6: 2300, 7: 2900, 8: 3900, 9: 5000, 10: 5900,
    11: 7200, 12: 8400, 13: 10000, 14: 11500, 15: 13000,
    16: 15000, 17: 18000, 18: 20000, 19: 22000, 20: 25000,
    21: 33000, 22: 41000, 23: 50000, 24: 62000, 25: 75000,
    26: 90000, 27: 105000, 28: 120000, 29: 135000, 30: 155000,
}

# Multiplicateurs XP selon le nombre de monstres. Source : DMG p. 82.
XP_MULTIPLIERS: list[tuple[int, int | None, float]] = [
    (1,  1,    1.0),
    (2,  2,    1.5),
    (3,  6,    2.0),
    (7,  10,   2.5),
    (11, 14,   3.0),
    (15, None, 4.0),
]

VALID_CRS: frozenset[float] = frozenset(XP_BY_CR.keys())
CLASSES: list[str] = ["Easy", "Medium", "Hard", "Deadly"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Fonctions utilitaires DMG
# ---------------------------------------------------------------------------


def get_xp_multiplier(n_monsters: int) -> float:
    """Retourne le multiplicateur XP pour n monstres (DMG p. 82)."""
    for low, high, mult in XP_MULTIPLIERS:
        if high is None or low <= n_monsters <= high:
            return mult
    return 4.0


def party_thresholds(levels: list[int]) -> tuple[int, int, int, int]:
    """Retourne les seuils XP cumulés du groupe (easy, medium, hard, deadly).

    Args:
        levels: Niveaux des personnages du groupe.

    Returns:
        Tuple (easy, medium, hard, deadly) en XP total.
    """
    easy = medium = hard = deadly = 0
    for lvl in levels:
        e, m, h, d = XP_THRESHOLDS_BY_LEVEL[lvl]
        easy += e; medium += m; hard += h; deadly += d
    return easy, medium, hard, deadly


def label_encounter(levels: list[int], crs: list[float]) -> str:
    """Attribue un label de difficulté à une rencontre D&D 5e.

    Args:
        levels: Niveaux des personnages joueurs.
        crs:    CR de chaque monstre dans la rencontre.

    Returns:
        "Easy", "Medium", "Hard" ou "Deadly".
    """
    easy, medium, hard, deadly = party_thresholds(levels)
    raw_xp = sum(XP_BY_CR[cr] for cr in crs)
    adjusted_xp = raw_xp * get_xp_multiplier(len(crs))

    if adjusted_xp >= deadly:
        return "Deadly"
    if adjusted_xp >= hard:
        return "Hard"
    if adjusted_xp >= medium:
        return "Medium"
    return "Easy"


# ---------------------------------------------------------------------------
# Génération d'une rencontre
# ---------------------------------------------------------------------------


def sample_encounter(
    chars_df: pd.DataFrame,
    monsters_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Génère une rencontre aléatoire et retourne ses features + label.

    Args:
        chars_df:    DataFrame personnages nettoyé (colonnes : level, HP, AC, Str, Dex, Con).
        monsters_df: DataFrame monstres nettoyé (colonnes : cr, hit_points, armor_class).
        rng:         Générateur numpy pour la reproductibilité.

    Returns:
        Dictionnaire de features prêt à être ajouté au dataset.
    """
    target_level = int(rng.integers(1, 21))
    pool = chars_df[chars_df["level"].between(max(1, target_level - 2), min(20, target_level + 2))]
    if len(pool) < 3:
        pool = chars_df

    party_size = int(rng.integers(2, 7))  # 2 à 6 personnages
    party = pool.sample(n=party_size, replace=True)
    party_levels = party["level"].tolist()
    party_avg_level = party["level"].mean()

    easy, medium, hard, deadly = party_thresholds(party_levels)

    cr_min = max(0, party_avg_level / 4 - 1)
    cr_max = party_avg_level + 3
    valid_monsters = monsters_df[monsters_df["cr"].between(cr_min, cr_max)]
    if len(valid_monsters) < 5:
        valid_monsters = monsters_df

    n_monsters = int(rng.integers(1, 7))  # 1 à 6 monstres
    encounter_monsters = valid_monsters.sample(n=n_monsters, replace=True)
    monster_crs = encounter_monsters["cr"].tolist()

    raw_xp = sum(XP_BY_CR[cr] for cr in monster_crs)
    adjusted_xp = raw_xp * get_xp_multiplier(n_monsters)

    return {
        "party_size":       party_size,
        "party_avg_level":  round(party_avg_level, 2),
        "party_avg_hp":     round(party["HP"].mean(), 1),
        "party_avg_ac":     round(party["AC"].mean(), 1),
        "party_avg_str":    round(party["Str"].mean(), 1),
        "party_avg_dex":    round(party["Dex"].mean(), 1),
        "party_avg_con":    round(party["Con"].mean(), 1),
        "monster_count":    n_monsters,
        "monster_avg_cr":   round(sum(monster_crs) / n_monsters, 3),
        "monster_avg_hp":   round(encounter_monsters["hit_points"].mean(), 1),
        "monster_avg_ac":   round(encounter_monsters["armor_class"].mean(), 1),
        "xp_raw":           int(raw_xp),
        "xp_adjusted":      int(adjusted_xp),
        "xp_ratio":         round(adjusted_xp / deadly, 3),
        "threshold_easy":   easy,
        "threshold_medium": medium,
        "threshold_hard":   hard,
        "threshold_deadly": deadly,
        "difficulty":       label_encounter(party_levels, monster_crs),
    }


# ---------------------------------------------------------------------------
# Génération du dataset complet (stratifié)
# ---------------------------------------------------------------------------


def generate_balanced_dataset(
    chars_df: pd.DataFrame,
    monsters_df: pd.DataFrame,
    target_per_class: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Génère un dataset équilibré avec target_per_class exemples par classe.

    La génération naive sur-représente "Deadly". Cette fonction remplit des
    buckets par classe et rejette les exemples excédentaires.

    Args:
        chars_df:         DataFrame personnages nettoyé.
        monsters_df:      DataFrame monstres nettoyé.
        target_per_class: Nombre d'exemples cible par classe de difficulté.
        random_state:     Graine pour la reproductibilité.

    Returns:
        DataFrame équilibré avec target_per_class × 4 lignes.
    """
    rng = np.random.default_rng(random_state)
    buckets: dict[str, list[dict[str, Any]]] = {c: [] for c in CLASSES}
    attempts = 0

    while any(len(buckets[c]) < target_per_class for c in CLASSES):
        row = sample_encounter(chars_df, monsters_df, rng)
        label = row["difficulty"]
        if len(buckets[label]) < target_per_class:
            buckets[label].append(row)
        attempts += 1

    df = pd.DataFrame([row for c in CLASSES for row in buckets[c]])
    print(f"Rencontres générées : {len(df)} ({attempts} tentatives, "
          f"taux d'acceptation {len(df) / attempts * 100:.1f}%)")
    return df


def _load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge et nettoie les données brutes depuis data/raw/."""
    monsters = pd.read_csv(RAW_DIR / "monsters.csv")
    characters = pd.read_csv(RAW_DIR / "characters.csv")

    monsters_clean = monsters[monsters["cr"].isin(VALID_CRS)].copy()
    monsters_clean["_is_core"] = monsters_clean["document__title"] == "5e Core Rules"
    monsters_clean = (
        monsters_clean
        .sort_values("_is_core", ascending=False)
        .drop_duplicates(subset="name", keep="first")
        .drop(columns="_is_core")
        .reset_index(drop=True)
    )

    char_cols = ["level", "HP", "AC", "Str", "Dex", "Con"]
    chars_clean = (
        characters[char_cols]
        .dropna()
        .query("1 <= level <= 20")
        .query("0 < HP <= 1000")
        .query("5 <= AC <= 40")
        .reset_index(drop=True)
    )

    return chars_clean, monsters_clean


def main() -> None:
    """Génère encounters.csv dans data/raw/."""
    chars_df, monsters_df = _load_raw_data()
    print(f"Personnages : {len(chars_df)}  |  Monstres : {len(monsters_df)}")

    df = generate_balanced_dataset(chars_df, monsters_df, target_per_class=1000)

    out_path = RAW_DIR / "encounters.csv"
    df.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path}  ({df.shape[0]} lignes x {df.shape[1]} colonnes)")
    print("\nDistribution :")
    print(df["difficulty"].value_counts().reindex(CLASSES).to_string())


if __name__ == "__main__":
    main()
