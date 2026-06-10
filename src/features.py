"""Ingénierie des features pour le modèle de difficulté D&D 5e.

Ce module transforme le dataset brut d'encounters (data/raw/encounters.csv) en
features prêtes à l'entraînement et sauvegarde les splits train/test dans
data/processed/.

Features créées :
- cr_level_delta    : monster_avg_cr - party_avg_level (rapport de force relatif)
- hp_ratio          : monster_avg_hp / party_avg_hp (endurance relative)
- ac_gap            : monster_avg_ac - party_avg_ac (avantage défensif)
- log_xp_raw        : log1p(xp_raw)
- log_xp_adjusted   : log1p(xp_adjusted)
- log_monster_avg_hp: log1p(monster_avg_hp)
- log_party_avg_hp  : log1p(party_avg_hp)

Usage :
    python src/features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR       = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DIFFICULTY_ORDER: dict[str, int] = {"Easy": 0, "Medium": 1, "Hard": 2, "Deadly": 3}

# Colonnes supprimées : seuils XP (fuite de données via le label) et colonnes
# remplacées par leur version log ou dérivée.
DROP_COLS: list[str] = [
    "threshold_easy", "threshold_medium", "threshold_hard", "threshold_deadly",
    "xp_raw", "xp_adjusted",
    "monster_avg_hp", "party_avg_hp",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features dérivées et supprime les colonnes redondantes.

    Args:
        df: DataFrame brut issu de data/raw/encounters.csv.

    Returns:
        DataFrame avec les features finales + colonnes 'difficulty' et 'difficulty_encoded'.
    """
    df = df.copy()

    df["cr_level_delta"]     = df["monster_avg_cr"] - df["party_avg_level"]
    df["hp_ratio"]           = df["monster_avg_hp"] / df["party_avg_hp"]
    df["ac_gap"]             = df["monster_avg_ac"] - df["party_avg_ac"]
    df["log_xp_raw"]         = np.log1p(df["xp_raw"])
    df["log_xp_adjusted"]    = np.log1p(df["xp_adjusted"])
    df["log_monster_avg_hp"] = np.log1p(df["monster_avg_hp"])
    df["log_party_avg_hp"]   = np.log1p(df["party_avg_hp"])

    df["difficulty_encoded"] = df["difficulty"].map(DIFFICULTY_ORDER)

    return df.drop(columns=DROP_COLS)


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare features/cible, effectue le train/test split et sauvegarde les fichiers.

    Args:
        df:           DataFrame avec features + 'difficulty' + 'difficulty_encoded'.
        output_dir:   Dossier de destination pour les CSV.
        test_size:    Fraction du jeu de test (défaut 0.2).
        random_state: Graine pour la reproductibilité.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in ("difficulty", "difficulty_encoded")]
    X = df[feature_cols]
    y = df["difficulty_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir  / "X_test.csv",  index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir  / "y_test.csv",  index=False)
    df.to_csv(output_dir / "encounters_features.csv", index=False)

    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"Sauvegardé dans : {output_dir}")
    return X_train, X_test, y_train, y_test


def main() -> None:
    """Charge encounters.csv, construit les features et sauvegarde les splits."""
    df_raw = pd.read_csv(RAW_DIR / "encounters.csv")
    print(f"Données brutes : {df_raw.shape[0]} lignes x {df_raw.shape[1]} colonnes")

    df_feat = build_features(df_raw)
    print(f"Features finales : {df_feat.shape[1]} colonnes")
    print(f"  {df_feat.columns.tolist()}")

    split_and_save(df_feat)


if __name__ == "__main__":
    main()
