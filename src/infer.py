"""Inférence de la difficulté d'une rencontre D&D 5e.

Ce script charge le modèle XGBoost sauvegardé et prédit la difficulté
(Easy / Medium / Hard / Deadly) à partir des stats brutes d'une rencontre.

Les features dérivées (cr_level_delta, hp_ratio, ac_gap, log transforms) sont
calculées automatiquement — seules les stats brutes du groupe et des monstres
sont nécessaires en entrée.

Usage :
    # Exemple prédéfini
    python src/infer.py --example

    # Rencontre personnalisée
    python src/infer.py \\
        --party-size 4 --party-avg-level 5 --party-avg-hp 35 \\
        --party-avg-ac 14 --party-avg-str 12 --party-avg-dex 13 --party-avg-con 12 \\
        --monster-count 3 --monster-avg-cr 1.0 --monster-avg-hp 30 --monster-avg-ac 13

    # Sortie JSON (pour intégration API)
    python src/infer.py --example --json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import sys
from pathlib import Path as _Path

# Permet d'exécuter `python src/infer.py` depuis la racine du projet
# sans avoir à définir PYTHONPATH manuellement.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models import LABEL_NAMES, load_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "xgboost_difficulty.joblib"

# Features dans l'ordre exact attendu par le modèle
FEATURE_ORDER: list[str] = [
    "party_size", "party_avg_level", "party_avg_ac",
    "party_avg_str", "party_avg_dex", "party_avg_con",
    "monster_count", "monster_avg_cr", "monster_avg_ac",
    "cr_level_delta", "hp_ratio", "ac_gap",
    "log_monster_avg_hp", "log_party_avg_hp",
]


def build_feature_vector(
    party_size: int,
    party_avg_level: float,
    party_avg_hp: float,
    party_avg_ac: float,
    party_avg_str: float,
    party_avg_dex: float,
    party_avg_con: float,
    monster_count: int,
    monster_avg_cr: float,
    monster_avg_hp: float,
    monster_avg_ac: float,
) -> pd.DataFrame:
    """Construit le vecteur de features à partir des stats brutes.

    Args:
        party_size:      Nombre de personnages dans le groupe (2-6).
        party_avg_level: Niveau moyen du groupe (1-20).
        party_avg_hp:    HP moyen des personnages.
        party_avg_ac:    AC moyen des personnages.
        party_avg_str:   Force moyenne des personnages.
        party_avg_dex:   Dextérité moyenne des personnages.
        party_avg_con:   Constitution moyenne des personnages.
        monster_count:   Nombre de monstres dans la rencontre (1-6+).
        monster_avg_cr:  CR moyen des monstres.
        monster_avg_hp:  HP moyen des monstres.
        monster_avg_ac:  AC moyen des monstres.

    Returns:
        DataFrame d'une ligne avec les 14 features attendues par le modèle.
    """
    row: dict[str, float] = {
        "party_size":        party_size,
        "party_avg_level":   party_avg_level,
        "party_avg_ac":      party_avg_ac,
        "party_avg_str":     party_avg_str,
        "party_avg_dex":     party_avg_dex,
        "party_avg_con":     party_avg_con,
        "monster_count":     monster_count,
        "monster_avg_cr":    monster_avg_cr,
        "monster_avg_ac":    monster_avg_ac,
        # Features dérivées
        "cr_level_delta":    monster_avg_cr - party_avg_level,
        "hp_ratio":          monster_avg_hp / party_avg_hp,
        "ac_gap":            monster_avg_ac - party_avg_ac,
        "log_monster_avg_hp": math.log1p(monster_avg_hp),
        "log_party_avg_hp":   math.log1p(party_avg_hp),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def predict(feature_df: pd.DataFrame) -> dict[str, Any]:
    """Charge le modèle et retourne la prédiction avec les probabilités par classe.

    Args:
        feature_df: DataFrame d'une ligne produit par build_feature_vector().

    Returns:
        Dictionnaire avec 'difficulty', 'confidence', et 'probabilities'.
    """
    model  = load_model(MODEL_PATH)
    probas = model.predict_proba(feature_df)[0]
    pred   = int(model.predict(feature_df)[0])

    return {
        "difficulty":   LABEL_NAMES[pred],
        "confidence":   round(float(probas[pred]) * 100, 1),
        "probabilities": {
            label: round(float(p) * 100, 1)
            for label, p in zip(LABEL_NAMES, probas)
        },
    }


def print_result(result: dict[str, Any], raw: dict[str, Any]) -> None:
    """Affiche le résultat de façon lisible."""
    diff   = result["difficulty"]
    conf   = result["confidence"]
    probas = result["probabilities"]

    DIFF_COLORS = {"Easy": "✅", "Medium": "🟡", "Hard": "🟠", "Deadly": "🔴"}
    icon = DIFF_COLORS.get(diff, "")

    print()
    print("─── Rencontre ──────────────────────────────────────")
    print(f"  Groupe : {raw['party_size']} personnages  "
          f"| niveau moyen {raw['party_avg_level']:.1f}  "
          f"| HP moy. {raw['party_avg_hp']:.0f}  "
          f"| AC moy. {raw['party_avg_ac']:.0f}")
    print(f"  Monstres : {raw['monster_count']}  "
          f"| CR moyen {raw['monster_avg_cr']:.2f}  "
          f"| HP moy. {raw['monster_avg_hp']:.0f}  "
          f"| AC moy. {raw['monster_avg_ac']:.0f}")
    print()
    print(f"─── Prédiction ─────────────────────────────────────")
    print(f"  {icon}  {diff}  (confiance : {conf}%)")
    print()
    print("  Probabilités par classe :")
    bar_width = 30
    for label, pct in probas.items():
        filled = int(pct / 100 * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        marker = " ◄" if label == diff else ""
        print(f"    {label:<8} {bar} {pct:5.1f}%{marker}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prédit la difficulté d'une rencontre D&D 5e.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--example", action="store_true",
                        help="Utilise une rencontre exemple prédéfinie.")
    parser.add_argument("--json", action="store_true",
                        help="Affiche le résultat en JSON brut.")

    g = parser.add_argument_group("Groupe de personnages")
    g.add_argument("--party-size",      type=int,   default=4,    metavar="N")
    g.add_argument("--party-avg-level", type=float, default=5.0,  metavar="LVL")
    g.add_argument("--party-avg-hp",    type=float, default=35.0, metavar="HP")
    g.add_argument("--party-avg-ac",    type=float, default=14.0, metavar="AC")
    g.add_argument("--party-avg-str",   type=float, default=12.0, metavar="STR")
    g.add_argument("--party-avg-dex",   type=float, default=13.0, metavar="DEX")
    g.add_argument("--party-avg-con",   type=float, default=12.0, metavar="CON")

    m = parser.add_argument_group("Monstres")
    m.add_argument("--monster-count",   type=int,   default=3,    metavar="N")
    m.add_argument("--monster-avg-cr",  type=float, default=1.0,  metavar="CR")
    m.add_argument("--monster-avg-hp",  type=float, default=30.0, metavar="HP")
    m.add_argument("--monster-avg-ac",  type=float, default=13.0, metavar="AC")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Rencontre exemple : 4 personnages niveau 5 vs 5 orcs CR 1/2 → devrait être Hard/Deadly
    if args.example:
        raw = dict(
            party_size=4, party_avg_level=5.0, party_avg_hp=35.0,
            party_avg_ac=14.0, party_avg_str=12.0, party_avg_dex=13.0, party_avg_con=12.0,
            monster_count=5, monster_avg_cr=0.5, monster_avg_hp=15.0, monster_avg_ac=13.0,
        )
    else:
        raw = dict(
            party_size=args.party_size,
            party_avg_level=args.party_avg_level,
            party_avg_hp=args.party_avg_hp,
            party_avg_ac=args.party_avg_ac,
            party_avg_str=args.party_avg_str,
            party_avg_dex=args.party_avg_dex,
            party_avg_con=args.party_avg_con,
            monster_count=args.monster_count,
            monster_avg_cr=args.monster_avg_cr,
            monster_avg_hp=args.monster_avg_hp,
            monster_avg_ac=args.monster_avg_ac,
        )

    features = build_feature_vector(**raw)
    result   = predict(features)

    if args.json:
        print(json.dumps({**result, "input": raw}, indent=2))
    else:
        print_result(result, raw)


if __name__ == "__main__":
    main()
