"""Récupération des monstres depuis l'API Open5e (v1).

Ce module télécharge l'intégralité du bestiaire exposé par l'endpoint
``https://api.open5e.com/v1/monsters/`` en suivant la pagination, puis
sauvegarde les données brutes dans ``data/raw/`` sous deux formes :

- ``monsters.json`` : la liste complète des monstres (tous les champs).
- ``monsters.csv``  : un tableau plat avec les colonnes scalaires utiles
  pour l'entraînement (CR, AC, HP, caractéristiques...). Les champs
  imbriqués (actions, speed, skills...) sont sérialisés en JSON.

Usage :
    python src/fetch_monsters.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_URL = "https://api.open5e.com/v1/monsters/"
PAGE_SIZE = 500
TIMEOUT = 60

# Colonnes scalaires conservées telles quelles dans le CSV. Les autres
# champs (dicts/listes) sont sérialisés en chaîne JSON pour rester lisibles.
SCALAR_COLUMNS = [
    "slug",
    "name",
    "size",
    "type",
    "subtype",
    "alignment",
    "armor_class",
    "armor_desc",
    "hit_points",
    "hit_dice",
    "strength",
    "dexterity",
    "constitution",
    "intelligence",
    "wisdom",
    "charisma",
    "challenge_rating",
    "cr",
    "document__title",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_all_monsters(url: str = API_URL) -> list[dict[str, Any]]:
    """Télécharge tous les monstres en suivant la pagination de l'API.

    Args:
        url: URL de départ de l'endpoint des monstres.

    Returns:
        La liste de tous les monstres (un dictionnaire par monstre).
    """
    monsters: list[dict[str, Any]] = []
    params: dict[str, Any] | None = {"limit": PAGE_SIZE}

    while url:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        monsters.extend(payload["results"])
        print(f"  {len(monsters)} / {payload['count']} monstres récupérés...")

        # Les pages suivantes embarquent déjà les paramètres dans l'URL `next`.
        url = payload["next"]
        params = None

    return monsters


def to_dataframe(monsters: list[dict[str, Any]]) -> pd.DataFrame:
    """Convertit la liste de monstres en DataFrame plat.

    Les colonnes scalaires sont conservées ; les champs imbriqués restants
    sont sérialisés en chaînes JSON pour tenir dans un CSV.

    Args:
        monsters: Liste de monstres telle que renvoyée par l'API.

    Returns:
        Un DataFrame pandas prêt à être sauvegardé en CSV.
    """
    rows: list[dict[str, Any]] = []
    for monster in monsters:
        row: dict[str, Any] = {col: monster.get(col) for col in SCALAR_COLUMNS}
        # On garde les champs complexes utiles, sérialisés en JSON.
        for field in ("speed", "skills", "actions", "special_abilities"):
            value = monster.get(field)
            row[field] = json.dumps(value, ensure_ascii=False) if value else None
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Récupère les monstres et écrit les fichiers bruts dans data/raw/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Récupération des monstres depuis {API_URL}")
    monsters = fetch_all_monsters()
    print(f"Total : {len(monsters)} monstres.")

    json_path = OUTPUT_DIR / "monsters.json"
    json_path.write_text(
        json.dumps(monsters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON brut écrit : {json_path}")

    df = to_dataframe(monsters)
    csv_path = OUTPUT_DIR / "monsters.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV écrit : {csv_path}  ({df.shape[0]} lignes x {df.shape[1]} colonnes)")


if __name__ == "__main__":
    main()