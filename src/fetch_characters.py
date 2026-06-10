"""Récupération des personnages joueurs depuis oganm/dnddata (GitHub).

Le dépôt ``oganm/dnddata`` agrège des fiches de personnages D&D 5e
réellement créées par des joueurs. Les fichiers tabulaires sont exposés en
TSV dans ``data-raw/`` :

- ``dnd_chars_unique.tsv`` : personnages dédupliqués (recommandé pour le ML).
- ``dnd_chars_all.tsv``    : toutes les fiches, y compris les doublons.

Ce module télécharge ces TSV et les sauvegarde en CSV dans ``data/raw/``.

Note : les fichiers contiennent des colonnes d'identification utilisées
pour la déduplication côté source (``ip``, ``finger``, ``hash``). On peut
les ignorer plus tard ; ici on conserve les données brutes telles quelles.

Usage :
    python src/fetch_characters.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/oganm/dnddata/master/data-raw"
TIMEOUT = 120

# Fichiers à récupérer : nom distant -> nom local (sans extension).
DATASETS = {
    "dnd_chars_unique.tsv": "characters_unique",
    "dnd_chars_all.tsv": "characters_all",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_tsv(filename: str) -> pd.DataFrame:
    """Télécharge un fichier TSV du dépôt dnddata et le charge en DataFrame.

    Args:
        filename: Nom du fichier TSV dans ``data-raw/`` du dépôt.

    Returns:
        Un DataFrame pandas contenant les personnages.
    """
    url = f"{BASE_URL}/{filename}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()

    # `requests` décompresse automatiquement le Content-Encoding gzip ;
    # on lit donc le texte décodé directement.
    return pd.read_csv(io.StringIO(response.text), sep="\t")


def main() -> None:
    """Récupère les jeux de personnages et les écrit en CSV dans data/raw/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for remote_name, local_name in DATASETS.items():
        print(f"Récupération de {remote_name}...")
        df = fetch_tsv(remote_name)

        csv_path = OUTPUT_DIR / f"{local_name}.csv"
        df.to_csv(csv_path, index=False)
        print(
            f"CSV écrit : {csv_path}  "
            f"({df.shape[0]} lignes x {df.shape[1]} colonnes)"
        )


if __name__ == "__main__":
    main()