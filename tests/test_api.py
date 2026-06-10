"""Tests de l'API REST D&D 5e Difficulty Classifier.

Utilise le TestClient de FastAPI (starlette) — pas besoin de lancer le serveur.
Le lifespan est déclenché par le context manager `with TestClient(app)`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ── Données de test ────────────────────────────────────────────────────────────

VALID_ENCOUNTER = {
    "party_size":      4,
    "party_avg_level": 5.0,
    "party_avg_hp":    35.0,
    "party_avg_ac":    14.0,
    "party_avg_str":   12.0,
    "party_avg_dex":   13.0,
    "party_avg_con":   12.0,
    "monster_count":   3,
    "monster_avg_cr":  1.0,
    "monster_avg_hp":  30.0,
    "monster_avg_ac":  13.0,
}

# 1 gobelin (CR 1/4) contre 4 personnages niveau 10 → Easy
EASY_ENCOUNTER = {
    "party_size":      4,
    "party_avg_level": 10,
    "party_avg_hp":    60.0,
    "party_avg_ac":    16.0,
    "monster_count":   1,
    "monster_avg_cr":  0.25,
    "monster_avg_hp":  7.0,
    "monster_avg_ac":  13.0,
}

# 6 zombies contre 2 personnages niveau 1 → Deadly
DEADLY_ENCOUNTER = {
    "party_size":      2,
    "party_avg_level": 1,
    "party_avg_hp":    9.0,
    "party_avg_ac":    12.0,
    "monster_count":   6,
    "monster_avg_cr":  0.25,
    "monster_avg_hp":  22.0,
    "monster_avg_ac":  8.0,
}

LABEL_NAMES = {"Easy", "Medium", "Hard", "Deadly"}


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Client HTTP — le modèle est chargé une seule fois via le lifespan."""
    with TestClient(app) as c:
        yield c


# ── GET / ──────────────────────────────────────────────────────────────────────

def test_root_returns_200(client: TestClient) -> None:
    assert client.get("/").status_code == 200


def test_root_status_ok(client: TestClient) -> None:
    assert client.get("/").json()["status"] == "ok"


# ── POST /predict — structure de la réponse ────────────────────────────────────

def test_predict_returns_200(client: TestClient) -> None:
    assert client.post("/predict", json=VALID_ENCOUNTER).status_code == 200


def test_predict_has_required_keys(client: TestClient) -> None:
    data = client.post("/predict", json=VALID_ENCOUNTER).json()
    assert {"difficulty", "confidence", "probabilities"}.issubset(data.keys())


def test_predict_difficulty_is_valid_class(client: TestClient) -> None:
    data = client.post("/predict", json=VALID_ENCOUNTER).json()
    assert data["difficulty"] in LABEL_NAMES


def test_predict_probabilities_have_all_classes(client: TestClient) -> None:
    data = client.post("/predict", json=VALID_ENCOUNTER).json()
    assert set(data["probabilities"].keys()) == LABEL_NAMES


def test_predict_probabilities_sum_to_100(client: TestClient) -> None:
    data = client.post("/predict", json=VALID_ENCOUNTER).json()
    total = sum(data["probabilities"].values())
    assert abs(total - 100.0) < 0.5


def test_predict_confidence_matches_predicted_class(client: TestClient) -> None:
    data = client.post("/predict", json=VALID_ENCOUNTER).json()
    assert data["confidence"] == data["probabilities"][data["difficulty"]]


def test_predict_confidence_is_positive(client: TestClient) -> None:
    data = client.post("/predict", json=VALID_ENCOUNTER).json()
    assert 0.0 < data["confidence"] <= 100.0


# ── POST /predict — cas extrêmes ───────────────────────────────────────────────

def test_predict_easy_encounter(client: TestClient) -> None:
    data = client.post("/predict", json=EASY_ENCOUNTER).json()
    assert data["difficulty"] == "Easy"


def test_predict_deadly_encounter(client: TestClient) -> None:
    data = client.post("/predict", json=DEADLY_ENCOUNTER).json()
    assert data["difficulty"] == "Deadly"


def test_predict_default_str_dex_con_accepted(client: TestClient) -> None:
    """party_avg_str/dex/con sont optionnels — la requête doit passer sans eux."""
    body = {k: v for k, v in VALID_ENCOUNTER.items()
            if k not in ("party_avg_str", "party_avg_dex", "party_avg_con")}
    assert client.post("/predict", json=body).status_code == 200


# ── POST /predict — validation Pydantic (422) ──────────────────────────────────

def test_predict_missing_required_field_returns_422(client: TestClient) -> None:
    body = {k: v for k, v in VALID_ENCOUNTER.items() if k != "party_size"}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_party_size_too_small_returns_422(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID_ENCOUNTER, "party_size": 0}).status_code == 422


def test_predict_solo_party_accepted(client: TestClient) -> None:
    """Groupe de 1 accepté (extrapolation — le modèle est entraîné sur 2-6)."""
    assert client.post("/predict", json={**VALID_ENCOUNTER, "party_size": 1}).status_code == 200


def test_predict_party_size_too_large_returns_422(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID_ENCOUNTER, "party_size": 7}).status_code == 422


def test_predict_monster_count_zero_returns_422(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID_ENCOUNTER, "monster_count": 0}).status_code == 422


def test_predict_negative_hp_returns_422(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID_ENCOUNTER, "party_avg_hp": -5.0}).status_code == 422


def test_predict_invalid_type_returns_422(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID_ENCOUNTER, "party_avg_hp": "abc"}).status_code == 422
