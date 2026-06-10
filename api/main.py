"""API REST pour la prédiction de difficulté D&D 5e.

Expose un endpoint POST /predict qui prend les stats brutes d'une rencontre
et retourne la difficulté prédite par le modèle XGBoost.

Usage :
    uvicorn api.main:app --reload
    # puis POST http://localhost:8000/predict
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import build_feature_vector
from src.models import LABEL_NAMES, load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_difficulty.joblib"

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lance d'abord `python src/models.py` pour entraîner et sauvegarder le modèle."
        )
    _model = load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="D&D 5e Combat Difficulty API",
    description="Prédit la difficulté d'un combat D&D 5e — Easy / Medium / Hard / Deadly",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schémas d'entrée / sortie ──────────────────────────────────────────────────

class EncounterInput(BaseModel):
    """Stats brutes d'une rencontre D&D 5e."""

    party_size:      int   = Field(..., ge=2, le=6,   description="Nombre de personnages")
    party_avg_level: float = Field(..., ge=1, le=20,  description="Niveau moyen du groupe")
    party_avg_hp:    float = Field(..., gt=0,          description="HP moyen des personnages")
    party_avg_ac:    float = Field(..., gt=0,          description="AC moyen des personnages")
    party_avg_str:   float = Field(12.0, gt=0,         description="Force moyenne")
    party_avg_dex:   float = Field(13.0, gt=0,         description="Dextérité moyenne")
    party_avg_con:   float = Field(12.0, gt=0,         description="Constitution moyenne")
    monster_count:   int   = Field(..., ge=1,           description="Nombre de monstres")
    monster_avg_cr:  float = Field(..., ge=0,           description="CR moyen des monstres")
    monster_avg_hp:  float = Field(..., gt=0,           description="HP moyen des monstres")
    monster_avg_ac:  float = Field(..., gt=0,           description="AC moyen des monstres")

    model_config = {
        "json_schema_extra": {
            "example": {
                "party_size": 4,
                "party_avg_level": 5.0,
                "party_avg_hp": 35.0,
                "party_avg_ac": 14.0,
                "party_avg_str": 12.0,
                "party_avg_dex": 13.0,
                "party_avg_con": 12.0,
                "monster_count": 5,
                "monster_avg_cr": 0.5,
                "monster_avg_hp": 15.0,
                "monster_avg_ac": 13.0,
            }
        }
    }


class PredictionOutput(BaseModel):
    """Résultat de la prédiction."""

    difficulty:    str              = Field(..., description="Classe prédite : Easy / Medium / Hard / Deadly")
    confidence:    float            = Field(..., description="Probabilité de la classe prédite (%)")
    probabilities: dict[str, float] = Field(..., description="Probabilités pour chaque classe (%)")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Statut de l'API")
def root() -> dict:
    return {
        "status": "ok",
        "model":  "XGBoost D&D 5e Difficulty Classifier",
        "docs":   "/docs",
    }


@app.post("/predict", response_model=PredictionOutput, summary="Prédit la difficulté d'une rencontre")
def predict(encounter: EncounterInput) -> PredictionOutput:
    """Prédit la difficulté d'un combat D&D 5e à partir des stats brutes.

    Les features dérivées (`cr_level_delta`, `hp_ratio`, `ac_gap`, etc.)
    sont calculées automatiquement — seules les stats brutes sont nécessaires.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    features = build_feature_vector(**encounter.model_dump())
    probas   = _model.predict_proba(features)[0]
    pred     = int(_model.predict(features)[0])

    return PredictionOutput(
        difficulty=LABEL_NAMES[pred],
        confidence=round(float(probas[pred]) * 100, 1),
        probabilities={
            label: round(float(p) * 100, 1)
            for label, p in zip(LABEL_NAMES, probas)
        },
    )
