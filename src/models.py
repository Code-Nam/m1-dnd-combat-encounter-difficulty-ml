"""Entraînement et sauvegarde du modèle XGBoost pour la difficulté D&D 5e.

Ce module charge les splits préparés par features.py, entraîne le meilleur
modèle identifié en Phase 5 (XGBoost, 82.0% test accuracy), affiche les
métriques et sauvegarde le modèle dans models/.

Usage :
    python src/models.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR   = Path(__file__).resolve().parent.parent / "results"

LABEL_NAMES: list[str] = ["Easy", "Medium", "Hard", "Deadly"]

# Features XP exclues : encodent directement le label (fuite de données).
XP_FEATURES: list[str] = ["xp_ratio", "log_xp_raw", "log_xp_adjusted"]

# Hyperparamètres XGBoost issus de la Phase 5.
XGB_PARAMS: dict = {
    "n_estimators":  300,
    "learning_rate": 0.1,    # pas de correction par round
    "max_depth":     6,      # arbres peu profonds = weak learners intentionnels
    "subsample":     0.8,    # 80 % des données par arbre = diversité + robustesse
    "objective":     "multi:softmax",
    "num_class":     4,
    "eval_metric":   "mlogloss",
    "random_state":  42,
    "n_jobs":        -1,
}


def load_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Charge les splits train/test depuis data/processed/ sans les features XP.

    Args:
        processed_dir: Dossier contenant X_train.csv, X_test.csv, y_train.csv, y_test.csv.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    X_train = pd.read_csv(processed_dir / "X_train.csv").drop(columns=XP_FEATURES)
    X_test  = pd.read_csv(processed_dir / "X_test.csv").drop(columns=XP_FEATURES)
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test  = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> XGBClassifier:
    """Entraîne un XGBClassifier avec les hyperparamètres de Phase 5.

    Args:
        X_train: Features d'entraînement.
        y_train: Labels d'entraînement (encodés 0-3).
        params:  Hyperparamètres optionnels (surcharge XGB_PARAMS).

    Returns:
        Modèle entraîné.
    """
    hp = {**XGB_PARAMS, **(params or {})}
    model = XGBClassifier(**hp)
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Calcule, affiche et retourne les métriques complètes train/test.

    Args:
        model:   Modèle entraîné.
        X_train: Features d'entraînement.
        X_test:  Features de test.
        y_train: Labels d'entraînement.
        y_test:  Labels de test.

    Returns:
        Dictionnaire structuré avec accuracy, gap, F1 par classe et matrice de confusion.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    acc_train = float(accuracy_score(y_train, y_pred_train))
    acc_test  = float(accuracy_score(y_test,  y_pred_test))
    gap       = (acc_train - acc_test) * 100

    f1_per_class = f1_score(y_test, y_pred_test, average=None)
    cm           = confusion_matrix(y_test, y_pred_test).tolist()

    print(f"Accuracy train : {acc_train:.3f} ({acc_train * 100:.1f}%)")
    print(f"Accuracy test  : {acc_test:.3f}  ({acc_test * 100:.1f}%)")
    print(f"Écart (gap)    : {gap:.1f}pp")
    print()
    print(classification_report(y_test, y_pred_test, target_names=LABEL_NAMES))

    return {
        "acc_train":        acc_train,
        "acc_test":         acc_test,
        "gap_pp":           float(gap),
        "f1_per_class":     {label: round(float(f1), 4) for label, f1 in zip(LABEL_NAMES, f1_per_class)},
        "confusion_matrix": cm,
        "hyperparameters":  XGB_PARAMS,
    }


def save_metrics(metrics: dict, path: Path) -> None:
    """Sauvegarde les métriques dans un fichier JSON horodaté.

    Args:
        metrics: Dictionnaire retourné par evaluate().
        path:    Chemin du fichier JSON de sortie.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Métriques sauvegardées : {path}")


def save_model(model: XGBClassifier, path: Path) -> None:
    """Sauvegarde le modèle avec joblib.

    Args:
        model: Modèle entraîné à sauvegarder.
        path:  Chemin complet du fichier de sortie (.joblib).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path: Path) -> XGBClassifier:
    """Charge un modèle sauvegardé avec joblib.

    Args:
        path: Chemin vers le fichier .joblib.

    Returns:
        Modèle XGBClassifier chargé.
    """
    return joblib.load(path)


def main() -> None:
    """Pipeline complet : chargement → entraînement → évaluation → sauvegarde."""
    print("=== Chargement des données ===")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}\n")

    print("=== Entraînement XGBoost ===")
    model = train_xgboost(X_train, y_train)
    print("Entraînement terminé.\n")

    print("=== Évaluation ===")
    metrics = evaluate(model, X_train, X_test, y_train, y_test)

    print("=== Sauvegarde ===")
    save_model(model, MODELS_DIR / "xgboost_difficulty.joblib")
    save_metrics(metrics, RESULTS_DIR / "metrics.json")


if __name__ == "__main__":
    main()
