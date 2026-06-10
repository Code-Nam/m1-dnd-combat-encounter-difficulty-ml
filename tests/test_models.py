"""Tests unitaires pour src/models.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.models import evaluate, load_best_params, load_model, save_metrics, save_model, train_xgboost


# ---------------------------------------------------------------------------
# load_best_params
# ---------------------------------------------------------------------------


def test_load_best_params_returns_dict() -> None:
    params = load_best_params()
    assert isinstance(params, dict)


def test_load_best_params_has_required_keys() -> None:
    required = {"n_estimators", "learning_rate", "max_depth", "subsample",
                "objective", "num_class", "eval_metric", "random_state"}
    assert required.issubset(load_best_params().keys())


def test_load_best_params_fallback_without_json(tmp_path, monkeypatch) -> None:
    """Sans best_params.json, doit retourner XGB_PARAMS par défaut."""
    import src.models as m
    monkeypatch.setattr(m, "BEST_PARAMS_PATH", tmp_path / "nonexistent.json")
    params = load_best_params()
    assert params["n_estimators"] == m.XGB_PARAMS["n_estimators"]


def test_load_best_params_loads_json(tmp_path, monkeypatch) -> None:
    """Avec best_params.json, les valeurs tuned remplacent XGB_PARAMS."""
    import json, src.models as m
    json_path = tmp_path / "best_params.json"
    json_path.write_text(json.dumps({"n_estimators": 42, "max_depth": 3}), encoding="utf-8")
    monkeypatch.setattr(m, "BEST_PARAMS_PATH", json_path)
    params = load_best_params()
    assert params["n_estimators"] == 42
    assert params["max_depth"] == 3


# ---------------------------------------------------------------------------
# train_xgboost
# ---------------------------------------------------------------------------


def test_train_xgboost_returns_xgbclassifier(tiny_splits) -> None:
    X_train, _, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train)
    assert isinstance(model, XGBClassifier)


def test_train_xgboost_can_predict(tiny_splits) -> None:
    X_train, X_test, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1, 2, 3})


def test_train_xgboost_custom_params(tiny_splits) -> None:
    X_train, _, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train, params={"n_estimators": 10, "max_depth": 3})
    assert model.n_estimators == 10
    assert model.max_depth == 3


def test_train_xgboost_perfect_train_accuracy(tiny_splits) -> None:
    """Un modèle sans contrainte doit mémoriser les données d'entraînement."""
    X_train, _, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_train)
    accuracy = (preds == y_train.values).mean()
    assert accuracy == 1.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict_with_keys(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    assert {"acc_train", "acc_test", "gap_pp", "f1_per_class",
            "confusion_matrix", "hyperparameters"}.issubset(metrics.keys())


def test_evaluate_accuracy_between_0_and_1(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    assert 0.0 <= metrics["acc_train"] <= 1.0
    assert 0.0 <= metrics["acc_test"]  <= 1.0


def test_evaluate_gap_equals_difference(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    expected_gap = (metrics["acc_train"] - metrics["acc_test"]) * 100
    assert abs(metrics["gap_pp"] - expected_gap) < 1e-6


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


def test_save_model_creates_file(tiny_splits) -> None:
    X_train, _, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        save_model(model, path)
        assert path.exists()


def test_load_model_returns_xgbclassifier(tiny_splits) -> None:
    X_train, _, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        save_model(model, path)
        loaded = load_model(path)
    assert isinstance(loaded, XGBClassifier)


def test_loaded_model_same_predictions(tiny_splits) -> None:
    """Le modèle chargé doit produire exactement les mêmes prédictions."""
    X_train, X_test, y_train, _ = tiny_splits
    model = train_xgboost(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        save_model(model, path)
        loaded = load_model(path)
    np.testing.assert_array_equal(model.predict(X_test), loaded.predict(X_test))


# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------


def test_save_metrics_creates_json(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model   = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        save_metrics(metrics, path)
        assert path.exists()


def test_save_metrics_valid_json(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model   = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        save_metrics(metrics, path)
        data = json.loads(path.read_text())

    assert "trained_at" in data
    assert "acc_train" in data
    assert "acc_test" in data
    assert "gap_pp" in data
    assert "f1_per_class" in data
    assert "confusion_matrix" in data
    assert "hyperparameters" in data


def test_save_metrics_f1_keys(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model   = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        save_metrics(metrics, path)
        data = json.loads(path.read_text())

    assert set(data["f1_per_class"].keys()) == {"Easy", "Medium", "Hard", "Deadly"}


def test_save_metrics_confusion_matrix_shape(tiny_splits) -> None:
    X_train, X_test, y_train, y_test = tiny_splits
    model   = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        save_metrics(metrics, path)
        data = json.loads(path.read_text())

    cm = data["confusion_matrix"]
    assert len(cm) == 4
    assert all(len(row) == 4 for row in cm)
