# D&D 5e Combat Encounter Difficulty — ML Classifier

Modèle de machine learning pour prédire la difficulté d'un combat D&D 5e (**Easy / Medium / Hard / Deadly**) à partir des statistiques brutes du groupe de joueurs et des monstres, sans utiliser la formule XP officielle du DMG.

---

## Résultats

| Modèle | Accuracy (test) | Écart overfitting |
|---|---|---|
| Logistic Regression | 69.0% | 2.5pp |
| Decision Tree | 71.5% | 19.8pp |
| Random Forest | 77.1% | 22.9pp |
| **XGBoost** | **82.0%** | **18.0pp** |

Les erreurs du modèle se concentrent aux frontières entre classes adjacentes (Easy↔Medium, Medium↔Hard, Hard↔Deadly) — exactement là où la méthode officielle est elle-même ambiguë.

---

## Structure du projet

```
├── data/
│   ├── raw/                  # Données brutes (monstres, personnages, encounters)
│   └── processed/            # Splits train/test prêts à l'entraînement
├── models/
│   └── xgboost_difficulty.joblib   # Modèle entraîné
├── notebooks/
│   ├── 02a_exploration_monsters.ipynb
│   ├── 02b_exploration_characters.ipynb
│   ├── 03_encounter_generation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_base_model.ipynb         # LR, DT, RF, XGBoost + comparatif
│   ├── 06_analysis.ipynb           # SHAP, comparaison DMG, bilan
│   └── 07_hyperparameter_tuning.ipynb  # CV baseline, RandomizedSearch, early stopping
├── results/
│   ├── 05_base_model/        # Graphiques des modèles (evaluation, learning curves...)
│   ├── 06_analysis/          # SHAP plots, waterfall, comparaison DMG
│   └── 07_hyperparameter_tuning/   # CV scores, impact params, baseline vs tuné
├── src/
│   ├── fetch_monsters.py     # Téléchargement du bestiaire (API Open5e)
│   ├── fetch_characters.py   # Téléchargement des personnages
│   ├── data_generator.py     # Génération de rencontres synthétiques (DMG p.82)
│   ├── features.py           # Feature engineering + train/test split
│   ├── models.py             # Entraînement, évaluation, sauvegarde XGBoost
│   └── infer.py              # Script d'inférence CLI
└── tests/
    ├── conftest.py
    ├── test_data_generator.py
    ├── test_features.py
    ├── test_models.py
    └── test_api.py
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
```

---

## Utilisation

### Inférence — tester le modèle avec `src/infer.py`

Le script calcule automatiquement les features dérivées (`cr_level_delta`, `hp_ratio`, etc.)
à partir des stats brutes. Seuls les paramètres ci-dessous sont nécessaires.

**Paramètres disponibles**

| Paramètre | Description | Défaut |
| --- | --- | --- |
| `--party-size` | Nombre de personnages (2–6) | 4 |
| `--party-avg-level` | Niveau moyen du groupe (1–20) | 5.0 |
| `--party-avg-hp` | HP moyen des personnages | 35.0 |
| `--party-avg-ac` | AC moyen des personnages | 14.0 |
| `--party-avg-str/dex/con` | Stats moyennes | 12 / 13 / 12 |
| `--monster-count` | Nombre de monstres | 3 |
| `--monster-avg-cr` | CR moyen des monstres | 1.0 |
| `--monster-avg-hp` | HP moyen des monstres | 30.0 |
| `--monster-avg-ac` | AC moyen des monstres | 13.0 |
| `--example` | Lance une rencontre prédéfinie | — |
| `--json` | Retourne le résultat en JSON | — |

**Exemples couvrant les 4 classes**

```bash
# Easy — 1 gobelin (CR 1/4) contre 4 personnages niveau 10
python src/infer.py \
    --party-size 4 --party-avg-level 10 --party-avg-hp 60 \
    --party-avg-ac 16 --party-avg-str 12 --party-avg-dex 14 --party-avg-con 13 \
    --monster-count 1 --monster-avg-cr 0.25 --monster-avg-hp 7 --monster-avg-ac 13

# Medium — 2 orcs (CR 1/2) contre 3 personnages niveau 3
python src/infer.py \
    --party-size 3 --party-avg-level 3 --party-avg-hp 22 \
    --party-avg-ac 13 --party-avg-str 11 --party-avg-dex 11 --party-avg-con 11 \
    --monster-count 2 --monster-avg-cr 0.5 --monster-avg-hp 15 --monster-avg-ac 13

# Hard — 4 loups (CR 1/4) contre 3 personnages niveau 2
python src/infer.py \
    --party-size 3 --party-avg-level 2 --party-avg-hp 15 \
    --party-avg-ac 13 --party-avg-str 11 --party-avg-dex 11 --party-avg-con 11 \
    --monster-count 4 --monster-avg-cr 0.25 --monster-avg-hp 11 --monster-avg-ac 13

# Deadly — 6 zombies (CR 1/4) contre 2 personnages niveau 1
python src/infer.py \
    --party-size 2 --party-avg-level 1 --party-avg-hp 9 \
    --party-avg-ac 12 --party-avg-str 10 --party-avg-dex 10 --party-avg-con 10 \
    --monster-count 6 --monster-avg-cr 0.25 --monster-avg-hp 22 --monster-avg-ac 8
```

**Sortie type**

```text
─── Rencontre ──────────────────────────────────────
  Groupe : 2 personnages  | niveau moyen 1.0  | HP moy. 9  | AC moy. 12
  Monstres : 6  | CR moyen 0.25  | HP moy. 22  | AC moy. 8

─── Prédiction ─────────────────────────────────────
  🔴  Deadly  (confiance : 99.4%)

  Probabilités par classe :
    Easy     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.1%
    Medium   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.2%
    Hard     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.4%
    Deadly   █████████████████████████████░  99.4% ◄
```

**Sortie JSON** (pour intégration dans un autre script)

```bash
python src/infer.py --example --json
```

```json
{
  "difficulty": "Deadly",
  "confidence": 99.4,
  "probabilities": { "Easy": 0.1, "Medium": 0.2, "Hard": 0.4, "Deadly": 99.4 },
  "input": { "party_size": 2, "party_avg_level": 1.0, "..." : "..." }
}
```

### API REST (Phase 7)

**Sans Docker**

```bash
uvicorn api.main:app --reload
```

**Avec Docker**

```bash
docker compose up --build
```

L'API démarre sur `http://localhost:8000`. Documentation interactive disponible sur `/docs`.

**Endpoint principal**

```
POST /predict
Content-Type: application/json
```

```json
{
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
  "monster_avg_ac": 13.0
}
```

```json
{
  "difficulty": "Hard",
  "confidence": 84.2,
  "probabilities": {
    "Easy": 1.1,
    "Medium": 8.3,
    "Hard": 84.2,
    "Deadly": 6.4
  }
}
```

Les features dérivées (`cr_level_delta`, `hp_ratio`, `ac_gap`…) sont calculées automatiquement — seules les stats brutes sont nécessaires.

---

### Pipeline complet (re-génération depuis zéro)

```bash
# 1. Récupérer les données brutes
python src/fetch_monsters.py
python src/fetch_characters.py

# 2. Générer les rencontres synthétiques (4 000 combats, 1 000 par classe)
python src/data_generator.py

# 3. Construire les features et les splits train/test
python src/features.py

# 4. Entraîner le modèle et le sauvegarder
python src/models.py
```

### Tests

```bash
pytest tests/ -v
```

73 tests couvrant `data_generator`, `features`, `models` et l'API REST.

### Notebooks

```bash
jupyter notebook
```

Les notebooks sont numérotés dans l'ordre d'exécution (02 → 07).

---

## Features du modèle

| Feature | Description |
|---|---|
| `party_size` | Nombre de personnages (2–6) |
| `party_avg_level` | Niveau moyen du groupe |
| `party_avg_ac/str/dex/con` | Stats moyennes du groupe |
| `monster_count` | Nombre de monstres |
| `monster_avg_cr` | CR moyen des monstres |
| `monster_avg_ac` | AC moyenne des monstres |
| `cr_level_delta` | `monster_avg_cr − party_avg_level` (rapport de force) |
| `hp_ratio` | `monster_avg_hp / party_avg_hp` (endurance relative) |
| `ac_gap` | `monster_avg_ac − party_avg_ac` (avantage défensif) |
| `log_monster_avg_hp` | `log1p(monster_avg_hp)` |
| `log_party_avg_hp` | `log1p(party_avg_hp)` |

Les features XP (`xp_ratio`, `log_xp_raw`, `log_xp_adjusted`) sont exclues — elles encodent directement la formule du label et introduisent une fuite de données.

---

## Stack

| Catégorie | Librairies |
|---|---|
| Données | pandas, numpy |
| Modèles | scikit-learn, xgboost |
| Visualisation | matplotlib, seaborn, yellowbrick |
| Interprétabilité | shap |
| Suivi d'entraînement | tqdm (barres de progression), xgboost eval_set (courbe de loss) |
| Tests | pytest |
| API (Phase 7) | fastapi, pydantic, uvicorn |
