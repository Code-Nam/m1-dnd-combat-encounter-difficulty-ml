# Contexte du projet

Lorsque vous travaillez avec cette base de code, donnez la priorité à la clarté plutôt qu'à l'optimisation prématurée. Posez des questions avant de changer la structure du projet.

## À propos de ce projet

Modèle ML pour prédire la difficulté des combats D&D 5e (Facile/Moyen/Difficile/Mortel). Le projet passe par la génération de données synthétiques, l'exploration, l'ingénierie des features, et l'entraînement progressif du modèle.

- *Approche**: Apprentissage pas-à-pas, chaque phase consolide la précédente.

## Répertoires clés

- `data/raw/` - Données générées synthétiquement
- `data/processed/` - Données nettoyées et préparées
- `notebooks/` - Notebooks Jupyter pour l'exploration et l'apprentissage
- `src/` - Code Python réutilisable (générateur, features, modèles)
- `models/` - Modèles entraînés sauvegardés
- `results/` - Métriques, graphiques, analyses

## Normes

- Type hints requis pour toutes les fonctions
- Lisibilité avant l'optimisation
- Docstrings explicatives sur chaque module
- pytest pour les tests (fixtures dans `tests/conftest.py`)
- PEP 8 avec des lignes de 100 caractères

## Stack technologique

- *Apprentissage (essentiels)**
- pandas, numpy - Manipulation et exploration des données
- scikit-learn, xgboost - Modèles ML
- matplotlib, seaborn - Visualisation
- jupyter - Notebooks interactifs
- *Déploiement (optionnel, Phase 7)**
- fastapi - API REST pour le modèle
- pydantic - Validation des données
- docker - Conteneurisation
- pytest - Tests unitaires

## Commandes courantes

```bash

# Environnement

python -m venv venv

source venv/bin/activate                    # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Développement

jupyter notebook                             # Lancer les notebooks

pytest tests/ -v                            # Exécuter les tests

# Modèle

python src/data_generator.py                # Générer données synthétiques

python src/models.py                        # Entraîner le modèle

# API (Phase 7)

uvicorn api.main:app --reload               # Serveur développement

```

## Plan de développement (7 phases)

### Phase 1: Génération de données synthétiques

Créer 500-1000 combats D&D avec features: niveau du groupe, CR moyen, AC moyen, etc.

Labeller chaque combat: Facile/Moyen/Difficile/Mortel

### Phase 2: Exploration et nettoyage

Charger les données, explorer les distributions, vérifier la qualité (valeurs manquantes, doublons)

### Phase 3: Ingénierie des features

Créer features dérivées: damage_ratio, action_economy_ratio, ac_gap, etc.

### Phase 4: Modèle de base

Entraîner un DecisionTree ou RandomForest simple, évaluer la performance

### Phase 5: Amélioration du modèle

Essayer XGBoost, tuner les hyperparamètres, valider avec cross-validation

### Phase 6: Analyse et interprétation

Feature importance, comparaison avec la méthode XP officielle, leçons apprises

### Phase 7: Déploiement (optionnel)

Créer API FastAPI, conteneuriser avec Docker

## Concepts clés à maîtriser

- *D&D Mechanics**
- CR (Challenge Rating): Difficulté officielle du monstre
- AC (Armor Class): Plus élevé = plus difficile à toucher
- HP: Points de vie = durabilité
- Action Economy: Nombre d'actions disponibles par tour
- *ML Concepts**
- Classification: Prédire une catégorie parmi N (ici: 4 difficultés)
- Features: Colonnes qui décrivent les données
- Train/Test Split: 80% entraînement, 20% test
- Overfitting: Mémorisation au lieu d'apprentissage généraliste
- Metrics: Accuracy, Precision, Recall, F1-Score
- *Python Skills**
- pandas: df.head(), df.describe(), créer colonnes
- sklearn: train_test_split, fit, predict, evaluate
- Visualisation: matplotlib plots, confusion matrices

## Remarques importantes

- Les modèles basés sur arbres (RandomForest, XGBoost) performent mieux sur données tabulaires que les réseaux de neurones
- Toujours splitter les données AVANT tout preprocessing
- Ne pas évaluer sur les données d'entraînement (sur-optimiste)
- Commencer simple, ajouter de la complexité progressivement
- Documenter les décisions de design au fur et à mesure

## Quand demander de l'aide

Incluez:

- Quelle phase vous êtes
- Ce que vous avez essayé
- L'erreur exacte ou résultat obtenu
- Shape des données et types de colonnes si relevant

Exemple:

```

Phase 3, feature engineering. Je crée damage_ratio.

Code: df['damage_ratio'] = df['party_dpr'] / df['monster_dpr']

Erreur: ZeroDivisionError ligne 45

Données: 500 rows × 12 columns, types: float/int

```