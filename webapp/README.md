# Oracle des Rencontres — Webapp

SPA React + Vite pour prédire la difficulté d'un combat D&D 5e via l'API du projet.

## Fonctionnalités

- **Groupe d'aventuriers** : 1 à 6 personnages renommables, chaque stat (Niveau, PV, CA,
  FOR, DEX, CON) ajustable par flèches ▲/▼ ou saisie directe. Les moyennes attendues
  par l'API sont calculées automatiquement côté client.
- **Monstres** : recherche dans le bestiaire [Open5e](https://open5e.com) (filtre par nom),
  plusieurs monstres cumulables avec quantité par monstre. CR fractionnaires ("1/4") gérés.
- **Verdict** : difficulté prédite (Facile / Moyen / Difficile / Mortel) avec confiance
  et probabilités par classe.
- **Session persistante** : le groupe et les monstres sont sauvegardés dans le
  `localStorage` — fermer l'onglet par accident ne fait rien perdre.
- Design mobile-first, thème D&D (parchemin, or, rouge sang, police Cinzel).

## Lancer en développement

L'API doit tourner (depuis la racine du projet) :

```bash
uvicorn api.main:app --reload
```

Puis :

```bash
cd webapp
npm install
npm run dev
```

L'app démarre sur `http://localhost:5173`.

## Configuration

Copier `.env.example` en `.env` et adapter :

```bash
cp .env.example .env
```

| Variable | Description | Défaut |
| --- | --- | --- |
| `VITE_API_URL` | URL de l'API de prédiction | `http://localhost:8000` |

## Lancer avec Docker

Depuis la racine du projet, API et webapp ensemble (hot reload) :

```bash
docker compose up --build
```

Le `Dockerfile` de la webapp a deux cibles : `dev` (serveur Vite, utilisé par
docker-compose) et `prod` (build statique servi par nginx).

## Build de production

```bash
npm run build      # génère dist/
npm run preview    # sert le build localement
```

## Structure

```
src/
├── App.jsx                    # Assemblage : groupe, monstres, prédiction
├── index.css                  # Thème D&D mobile-first
├── components/
│   ├── Stepper.jsx            # Champ numérique ▲/▼ + saisie directe
│   ├── CharacterCard.jsx      # Carte personnage (6 stats)
│   ├── MonsterSearch.jsx      # Recherche Open5e avec debounce
│   ├── MonsterCard.jsx        # Monstre sélectionné + quantité
│   └── ResultPanel.jsx        # Bannière de verdict + probabilités
└── lib/
    ├── stats.js               # parseCr + agrégation du payload /predict
    ├── api.js                 # Client de l'API FastAPI
    └── open5e.js              # Client du bestiaire Open5e
```

## Note sur les groupes de 1

Le modèle a été entraîné sur des groupes de 2 à 6 personnages. La prédiction pour
un aventurier solo est une **extrapolation** hors distribution d'entraînement —
à prendre avec prudence.
