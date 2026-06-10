import { useEffect, useRef, useState } from "react";
import CharacterCard from "./components/CharacterCard.jsx";
import MonsterSearch from "./components/MonsterSearch.jsx";
import MonsterCard from "./components/MonsterCard.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import { predictDifficulty } from "./lib/api.js";
import { buildEncounterPayload } from "./lib/stats.js";

const MAX_PARTY = 6;
const DEFAULT_CHARACTER = { name: "", level: 5, hp: 35, ac: 14, str: 12, dex: 13, con: 12 };
const STORAGE_KEY = "oracle-rencontres-v1";

let nextId = 1;
const newCharacter = () => ({ id: nextId++, ...DEFAULT_CHARACTER });

// Restaure la session précédente (fermeture accidentelle de l'onglet).
// Les données invalides ou corrompues sont ignorées silencieusement.
function loadSavedState() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!Array.isArray(data?.characters) || data.characters.length === 0) return null;
    const characters = data.characters
      .slice(0, MAX_PARTY)
      .map((c) => ({ ...DEFAULT_CHARACTER, ...c, id: nextId++ }));
    const monsters = Array.isArray(data.monsters)
      ? data.monsters.filter((m) => m?.slug).map((m) => ({ ...m, qty: Number(m.qty) || 1 }))
      : [];
    return { characters, monsters };
  } catch {
    return null;
  }
}

const savedState = loadSavedState();

export default function App() {
  const [characters, setCharacters] = useState(savedState?.characters ?? [newCharacter()]);
  const [monsters, setMonsters] = useState(savedState?.monsters ?? []);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const resultRef = useRef(null);

  // Sauvegarde la session à chaque modification du groupe ou des monstres.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ characters, monsters }));
    } catch {
      // Stockage indisponible (navigation privée, quota) — l'app reste utilisable.
    }
  }, [characters, monsters]);

  const updateCharacter = (id, updated) =>
    setCharacters((chars) => chars.map((c) => (c.id === id ? { ...updated, id } : c)));

  const addMonster = (monster) =>
    setMonsters((prev) => {
      const existing = prev.find((m) => m.slug === monster.slug);
      if (existing) {
        return prev.map((m) => (m.slug === monster.slug ? { ...m, qty: m.qty + 1 } : m));
      }
      return [...prev, { ...monster, qty: 1 }];
    });

  const totalMonsters = monsters.reduce((sum, m) => sum + m.qty, 0);
  const canPredict = monsters.length > 0 && !loading;

  const resetSession = () => {
    if (!window.confirm("Réinitialiser le groupe et les monstres ?")) return;
    localStorage.removeItem(STORAGE_KEY);
    setCharacters([newCharacter()]);
    setMonsters([]);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = buildEncounterPayload(characters, monsters);
      setResult(await predictDifficulty(payload));
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header className="app-header">
        <h1>⚔️ Oracle des Rencontres</h1>
        <p>Prédit la difficulté d'un combat D&D 5e — propulsé par XGBoost</p>
      </header>

      <main>
        <section>
          <h2 className="section-title">🛡️ Groupe d'aventuriers ({characters.length}/{MAX_PARTY})</h2>
          <div className="party-grid">
            {characters.map((character, i) => (
              <CharacterCard
                key={character.id}
                index={i}
                character={character}
                removable={characters.length > 1}
                onChange={(updated) => updateCharacter(character.id, updated)}
                onRemove={() => setCharacters((chars) => chars.filter((c) => c.id !== character.id))}
              />
            ))}
            <button
              type="button"
              className="btn btn-add"
              disabled={characters.length >= MAX_PARTY}
              onClick={() => setCharacters((chars) => [...chars, newCharacter()])}
            >
              + Ajouter un personnage
            </button>
          </div>
          {characters.length === 1 && (
            <p className="warning-box">
              ⚠️ Le modèle a été entraîné sur des groupes de 2 à 6 aventuriers — la
              prédiction pour un solo est une extrapolation, à prendre avec prudence.
            </p>
          )}
        </section>

        <section>
          <h2 className="section-title">👹 Monstres{totalMonsters > 0 ? ` (${totalMonsters})` : ""}</h2>
          <MonsterSearch onAdd={addMonster} />
          {monsters.length === 0 ? (
            <p className="empty-hint">
              Aucun monstre — cherche dans le bestiaire Open5e ci-dessus.
            </p>
          ) : (
            monsters.map((monster) => (
              <MonsterCard
                key={monster.slug}
                monster={monster}
                onQtyChange={(qty) =>
                  setMonsters((prev) =>
                    prev.map((m) => (m.slug === monster.slug ? { ...m, qty: qty || 1 } : m))
                  )
                }
                onRemove={() => setMonsters((prev) => prev.filter((m) => m.slug !== monster.slug))}
              />
            ))
          )}
        </section>

        <p className="encounter-summary">
          {characters.length} aventurier{characters.length > 1 ? "s" : ""}
          {totalMonsters > 0 &&
            ` contre ${totalMonsters} monstre${totalMonsters > 1 ? "s" : ""}`}
        </p>

        <div className="reset-row">
          <button type="button" className="btn btn-reset" onClick={resetSession}>
            🧹 Réinitialiser la session
          </button>
        </div>

        {error && <div className="error-box">⚠️ {error}</div>}

        <div ref={resultRef}>{result && <ResultPanel result={result} />}</div>
      </main>

      <button type="button" className="btn btn-predict" disabled={!canPredict} onClick={handlePredict}>
        {loading ? "🎲 L'Oracle consulte les augures…" : "🎲 Prédire la difficulté"}
      </button>
    </>
  );
}
