// Recherche de monstres dans le bestiaire Open5e, avec debounce.

import { useEffect, useRef, useState } from "react";
import { searchMonsters } from "../lib/open5e.js";

const DEBOUNCE_MS = 400;

export default function MonsterSearch({ onAdd }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setError(null);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const found = await searchMonsters(query.trim(), controller.signal);
        setResults(found);
        setError(null);
      } catch (e) {
        if (e.name !== "AbortError") {
          setError("Recherche Open5e indisponible — réessaie dans un instant.");
          setResults([]);
        }
      } finally {
        setLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  // Ferme la liste si on clique ailleurs
  useEffect(() => {
    const close = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setResults([]);
      }
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);

  const handlePick = (monster) => {
    onAdd(monster);
    setQuery("");
    setResults([]);
  };

  return (
    <div className="monster-search" ref={containerRef}>
      <input
        className="search-input"
        type="search"
        placeholder="Rechercher un monstre… (ex. goblin, dragon)"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Rechercher un monstre dans le bestiaire Open5e"
      />
      {loading && <p className="search-status">Consultation du bestiaire…</p>}
      {error && <p className="search-status">{error}</p>}
      {results.length > 0 && (
        <ul className="search-results">
          {results.map((m) => (
            <li key={m.slug}>
              <button type="button" className="search-result" onClick={() => handlePick(m)}>
                <span className="name">{m.name}</span>
                <span className="meta">
                  CR {m.crText} · {m.hp} PV · CA {m.ac}
                  {m.source ? ` · ${m.source}` : ""}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
