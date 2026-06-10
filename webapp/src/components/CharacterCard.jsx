// Carte d'un personnage : 6 stats éditables au stepper ou à la saisie.

import Stepper from "./Stepper.jsx";

const STATS = [
  { key: "level", label: "Niveau", min: 1, max: 20 },
  { key: "hp",    label: "PV",     min: 1, max: 500 },
  { key: "ac",    label: "CA",     min: 1, max: 30 },
  { key: "str",   label: "FOR",    min: 1, max: 30 },
  { key: "dex",   label: "DEX",    min: 1, max: 30 },
  { key: "con",   label: "CON",    min: 1, max: 30 },
];

export default function CharacterCard({ index, character, onChange, onRemove, removable }) {
  return (
    <article className="card">
      <header className="card-header">
        <span className="card-icon" aria-hidden="true">🛡️</span>
        <input
          className="name-input"
          type="text"
          value={character.name}
          placeholder={`Personnage ${index + 1}`}
          maxLength={40}
          onChange={(e) => onChange({ ...character, name: e.target.value })}
          aria-label={`Nom du personnage ${index + 1}`}
        />
        {removable && (
          <button
            type="button"
            className="remove-btn"
            onClick={onRemove}
            aria-label={`Retirer le personnage ${index + 1}`}
          >
            ✕
          </button>
        )}
      </header>
      <div className="stat-grid">
        {STATS.map(({ key, label, min, max }) => (
          <Stepper
            key={key}
            label={label}
            value={character[key]}
            min={min}
            max={max}
            onChange={(v) => onChange({ ...character, [key]: v })}
          />
        ))}
      </div>
    </article>
  );
}
