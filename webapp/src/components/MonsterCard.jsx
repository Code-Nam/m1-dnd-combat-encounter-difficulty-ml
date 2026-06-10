// Monstre sélectionné : infos + quantité ajustable + retrait.

import Stepper from "./Stepper.jsx";

export default function MonsterCard({ monster, onQtyChange, onRemove }) {
  return (
    <article className="card monster-card">
      <div className="monster-info">
        <span className="name">👹 {monster.name}</span>
        <span className="meta">
          CR {monster.crText} · {monster.hp} PV · CA {monster.ac}
        </span>
      </div>
      <div className="monster-qty">
        <Stepper label="Nombre" value={monster.qty} min={1} max={20} onChange={onQtyChange} />
      </div>
      <button
        type="button"
        className="remove-btn"
        onClick={onRemove}
        aria-label={`Retirer ${monster.name}`}
      >
        ✕
      </button>
    </article>
  );
}
