// Champ numérique avec flèches ▲/▼ — saisie directe possible.

export default function Stepper({ label, value, min = 0, max = 99, step = 1, onChange }) {
  const clamp = (v) => Math.min(max, Math.max(min, v));

  const nudge = (dir) => {
    const base = Number(value);
    const next = (Number.isFinite(base) ? base : min) + dir * step;
    onChange(clamp(Math.round(next * 100) / 100));
  };

  return (
    <label className="stepper">
      <span className="stepper-label">{label}</span>
      <div className="stepper-controls">
        <button
          type="button"
          className="stepper-btn"
          onClick={() => nudge(-1)}
          aria-label={`Diminuer ${label}`}
        >
          ▼
        </button>
        <input
          className="stepper-input"
          type="number"
          inputMode="numeric"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
          onBlur={() => onChange(clamp(Number(value) || min))}
        />
        <button
          type="button"
          className="stepper-btn"
          onClick={() => nudge(1)}
          aria-label={`Augmenter ${label}`}
        >
          ▲
        </button>
      </div>
    </label>
  );
}
