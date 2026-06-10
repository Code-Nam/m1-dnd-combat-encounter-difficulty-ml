// Affichage du verdict : bannière de difficulté + probabilités par classe.

const DIFFICULTIES = {
  Easy:   { fr: "Facile",    icon: "🗡️", css: "easy" },
  Medium: { fr: "Moyen",     icon: "⚔️", css: "medium" },
  Hard:   { fr: "Difficile", icon: "🔥", css: "hard" },
  Deadly: { fr: "Mortel",    icon: "💀", css: "deadly" },
};

export default function ResultPanel({ result }) {
  const verdict = DIFFICULTIES[result.difficulty];

  return (
    <section aria-live="polite">
      <h2 className="section-title">📜 Verdict de l'Oracle</h2>

      <div className={`result-banner ${verdict.css}`}>
        <span className="icon">{verdict.icon}</span>
        <span className="label">{verdict.fr}</span>
        <div className="confidence">confiance : {result.confidence}%</div>
      </div>

      <div className="card">
        {Object.entries(DIFFICULTIES).map(([key, { fr, css }]) => (
          <div className="proba-row" key={key}>
            <span className="proba-label">{fr}</span>
            <div className="proba-track">
              <div
                className={`proba-fill ${css}`}
                style={{ width: `${result.probabilities[key]}%` }}
              />
            </div>
            <span className="proba-value">{result.probabilities[key]}%</span>
          </div>
        ))}
      </div>
    </section>
  );
}
