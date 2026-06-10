// Calculs : conversion CR et agrégation des stats pour l'API de prédiction.

/**
 * Convertit un Challenge Rating Open5e en nombre.
 * Gère les fractions ("1/8", "1/4", "1/2") et les valeurs numériques.
 */
export function parseCr(cr) {
  if (typeof cr === "number") return cr;
  if (cr == null) return 0;
  const s = String(cr).trim();
  if (s.includes("/")) {
    const [num, den] = s.split("/").map(Number);
    return den ? num / den : 0;
  }
  const n = Number(s);
  return Number.isFinite(n) ? n : 0;
}

const round2 = (x) => Math.round(x * 100) / 100;

/**
 * Construit le payload attendu par POST /predict à partir des personnages
 * individuels et des monstres sélectionnés (avec quantités).
 *
 * L'API attend des moyennes : on agrège ici côté client.
 */
export function buildEncounterPayload(characters, monsters) {
  const n = characters.length;
  const avg = (key) => round2(characters.reduce((sum, c) => sum + Number(c[key]), 0) / n);

  const totalMonsters = monsters.reduce((sum, m) => sum + m.qty, 0);
  const wavg = (key) =>
    round2(monsters.reduce((sum, m) => sum + Number(m[key]) * m.qty, 0) / totalMonsters);

  return {
    party_size: n,
    party_avg_level: avg("level"),
    party_avg_hp: avg("hp"),
    party_avg_ac: avg("ac"),
    party_avg_str: avg("str"),
    party_avg_dex: avg("dex"),
    party_avg_con: avg("con"),
    monster_count: totalMonsters,
    monster_avg_cr: wavg("cr"),
    monster_avg_hp: wavg("hp"),
    monster_avg_ac: wavg("ac"),
  };
}
