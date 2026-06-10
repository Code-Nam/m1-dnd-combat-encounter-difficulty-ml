// Recherche de monstres via l'API publique Open5e (https://open5e.com).

import { parseCr } from "./stats.js";

const OPEN5E_URL = "https://api.open5e.com/v1/monsters/";

export async function searchMonsters(query, signal) {
  // name__icontains filtre sur le nom uniquement — `search` matche aussi les
  // descriptions et renvoie des résultats sans rapport.
  const url = `${OPEN5E_URL}?name__icontains=${encodeURIComponent(query)}&limit=12`;
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Open5e a répondu ${res.status}`);
  const data = await res.json();

  return data.results.map((m) => ({
    slug: m.slug,
    name: m.name,
    cr: parseCr(m.cr ?? m.challenge_rating),
    crText: String(m.challenge_rating ?? m.cr ?? "?"),
    hp: m.hit_points ?? 1,
    ac: typeof m.armor_class === "number" ? m.armor_class : 13,
    source: m.document__title ?? "",
  }));
}
