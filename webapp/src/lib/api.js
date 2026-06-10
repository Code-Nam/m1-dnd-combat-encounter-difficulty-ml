// Client de l'API de prédiction (FastAPI, voir ../api/main.py).

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function predictDifficulty(payload) {
  let res;
  try {
    res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      `Impossible de joindre l'API (${API_URL}). Lance-la avec : uvicorn api.main:app --reload`
    );
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => `${d.loc?.at(-1)} : ${d.msg}`).join(" — ")
      : detail ?? `Erreur API (${res.status})`;
    throw new Error(message);
  }

  return res.json();
}
