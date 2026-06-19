const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  dashboardStats: () => request("/dashboard/stats"),
  modelComparison: () => request("/dashboard/model_comparison"),
  featureCatalog: () => request("/dashboard/feature_catalog"),

  listAttacks: (limit = 50) => request(`/attacks?limit=${limit}`),
  simulateAttack: (payload) =>
    request("/attacks/simulate", { method: "POST", body: JSON.stringify(payload) }),
  mitigateAttack: (id) => request(`/attacks/${id}/mitigate`, { method: "POST" }),

  detectAttack: (payload) =>
    request("/detect_attack", { method: "POST", body: JSON.stringify(payload) }),

  listRules: (limit = 50) => request(`/defense/rules?limit=${limit}`),
  toggleRule: (id) => request(`/defense/rules/${id}/toggle`, { method: "POST" }),

  listWhitelist: () => request("/defense/whitelist"),
  addWhitelist: (payload) =>
    request("/defense/whitelist", { method: "POST", body: JSON.stringify(payload) }),
  removeWhitelist: (id) => request(`/defense/whitelist/${id}`, { method: "DELETE" }),
};
