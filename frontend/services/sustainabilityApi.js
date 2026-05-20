const API_BASE = ''; // Use relative path to leverage Vite proxy

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(json.detail || json.message || `Request failed (${res.status})`);
  }
  return json;
}

export async function analyzeSustainability(payload) {
  const result = await request('/api/sustainability/analyze', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  return result.data;
}

export async function fetchSustainabilityHistory(userId, limit = 12) {
  const params = new URLSearchParams({ user_id: userId || 'anonymous', limit: String(limit) });
  const result = await request(`/api/sustainability/history?${params}`);
  return result.data;
}

export async function fetchSustainabilityFormulas() {
  const result = await request('/api/sustainability/formulas');
  return result.data;
}
