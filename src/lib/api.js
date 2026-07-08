// Client for the PreAssess API. All LLM access happens server-side; the
// browser never handles a model API key.

const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body?.detail || `Request failed with status ${response.status}`);
  }
  return body;
}

const summarizeProperty = (property) => ({
  address: property.address,
  parcelNumber: property.parcelNumber,
  lotSizeSqFt: property.lotSizeSqFt,
  lotSizeAcres: property.lotSizeAcres,
  propertyType: property.propertyType,
  presentUse: property.presentUse,
  zoneClassification: property.zoneClassification,
  zoneDescription: property.zoneDescription,
  jurisdiction: property.jurisdiction,
  neighborhood: property.neighborhood,
  latitude: property.latitude,
  longitude: property.longitude
});

export function fetchContext(lat, lon) {
  const params = new URLSearchParams({ lat: String(lat), lon: String(lon) });
  return request(`/api/context?${params}`);
}

export function fetchComplianceReport(property, analysis, description, context) {
  const questions = [];
  if (analysis?.projectTypes?.length) {
    questions.push(
      `What Seattle Municipal Code requirements apply to: ${analysis.projectTypes.join(', ')}?`
    );
  }
  return request('/api/report', {
    method: 'POST',
    body: JSON.stringify({
      address_profile: summarizeProperty(property),
      project_description: description || '',
      context: context || undefined,
      questions
    })
  });
}

export function searchCode(query, k = 5) {
  const params = new URLSearchParams({ q: query, k: String(k) });
  return request(`/api/search?${params}`);
}

export function fetchHealth() {
  return request('/api/health');
}

export function lookupCitation(citation) {
  return request(`/api/citation/${encodeURIComponent(citation)}`);
}
