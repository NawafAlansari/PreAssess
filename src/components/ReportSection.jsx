import React, { useMemo, useState } from 'react';
import { FileText, Info, ShieldCheck, ShieldAlert, ShieldQuestion } from 'lucide-react';
import { lookupCitation } from '../lib/api.js';

// Renders the AI compliance report with its citation audit made visible:
// every SMC citation the model used is badged with its verification status,
// and clicking one shows the actual code text that grounds (or fails to
// ground) it.

const STATUS_META = {
  grounded: {
    label: 'verified in evidence',
    className: 'bg-success-subtle text-success-emphasis border border-success-subtle',
    Icon: ShieldCheck
  },
  in_corpus_not_retrieved: {
    label: 'in code, not in evidence shown to the model',
    className: 'bg-warning-subtle text-warning-emphasis border border-warning-subtle',
    Icon: ShieldAlert
  },
  unknown: {
    label: 'not found in the ingested code',
    className: 'bg-danger-subtle text-danger-emphasis border border-danger-subtle',
    Icon: ShieldQuestion
  }
};

function ratioBadge(ratio) {
  const pct = Math.round(ratio * 100);
  const tone = ratio >= 0.8 ? 'success' : ratio >= 0.5 ? 'warning' : 'danger';
  return (
    <span className={`badge bg-${tone}-subtle text-${tone}-emphasis`}>
      {pct}% of citations verified against retrieved code
    </span>
  );
}

// Split the report text into plain segments and citation segments, matching
// the exact citation strings the audit found (longest first so subsection
// citations win over their parents).
function segment(text, citations) {
  if (!citations.length) return [{ text }];
  const sorted = [...citations].sort((a, b) => b.length - a.length);
  const pattern = sorted
    .map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|');
  const re = new RegExp(`(${pattern})`, 'g');
  return text.split(re).map((part) => ({
    text: part,
    citation: citations.includes(part) ? part : null
  }));
}

const ReportSection = ({ bundle, error }) => {
  const [selected, setSelected] = useState(null);
  const [lookup, setLookup] = useState({ loading: false, results: null });

  const auditByCitation = useMemo(() => {
    const map = {};
    (bundle?.citation_audit || []).forEach((v) => {
      map[v.citation] = v;
    });
    return map;
  }, [bundle]);

  const evidenceByChunk = useMemo(() => {
    const map = {};
    Object.values(bundle?.evidence || {}).forEach((hits) => {
      hits.forEach((hit) => {
        map[hit.chunk_id] = hit;
      });
    });
    return map;
  }, [bundle]);

  if (error) {
    return (
      <div className="alert alert-warning d-flex align-items-start gap-2 mb-4" role="alert">
        <Info size={18} className="mt-1" />
        <div>
          <div className="fw-semibold">AI report not available</div>
          <div className="small">{error}</div>
        </div>
      </div>
    );
  }
  if (!bundle) return null;

  const citations = Object.keys(auditByCitation);
  const segments = segment(bundle.report || '', citations);

  const onCitationClick = async (citation) => {
    const verdict = auditByCitation[citation];
    setSelected({ citation, verdict });
    const grounding = (verdict?.matched_chunk_ids || [])
      .map((id) => evidenceByChunk[id])
      .filter(Boolean);
    if (grounding.length) {
      setLookup({ loading: false, results: grounding });
      return;
    }
    setLookup({ loading: true, results: null });
    try {
      const body = await lookupCitation(citation);
      setLookup({ loading: false, results: body.results });
    } catch {
      setLookup({ loading: false, results: [] });
    }
  };

  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body">
        <div className="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-2">
          <div className="d-flex align-items-center gap-2">
            <FileText className="text-primary" size={22} />
            <h2 className="h5 mb-0">AI Compliance Report</h2>
          </div>
          {ratioBadge(bundle.grounded_ratio ?? 0)}
        </div>
        <div className="text-muted small mb-3">
          Model: <span className="text-monospace">{bundle.model}</span> — every SMC
          citation below is checked against the code that was actually retrieved.
          Click a citation to see the code text.
        </div>

        <div className="border rounded-3 bg-body-tertiary p-3 mb-3">
          <pre className="mb-0 small text-body-emphasis" style={{ whiteSpace: 'pre-wrap' }}>
            {segments.map((seg, i) =>
              seg.citation ? (
                <button
                  key={i}
                  type="button"
                  onClick={() => onCitationClick(seg.citation)}
                  className={`btn btn-sm py-0 px-1 fw-semibold ${
                    (STATUS_META[auditByCitation[seg.citation]?.status] || STATUS_META.unknown)
                      .className
                  }`}
                  style={{ fontSize: 'inherit' }}
                >
                  {seg.text}
                </button>
              ) : (
                <React.Fragment key={i}>{seg.text}</React.Fragment>
              )
            )}
          </pre>
        </div>

        {citations.length > 0 && (
          <div className="d-flex flex-wrap gap-2 mb-3">
            {Object.entries(STATUS_META).map(([status, meta]) => (
              <span key={status} className={`badge d-flex align-items-center gap-1 ${meta.className}`}>
                <meta.Icon size={14} />
                {meta.label}
              </span>
            ))}
          </div>
        )}

        {selected && (
          <div className="border rounded-3 p-3">
            <div className="d-flex align-items-center justify-content-between mb-2">
              <div className="fw-semibold">
                SMC {selected.citation}
                <span className="text-muted small ms-2">
                  {(STATUS_META[selected.verdict?.status] || STATUS_META.unknown).label}
                </span>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => setSelected(null)}
              >
                Close
              </button>
            </div>
            {lookup.loading && <div className="text-muted small">Looking up code text…</div>}
            {!lookup.loading && lookup.results?.length === 0 && (
              <div className="text-muted small">
                No matching section found in the ingested code (Titles 22 and 23).
              </div>
            )}
            {!lookup.loading &&
              (lookup.results || []).map((hit) => (
                <div key={hit.chunk_id} className="mb-2">
                  <div className="small fw-semibold">
                    {hit.citation} {hit.heading ? `— ${hit.heading}` : ''}
                  </div>
                  <div className="small text-muted" style={{ whiteSpace: 'pre-wrap' }}>
                    {hit.text}
                  </div>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportSection;
