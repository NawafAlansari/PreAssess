import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FileText, Info, ShieldCheck, ShieldAlert, ShieldQuestion, X } from 'lucide-react';
import { lookupCitation } from '../lib/api.js';

// Renders the AI compliance report as a document with its citation audit made
// visible: every SMC citation the model used is badged with its verification
// status, and clicking one opens the actual code text in a side drawer.

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
  const cardRef = useRef(null);

  useEffect(() => {
    if (bundle && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [bundle]);

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
  // Defensive: some models emit markdown emphasis despite the plain-text
  // instruction; stray asterisks read as typos in the rendered report.
  const reportText = (bundle.report || '').replace(/\*\*([^*]+)\*\*/g, '$1');
  const segments = segment(reportText, citations);

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
    <div className="card border-0 shadow-sm mb-4" ref={cardRef}>
      <div className="card-body">
        <div className="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-2">
          <div className="d-flex align-items-center gap-2">
            <FileText className="text-primary" size={22} />
            <h2 className="h5 mb-0">Compliance Report</h2>
          </div>
          {ratioBadge(bundle.grounded_ratio ?? 0)}
        </div>
        <div className="text-muted small mb-3">
          Model: <span className="text-monospace">{bundle.model}</span> — every SMC
          citation is checked against the code that was actually retrieved. Click a
          citation to read the code text.
        </div>

        <div className="pa-report mb-3">
          {segments.map((seg, i) =>
            seg.citation ? (
              <button
                key={i}
                type="button"
                onClick={() => onCitationClick(seg.citation)}
                className={`btn btn-sm py-0 pa-citation fw-semibold ${
                  (STATUS_META[auditByCitation[seg.citation]?.status] || STATUS_META.unknown)
                    .className
                }`}
              >
                {seg.text}
              </button>
            ) : (
              <React.Fragment key={i}>{seg.text}</React.Fragment>
            )
          )}
        </div>

        {citations.length > 0 && (
          <div className="d-flex flex-wrap gap-2">
            {Object.entries(STATUS_META).map(([status, meta]) => (
              <span key={status} className={`badge d-flex align-items-center gap-1 ${meta.className}`}>
                <meta.Icon size={14} />
                {meta.label}
              </span>
            ))}
          </div>
        )}

        {selected && (
          <div className="pa-drawer" role="dialog" aria-label={`Code text for SMC ${selected.citation}`}>
            <div className="pa-drawer-head d-flex align-items-start justify-content-between gap-2">
              <div>
                <div className="fw-bold">SMC {selected.citation}</div>
                <div className="text-muted small">
                  {(STATUS_META[selected.verdict?.status] || STATUS_META.unknown).label}
                </div>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => setSelected(null)}
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>
            <div className="pa-drawer-body">
              {lookup.loading && <div className="text-muted small">Looking up code text…</div>}
              {!lookup.loading && lookup.results?.length === 0 && (
                <div className="text-muted small">
                  No matching section found in the ingested code.
                </div>
              )}
              {!lookup.loading &&
                (lookup.results || []).map((hit) => (
                  <div key={hit.chunk_id} className="mb-3">
                    <div className="small fw-semibold mb-1">
                      {hit.citation} {hit.heading ? `— ${hit.heading}` : ''}
                    </div>
                    <div className="pa-code-text">{hit.text}</div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportSection;
