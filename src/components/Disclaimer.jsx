import React from 'react';
import { Scale } from 'lucide-react';

// Point-of-use disclaimer: regulators discount ToS-only disclosures, so this
// renders with every report, not just in the footer.

export const ReportDisclaimer = () => (
  <div className="d-flex align-items-start gap-2 border rounded-3 p-3 mt-3 bg-body-tertiary">
    <Scale size={16} className="mt-1 text-secondary flex-shrink-0" />
    <div className="small text-secondary">
      PreAssess provides <strong>legal information, not legal advice</strong>, and using
      it creates no attorney-client relationship. Reports are AI-generated and may
      contain errors; every code citation is audited against the ingested code text —
      click any citation to read the code yourself. Final determinations are made by
      the City of Seattle. For decisions with legal or financial consequences, consult
      a licensed professional.
    </div>
  </div>
);

export const SiteFooter = () => (
  <footer className="text-center text-secondary small py-4 mt-4">
    <div className="mb-1">
      PreAssess is an independent research tool and is not affiliated with the City of
      Seattle or King County. Legal information, not legal advice.
    </div>
    <div>
      <a href="/terms.html" className="link-secondary me-3">Terms</a>
      <a href="/privacy.html" className="link-secondary me-3">Privacy</a>
      <a
        href="https://github.com/NawafAlansari/PreAssess"
        className="link-secondary"
        target="_blank"
        rel="noreferrer"
      >
        Source &amp; evals
      </a>
    </div>
  </footer>
);
