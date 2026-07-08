import React from 'react';
import { Landmark, Layers, TreePine, TriangleAlert } from 'lucide-react';

// What the city's GIS says regulates this parcel: overlay districts (each
// linked to its SMC chapter), environmentally critical area flags, and the
// street-tree picture. These facts feed the AI report automatically.

const ECA_LABELS = {
  flood_prone: 'Flood-prone area',
  known_slide: 'Known landslide area',
  liquefaction_prone: 'Liquefaction-prone soil',
  peat_settlement: 'Peat settlement area',
  potential_slide: 'Potential slide area',
  riparian_corridor: 'Riparian corridor',
  steep_slope: 'Steep slope',
  wetland: 'Wetland',
  wildlife_habitat: 'Wildlife habitat',
};

const ContextPanel = ({ contextData }) => {
  if (!contextData) return null;
  const { zoning, overlays = [], eca = [], trees } = contextData;

  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body">
        <div className="d-flex align-items-center gap-2 mb-3">
          <Layers className="text-primary" size={20} />
          <h2 className="h6 mb-0">Regulatory Context (City of Seattle GIS)</h2>
        </div>

        <div className="d-flex flex-wrap gap-2 mb-2">
          {zoning?.zone && (
            <span className="badge bg-primary-subtle text-primary-emphasis d-flex align-items-center gap-1">
              <Landmark size={14} /> Zone: {zoning.zone}
            </span>
          )}
          {overlays.map((o, i) => (
            <span
              key={i}
              className="badge bg-info-subtle text-info-emphasis d-flex align-items-center gap-1"
              title={o.about || ''}
            >
              {o.name} ({o.type?.toLowerCase()})
              {o.chapter && <span className="fw-normal">— SMC {o.chapter.replace('Chapter ', '')}</span>}
            </span>
          ))}
          {overlays.length === 0 && (
            <span className="badge bg-secondary-subtle text-secondary-emphasis">
              No overlay districts at this location
            </span>
          )}
        </div>

        {eca.length > 0 && (
          <div className="d-flex flex-wrap gap-2 mb-2">
            {eca.map((flag) => (
              <span
                key={flag}
                className="badge bg-warning-subtle text-warning-emphasis d-flex align-items-center gap-1"
              >
                <TriangleAlert size={14} /> {ECA_LABELS[flag] || flag}
              </span>
            ))}
          </div>
        )}

        {trees && (
          <div className="d-flex align-items-center gap-2 text-muted small">
            <TreePine size={16} className="text-success" />
            {trees.count} inventoried street trees within {trees.radius_m} m
            {trees.largest?.[0]?.dbh_inches
              ? ` — largest: ${trees.largest[0].common_name} (${trees.largest[0].dbh_inches}" diameter)`
              : ''}
          </div>
        )}
      </div>
    </div>
  );
};

export default ContextPanel;
