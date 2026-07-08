import React from 'react';
import { CircleMarker, MapContainer, Polygon, Popup, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// The parcel and its regulatory surroundings on a real map: parcel boundary
// from King County GIS, street trees from the city's Combined Tree Point layer
// (marker size scales with trunk diameter).

const treeRadius = (dbh) => Math.min(3 + (dbh || 0) * 0.35, 12);

const MapPanel = ({ property, contextData }) => {
  if (!property?.latitude || !property?.longitude) return null;

  const center = [property.latitude, property.longitude];
  const rings = property.parcelRings || [];
  const trees = contextData?.trees?.points || [];

  return (
    <div className="card border-0 shadow-sm mb-4">
      <div className="card-body">
        <h2 className="h6 mb-2">Parcel Map</h2>
        <MapContainer
          center={center}
          zoom={18}
          style={{ height: '340px', width: '100%', borderRadius: '0.5rem' }}
          scrollWheelZoom={false}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {rings.map((ring, i) => (
            <Polygon
              key={i}
              positions={ring.map(([x, y]) => [y, x])}
              pathOptions={{ color: '#0d6efd', weight: 2, fillOpacity: 0.08 }}
            />
          ))}
          {trees.map((tree, i) => (
            <CircleMarker
              key={i}
              center={[tree.lat, tree.lon]}
              radius={treeRadius(tree.dbh_inches)}
              pathOptions={{ color: '#198754', fillColor: '#198754', fillOpacity: 0.5, weight: 1 }}
            >
              <Popup>
                <strong>{tree.common_name || 'Tree'}</strong>
                <br />
                {tree.dbh_inches ? `${tree.dbh_inches}" diameter` : 'diameter unknown'}
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
        <div className="text-muted small mt-2">
          Parcel boundary: King County GIS. Trees: City of Seattle street-tree
          inventory within {contextData?.trees?.radius_m ?? 30} m (circle size ∝ trunk
          diameter). Private-lot trees may not be inventoried.
        </div>
      </div>
    </div>
  );
};

export default MapPanel;
