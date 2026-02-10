import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Polygon, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import { useQuery } from '@tanstack/react-query';
import { Icon, LatLngBounds } from 'leaflet';
import { api } from '../lib/api';
import { formatDistanceToNow } from 'date-fns';
import 'leaflet/dist/leaflet.css';

// Zone colors
const ZONE_COLORS: Record<string, string> = {
  protected: '#4CAF50',
  feeding: '#FFC107',
  perimeter: '#2196F3',
  entry_point: '#FF5722',
  exclusion: '#F44336',
  patrol: '#9C27B0',
};

// Predator colors
const PREDATOR_COLORS: Record<string, string> = {
  fox: '#FF6B35',
  badger: '#4A4A4A',
  cat: '#8B4513',
  bird_of_prey: '#4169E1',
  rat: '#696969',
  unknown: '#808080',
};

interface MapState {
  zones: Array<{
    id: string;
    name: string;
    type: string;
    vertices: [number, number][];
    color: string;
  }>;
  cameras: Array<{
    id: string;
    name: string;
    position: [number, number];
    heading: number;
    fov: number;
    range: number;
    coverage?: [number, number][];
  }>;
  tracks: Array<{
    id: string;
    predator_type: string;
    trajectory: [number, number][];
    current_position: [number, number] | null;
    color: string;
  }>;
  bounds: {
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
  };
}

// Custom camera icon
const cameraIcon = new Icon({
  iconUrl: '/camera-icon.svg',
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

// Predator icon
const createPredatorIcon = (color: string) =>
  new Icon({
    iconUrl: `data:image/svg+xml,${encodeURIComponent(`
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="${color}" stroke="white" stroke-width="2"/>
      </svg>
    `)}`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });

function MapBoundsHandler({ bounds }: { bounds: MapState['bounds'] }) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      const leafletBounds = new LatLngBounds(
        [bounds.min_y, bounds.min_x],
        [bounds.max_y, bounds.max_x]
      );
      map.fitBounds(leafletBounds, { padding: [20, 20] });
    }
  }, [bounds, map]);

  return null;
}

export default function MapView() {
  const [showCoverage, setShowCoverage] = useState(true);
  const [showTracks, setShowTracks] = useState(true);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch initial map state
  const { data: mapState, refetch } = useQuery<MapState>({
    queryKey: ['mapState'],
    queryFn: () => api.get('/api/map/state').then((res) => res.data),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'track_update' || data.type === 'detection') {
        refetch();
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [refetch]);

  if (!mapState) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
      </div>
    );
  }

  // Convert coordinates: our system uses meters, Leaflet uses lat/lng
  // For local coordinates, we treat y as latitude and x as longitude
  const toLatLng = (coords: [number, number]): [number, number] => {
    // Simple meter to degree approximation for small areas
    const baseLat = 51.5074; // Configure based on actual location
    const baseLng = -0.1278;
    return [baseLat + coords[1] / 111000, baseLng + coords[0] / 71000];
  };

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-white border-b px-4 py-2 flex items-center gap-4">
        <h1 className="text-xl font-semibold">Garden Map</h1>

        <div className="flex-1" />

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showCoverage}
            onChange={(e) => setShowCoverage(e.target.checked)}
            className="rounded"
          />
          <span className="text-sm">Show Coverage</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showTracks}
            onChange={(e) => setShowTracks(e.target.checked)}
            className="rounded"
          />
          <span className="text-sm">Show Tracks</span>
        </label>

        <div className="text-sm text-gray-500">
          {mapState.tracks.length} active tracks
        </div>
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        <MapContainer
          center={[51.5074, -0.1278]}
          zoom={18}
          className="h-full w-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapBoundsHandler bounds={mapState.bounds} />

          {/* Zones */}
          {mapState.zones.map((zone) => (
            <Polygon
              key={zone.id}
              positions={zone.vertices.map(toLatLng)}
              pathOptions={{
                color: zone.color,
                fillColor: zone.color,
                fillOpacity: selectedZone === zone.id ? 0.5 : 0.2,
                weight: selectedZone === zone.id ? 3 : 1,
              }}
              eventHandlers={{
                click: () => setSelectedZone(zone.id),
              }}
            >
              <Popup>
                <div className="font-semibold">{zone.name}</div>
                <div className="text-sm text-gray-600 capitalize">{zone.type}</div>
              </Popup>
            </Polygon>
          ))}

          {/* Camera coverage */}
          {showCoverage &&
            mapState.cameras.map((camera) =>
              camera.coverage ? (
                <Polygon
                  key={`coverage-${camera.id}`}
                  positions={camera.coverage.map(toLatLng)}
                  pathOptions={{
                    color: '#2196F3',
                    fillColor: '#2196F3',
                    fillOpacity: 0.1,
                    weight: 1,
                    dashArray: '4,4',
                  }}
                />
              ) : null
            )}

          {/* Cameras */}
          {mapState.cameras.map((camera) => (
            <Marker
              key={camera.id}
              position={toLatLng(camera.position)}
              icon={cameraIcon}
            >
              <Popup>
                <div className="font-semibold">{camera.name}</div>
                <div className="text-sm text-gray-600">
                  Heading: {camera.heading}° | FOV: {camera.fov}°
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Predator tracks */}
          {showTracks &&
            mapState.tracks.map((track) => (
              <div key={track.id}>
                {/* Trajectory line */}
                {track.trajectory.length > 1 && (
                  <Polyline
                    positions={track.trajectory.map(toLatLng)}
                    pathOptions={{
                      color: track.color,
                      weight: 2,
                      opacity: 0.7,
                    }}
                  />
                )}

                {/* Current position */}
                {track.current_position && (
                  <Marker
                    position={toLatLng(track.current_position)}
                    icon={createPredatorIcon(track.color)}
                  >
                    <Popup>
                      <div className="font-semibold capitalize">
                        {track.predator_type}
                      </div>
                      <div className="text-sm text-gray-600">
                        Track ID: {track.id}
                      </div>
                    </Popup>
                  </Marker>
                )}
              </div>
            ))}
        </MapContainer>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 z-[1000]">
          <div className="text-sm font-semibold mb-2">Legend</div>
          <div className="space-y-1">
            {Object.entries(ZONE_COLORS).map(([type, color]) => (
              <div key={type} className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: color, opacity: 0.5 }}
                />
                <span className="text-xs capitalize">{type.replace('_', ' ')}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Active tracks panel */}
        {mapState.tracks.length > 0 && (
          <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg p-3 z-[1000] max-w-xs">
            <div className="text-sm font-semibold mb-2">Active Tracks</div>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {mapState.tracks.map((track) => (
                <div
                  key={track.id}
                  className="flex items-center gap-2 text-sm"
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: track.color }}
                  />
                  <span className="capitalize">{track.predator_type}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
