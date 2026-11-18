import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Crosshair, ExternalLink, MapPin } from 'lucide-react';
import { ComplaintRecord, GeoCoordinates } from '../types';
import { getSeverityMeta } from '../utils/geo';

interface HazardMapProps {
  complaints: ComplaintRecord[];
  currentLocation: GeoCoordinates | null;
  onSelectComplaint: (complaint: ComplaintRecord) => void;
  onOpenScanner: () => void;
}

export function HazardMap({
  complaints,
  currentLocation,
  onSelectComplaint,
  onOpenScanner,
}: HazardMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const userLayerRef = useRef<L.LayerGroup | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(
    complaints[0]?.id || null
  );

  // Initialize Leaflet map once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const initialLat = currentLocation?.lat ?? complaints[0]?.lat ?? 37.7749;
    const initialLng = currentLocation?.lng ?? complaints[0]?.lng ?? -122.4194;
    const initialZoom = currentLocation || complaints.length > 0 ? 14 : 4;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLng],
      zoom: initialZoom,
      zoomControl: true,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(map);

    markersLayerRef.current = L.layerGroup().addTo(map);
    userLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      markersLayerRef.current = null;
      userLayerRef.current = null;
    };
  }, []);

  // Update complaint markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer) return;

    layer.clearLayers();
    complaints.forEach((c) => {
      if (!Number.isFinite(c.lat) || !Number.isFinite(c.lng)) return;
      const meta = getSeverityMeta(c.severity);
      const marker = L.circleMarker([c.lat, c.lng], {
        radius: c.id === selectedId ? 11 : 8,
        fillColor: meta.hex,
        color: '#ffffff',
        weight: 2.5,
        opacity: 1,
        fillOpacity: 0.92,
      });

      marker.bindTooltip(`${c.trackingCode} · ${c.severity}`, {
        direction: 'top',
        offset: [0, -6],
      });

      marker.on('click', () => {
        setSelectedId(c.id);
        onSelectComplaint(c);
      });

      marker.addTo(layer);
    });
  }, [complaints, selectedId, onSelectComplaint]);

  // Update current user GPS marker
  useEffect(() => {
    const layer = userLayerRef.current;
    if (!layer) return;
    layer.clearLayers();

    if (currentLocation) {
      L.circle([currentLocation.lat, currentLocation.lng], {
        radius: Math.min(currentLocation.accuracy || 25, 120),
        fillColor: '#0284c7',
        color: '#0284c7',
        weight: 1,
        opacity: 0.35,
        fillOpacity: 0.12,
      }).addTo(layer);

      L.circleMarker([currentLocation.lat, currentLocation.lng], {
        radius: 7,
        fillColor: '#0284c7',
        color: '#ffffff',
        weight: 2.5,
        opacity: 1,
        fillOpacity: 1,
      })
        .bindTooltip('Your current GPS position')
        .addTo(layer);
    }
  }, [currentLocation]);

  function focusComplaint(c: ComplaintRecord) {
    setSelectedId(c.id);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([c.lat, c.lng], 16, { animate: true });
    }
  }

  function centerOnUser() {
    if (mapInstanceRef.current && currentLocation) {
      mapInstanceRef.current.setView(
        [currentLocation.lat, currentLocation.lng],
        16,
        { animate: true }
      );
    }
  }

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-5 pb-24 md:pb-10">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            Geotagged Hazard Map
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Live GPS position and all detected pothole locations pinned for municipal inspection.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {currentLocation && (
            <button
              type="button"
              onClick={centerOnUser}
              className="min-h-[44px] px-4 py-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-sm font-medium text-slate-800 inline-flex items-center gap-2 cursor-pointer"
            >
              <Crosshair className="w-4 h-4 text-sky-600" />
              Center GPS
            </button>
          )}
          <button
            type="button"
            onClick={onOpenScanner}
            className="min-h-[44px] px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-700 text-white text-sm font-semibold cursor-pointer"
          >
            Scan Road
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Map Canvas */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-2xl overflow-hidden min-h-[380px] lg:min-h-[540px] relative">
          <div ref={mapContainerRef} className="w-full h-full min-h-[380px] lg:min-h-[540px]" />
        </div>

        {/* Mapped Incidents Sidebar */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-2xl flex flex-col overflow-hidden max-h-[540px]">
          <div className="px-4 py-3.5 border-b border-slate-200 flex items-center justify-between">
            <h2 className="font-display text-base font-semibold text-slate-900">
              Pinned Hazards
            </h2>
            <span className="font-mono text-xs text-slate-500">
              {complaints.length} total
            </span>
          </div>

          {complaints.length === 0 ? (
            <div className="p-8 text-center flex-1 flex flex-col items-center justify-center space-y-3">
              <MapPin className="w-7 h-7 text-slate-400" />
              <p className="text-sm text-slate-600">
                No hazards mapped yet. Start recording in the Live Scanner to plot detections automatically.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-slate-200 overflow-y-auto flex-1">
              {complaints.map((c) => {
                const sev = getSeverityMeta(c.severity);
                const active = c.id === selectedId;
                return (
                  <div
                    key={c.id}
                    className={`p-4 transition-colors ${
                      active ? 'bg-amber-50/50' : 'hover:bg-slate-50'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => focusComplaint(c)}
                      className="w-full text-left cursor-pointer"
                    >
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="font-mono font-semibold text-slate-900">
                          {c.trackingCode}
                        </span>
                        <span className={`font-semibold ${sev.textClass}`}>
                          {sev.label}
                        </span>
                      </div>
                      <div className="font-semibold text-sm text-slate-900 mt-1 truncate">
                        {c.title}
                      </div>
                      <div className="text-xs text-slate-500 truncate mt-0.5">
                        {c.address}
                      </div>
                      <div className="font-mono text-[11px] text-slate-500 mt-1">
                        {c.lat.toFixed(5)}, {c.lng.toFixed(5)}
                      </div>
                    </button>

                    <div className="mt-2.5 flex items-center gap-3">
                      <button
                        type="button"
                        onClick={() => onSelectComplaint(c)}
                        className="text-xs font-semibold text-slate-800 hover:text-amber-700 cursor-pointer"
                      >
                        Open Dossier
                      </button>
                      <a
                        href={c.portalUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-medium text-amber-700 hover:text-amber-800 inline-flex items-center gap-1"
                      >
                        Govt Portal
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
