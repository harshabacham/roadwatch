import { useCallback, useEffect, useRef, useState } from 'react';
import { TopNav, ActiveTab } from './components/TopNav';
import { LiveScanner } from './components/LiveScanner';
import { KaggleDatasetExplorer } from './components/KaggleDatasetExplorer';
import { HazardMap } from './components/HazardMap';
import { ComplaintLog } from './components/ComplaintLog';
import { PortalSettings } from './components/PortalSettings';
import { MLTrainingPanel } from './components/MLTrainingPanel';
import { ComplaintDossierModal } from './components/ComplaintDossierModal';
import { OfflineSyncBar } from './components/OfflineSyncBar';
import { useOnlineStatus } from './hooks/useOnlineStatus';
import { setTrainedClassifierModel } from './ml/unsupervisedDetector';
import {
  cacheServerComplaints,
  getCachedComplaints,
  getOfflineQueueStats,
  getPendingComplaints,
  pendingQueueItemToComplaintRecord,
  removePendingComplaint,
} from './offline/indexedDb';
import {
  initOfflineSync,
  subscribeToOfflineSync,
  triggerOfflineSync,
} from './offline/syncManager';
import {
  ComplaintRecord,
  GeoCoordinates,
  GeocodedLocation,
  OfflineQueueStats,
  Open311Config,
  TrainedPotholeModel,
} from './types';
import { haversineMeters } from './utils/geo';

const OPEN311_STORAGE_KEY = 'roadwatch_open311_config_v1';

const DEFAULT_OPEN311_CONFIG: Open311Config = {
  enabled: false,
  mode: 'open311',
  endpointUrl: '',
  serviceCode: 'pothole',
  apiKey: '',
  jurisdictionId: '',
  reporterEmail: '',
  reporterName: '',
};

function loadSavedOpen311Config(): Open311Config {
  try {
    const raw = localStorage.getItem(OPEN311_STORAGE_KEY);
    if (!raw) return DEFAULT_OPEN311_CONFIG;
    return { ...DEFAULT_OPEN311_CONFIG, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_OPEN311_CONFIG;
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('scanner');
  const [externalScannerImage, setExternalScannerImage] = useState<{
    url: string;
    label: string;
  } | null>(null);
  const [complaints, setComplaints] = useState<ComplaintRecord[]>([]);
  const [loadingComplaints, setLoadingComplaints] = useState(true);
  const [selectedComplaint, setSelectedComplaint] =
    useState<ComplaintRecord | null>(null);

  const [currentLocation, setCurrentLocation] = useState<GeoCoordinates | null>(
    null
  );
  const [geocoded, setGeocoded] = useState<GeocodedLocation | null>(null);
  const [gpsStatus, setGpsStatus] = useState<
    'acquiring' | 'locked' | 'denied' | 'unavailable'
  >('acquiring');
  const [gpsError, setGpsError] = useState<string | null>(null);
  const [open311Config, setOpen311Config] = useState<Open311Config>(
    loadSavedOpen311Config
  );

  const isOnline = useOnlineStatus();
  const [queueStats, setQueueStats] = useState<OfflineQueueStats>({
    detectionsCount: 0,
    pendingComplaintsCount: 0,
    cachedComplaintsCount: 0,
  });
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncNotice, setLastSyncNotice] = useState<string | null>(null);
  const [trainedModel, setTrainedModel] = useState<TrainedPotholeModel | null>(
    null
  );

  const lastGeocodedCoordsRef = useRef<{ lat: number; lng: number } | null>(
    null
  );
  const watchIdRef = useRef<number | null>(null);

  const refreshQueueStats = useCallback(async () => {
    const stats = await getOfflineQueueStats();
    setQueueStats(stats);
  }, []);

  const fetchGeocodedAddress = useCallback(async (lat: number, lng: number) => {
    const prev = lastGeocodedCoordsRef.current;
    if (prev && haversineMeters(prev.lat, prev.lng, lat, lng) < 20) {
      return;
    }
    lastGeocodedCoordsRef.current = { lat, lng };
    try {
      const res = await fetch(
        `/api/geocode?lat=${encodeURIComponent(lat)}&lng=${encodeURIComponent(lng)}`
      );
      if (res.ok) {
        const data: GeocodedLocation = await res.json();
        setGeocoded(data);
      }
    } catch {
      // Keep raw coordinates when offline
    }
  }, []);

  const startGpsWatch = useCallback(() => {
    if (!navigator.geolocation) {
      setGpsStatus('unavailable');
      setGpsError('Geolocation is not supported by this browser.');
      return;
    }

    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }

    setGpsStatus('acquiring');
    setGpsError(null);

    watchIdRef.current = navigator.geolocation.watchPosition(
      (pos) => {
        const coords: GeoCoordinates = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
          speed: pos.coords.speed,
          heading: pos.coords.heading,
          timestamp: pos.timestamp,
          source: 'gps',
        };
        setCurrentLocation(coords);
        setGpsStatus('locked');
        setGpsError(null);
        fetchGeocodedAddress(coords.lat, coords.lng);
      },
      (err) => {
        if (err.code === err.PERMISSION_DENIED) {
          setGpsStatus('denied');
          setGpsError(
            'GPS location permission denied. Allow location access or tap Edit GPS.'
          );
        } else {
          setGpsStatus('unavailable');
          setGpsError(err.message || 'Unable to acquire GPS fix.');
        }
      },
      {
        enableHighAccuracy: true,
        maximumAge: 5000,
        timeout: 15000,
      }
    );
  }, [fetchGeocodedAddress]);

  const loadComplaintsAndQueue = useCallback(async () => {
    const [cached, pending] = await Promise.all([
      getCachedComplaints(),
      getPendingComplaints(),
    ]);
    const queuedRecords = pending.map(pendingQueueItemToComplaintRecord);

    if (cached.length > 0 || queuedRecords.length > 0) {
      setComplaints([...queuedRecords, ...cached]);
    }

    try {
      const res = await fetch('/api/complaints');
      if (res.ok) {
        const data = await res.json();
        const serverList: ComplaintRecord[] = Array.isArray(data.complaints)
          ? data.complaints
          : [];
        if (!data.offlineCached) {
          await cacheServerComplaints(serverList);
        }
        const latestPending = await getPendingComplaints();
        const latestQueuedRecords = latestPending.map(
          pendingQueueItemToComplaintRecord
        );
        setComplaints([...latestQueuedRecords, ...serverList]);
      }
    } catch {
      // Offline: retain cached + IndexedDB queued records
    } finally {
      setLoadingComplaints(false);
      await refreshQueueStats();
    }
  }, [refreshQueueStats]);

  // Initialize Service Worker, GPS, trained ML model, and IndexedDB-backed complaints on mount
  useEffect(() => {
    initOfflineSync();
    startGpsWatch();
    void loadComplaintsAndQueue();
    void (async () => {
      try {
        const res = await fetch('/api/ml/model');
        if (res.ok) {
          const data = await res.json();
          if (data?.model) {
            setTrainedModel(data.model);
            setTrainedClassifierModel(data.model);
          }
        }
      } catch {
        // Offline fallback uses unsupervised detector defaults
      }
    })();

    const unsubscribe = subscribeToOfflineSync((event) => {
      if (event.type === 'QUEUE_UPDATED') {
        void refreshQueueStats();
      } else if (event.type === 'COMPLAINT_SYNCED') {
        setComplaints((prev) => {
          const withoutQueued = prev.filter(
            (c) => c.id !== event.queueId && c.id !== event.complaint.id
          );
          return [event.complaint, ...withoutQueued];
        });
        void refreshQueueStats();
      } else if (event.type === 'SYNC_BATCH_COMPLETE') {
        setIsSyncing(false);
        void refreshQueueStats();
        if (event.syncedCount > 0) {
          setLastSyncNotice(
            `Synced ${event.syncedCount} offline complaint${
              event.syncedCount === 1 ? '' : 's'
            } automatically to the government portal.`
          );
          window.setTimeout(() => {
            setLastSyncNotice(null);
          }, 6000);
        }
      }
    });

    return () => {
      unsubscribe();
      if (watchIdRef.current !== null) {
        navigator.geolocation.clearWatch(watchIdRef.current);
      }
    };
  }, [startGpsWatch, loadComplaintsAndQueue, refreshQueueStats]);

  // Auto-trigger IndexedDB queue sync whenever connectivity returns
  useEffect(() => {
    if (isOnline && queueStats.pendingComplaintsCount > 0) {
      setIsSyncing(true);
      void triggerOfflineSync();
    }
  }, [isOnline, queueStats.pendingComplaintsCount]);

  function handleManualSyncNow() {
    if (!isOnline) return;
    setIsSyncing(true);
    void triggerOfflineSync();
  }

  function handleSetManualLocation(lat: number, lng: number) {
    lastGeocodedCoordsRef.current = null;
    setCurrentLocation({
      lat,
      lng,
      accuracy: 5,
      speed: null,
      heading: null,
      timestamp: Date.now(),
      source: 'manual',
    });
    setGpsStatus('locked');
    setGpsError(null);
    fetchGeocodedAddress(lat, lng);
  }

  function handleSaveOpen311Config(next: Open311Config) {
    setOpen311Config(next);
    try {
      localStorage.setItem(OPEN311_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage quota errors
    }
  }

  function handleComplaintCreated(complaint: ComplaintRecord) {
    setComplaints((prev) => {
      const deduped = prev.filter((c) => c.id !== complaint.id);
      return [complaint, ...deduped];
    });
    void refreshQueueStats();
  }

  async function handleUpdateComplaint(
    id: string,
    updates: Partial<Pick<ComplaintRecord, 'status' | 'externalTicketId'>>
  ) {
    if (id.startsWith('q_')) return;
    const res = await fetch(`/api/complaints/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (res.ok) {
      const data = await res.json();
      setComplaints((prev) =>
        prev.map((c) => (c.id === id ? data.complaint : c))
      );
      setSelectedComplaint(data.complaint);
    }
  }

  async function handleDeleteComplaint(id: string) {
    if (id.startsWith('q_')) {
      await removePendingComplaint(id);
      setComplaints((prev) => prev.filter((c) => c.id !== id));
      setSelectedComplaint(null);
      await refreshQueueStats();
      return;
    }

    const res = await fetch(`/api/complaints/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (res.ok) {
      setComplaints((prev) => prev.filter((c) => c.id !== id));
      setSelectedComplaint(null);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <TopNav
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        complaintCount={complaints.length}
      />

      <OfflineSyncBar
        isOnline={isOnline}
        pendingCount={queueStats.pendingComplaintsCount}
        savedDetectionsCount={queueStats.detectionsCount}
        isSyncing={isSyncing}
        lastSyncNotice={lastSyncNotice}
        onSyncNow={handleManualSyncNow}
      />

      <main className="flex-1">
        {activeTab === 'scanner' && (
          <LiveScanner
            currentLocation={currentLocation}
            geocoded={geocoded}
            gpsStatus={gpsStatus}
            gpsError={gpsError}
            onRetryGps={startGpsWatch}
            onSetManualLocation={handleSetManualLocation}
            open311Config={open311Config}
            onComplaintCreated={handleComplaintCreated}
            onOpenComplaint={(c) => setSelectedComplaint(c)}
            onDetectionsChanged={() => void refreshQueueStats()}
            externalImage={externalScannerImage}
          />
        )}

        {activeTab === 'dataset' && (
          <KaggleDatasetExplorer
            onOpenInScanner={(url, label) => {
              setExternalScannerImage({ url, label });
              setActiveTab('scanner');
            }}
          />
        )}

        {activeTab === 'map' && (
          <HazardMap
            complaints={complaints}
            currentLocation={currentLocation}
            onSelectComplaint={(c) => setSelectedComplaint(c)}
            onOpenScanner={() => setActiveTab('scanner')}
          />
        )}

        {activeTab === 'complaints' && (
          <ComplaintLog
            complaints={complaints}
            loading={loadingComplaints}
            onSelectComplaint={(c) => setSelectedComplaint(c)}
            onOpenScanner={() => setActiveTab('scanner')}
          />
        )}

        {activeTab === 'training' && (
          <MLTrainingPanel
            model={trainedModel}
            onModelUpdated={(nextModel) => {
              setTrainedModel(nextModel);
              setTrainedClassifierModel(nextModel);
            }}
          />
        )}

        {activeTab === 'portals' && (
          <PortalSettings
            config={open311Config}
            onSaveConfig={handleSaveOpen311Config}
          />
        )}
      </main>

      {selectedComplaint && (
        <ComplaintDossierModal
          complaint={selectedComplaint}
          onClose={() => setSelectedComplaint(null)}
          onUpdate={handleUpdateComplaint}
          onDelete={handleDeleteComplaint}
        />
      )}
    </div>
  );
}
