import {
  ComplaintRecord,
  DetectedPotholeCandidate,
  OfflineQueueStats,
  PendingComplaintPayload,
  PendingComplaintQueueItem,
} from '../types';
import {
  idbRequest,
  idbTxComplete,
  openRoadWatchDB,
  STORE_CACHED_COMPLAINTS,
  STORE_DETECTIONS,
  STORE_PENDING_COMPLAINTS,
} from './dbConstants';

const MAX_DETECTIONS_IN_IDB = 60;

// --- Pothole Detection Candidates Store ---

export async function saveDetectionCandidate(
  candidate: DetectedPotholeCandidate
): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_DETECTIONS, 'readwrite');
    const store = tx.objectStore(STORE_DETECTIONS);
    store.put(candidate);

    // Keep store bounded cleanly
    const all = await idbRequest(store.getAll() as IDBRequest<DetectedPotholeCandidate[]>);
    if (all.length > MAX_DETECTIONS_IN_IDB) {
      all.sort(
        (a, b) =>
          new Date(b.detectedAt).getTime() - new Date(a.detectedAt).getTime()
      );
      for (const stale of all.slice(MAX_DETECTIONS_IN_IDB)) {
        if (stale.decision !== 'queued_offline') {
          store.delete(stale.candidateId);
        }
      }
    }

    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal if IndexedDB is restricted
  }
}

export async function getAllDetectionCandidates(): Promise<
  DetectedPotholeCandidate[]
> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_DETECTIONS, 'readonly');
    const items = await idbRequest(
      tx.objectStore(STORE_DETECTIONS).getAll() as IDBRequest<
        DetectedPotholeCandidate[]
      >
    );
    await idbTxComplete(tx);
    db.close();
    return items.sort(
      (a, b) =>
        new Date(b.detectedAt).getTime() - new Date(a.detectedAt).getTime()
    );
  } catch {
    return [];
  }
}

export async function updateDetectionCandidate(
  candidateId: string,
  updates: Partial<DetectedPotholeCandidate>
): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_DETECTIONS, 'readwrite');
    const store = tx.objectStore(STORE_DETECTIONS);
    const existing = await idbRequest(
      store.get(candidateId) as IDBRequest<DetectedPotholeCandidate | undefined>
    );
    if (existing) {
      store.put({ ...existing, ...updates });
    }
    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal
  }
}

export async function clearFinishedDetections(): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_DETECTIONS, 'readwrite');
    const store = tx.objectStore(STORE_DETECTIONS);
    const all = await idbRequest(
      store.getAll() as IDBRequest<DetectedPotholeCandidate[]>
    );
    for (const item of all) {
      if (item.decision === 'raised' || item.decision === 'rejected') {
        store.delete(item.candidateId);
      }
    }
    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal
  }
}

export async function deleteAllDetectionCandidates(): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_DETECTIONS, 'readwrite');
    tx.objectStore(STORE_DETECTIONS).clear();
    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal
  }
}

// --- Pending Offline Complaints Queue Store ---

export async function enqueuePendingComplaint(
  payload: PendingComplaintPayload,
  candidateId?: string
): Promise<PendingComplaintQueueItem> {
  const item: PendingComplaintQueueItem = {
    queueId: `q_${Date.now()}_${Math.floor(1000 + Math.random() * 9000)}`,
    candidateId,
    createdAt: new Date().toISOString(),
    syncStatus: 'queued',
    retryCount: 0,
    payload,
  };

  const db = await openRoadWatchDB();
  const tx = db.transaction(
    [STORE_PENDING_COMPLAINTS, STORE_DETECTIONS],
    'readwrite'
  );
  tx.objectStore(STORE_PENDING_COMPLAINTS).put(item);

  if (candidateId) {
    const detStore = tx.objectStore(STORE_DETECTIONS);
    const existing = await idbRequest(
      detStore.get(candidateId) as IDBRequest<
        DetectedPotholeCandidate | undefined
      >
    );
    if (existing) {
      detStore.put({
        ...existing,
        decision: 'queued_offline',
        queuedOfflineId: item.queueId,
        error: undefined,
      });
    }
  }

  await idbTxComplete(tx);
  db.close();
  return item;
}

export async function getPendingComplaints(): Promise<
  PendingComplaintQueueItem[]
> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_PENDING_COMPLAINTS, 'readonly');
    const items = await idbRequest(
      tx.objectStore(STORE_PENDING_COMPLAINTS).getAll() as IDBRequest<
        PendingComplaintQueueItem[]
      >
    );
    await idbTxComplete(tx);
    db.close();
    return items.sort(
      (a, b) =>
        new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
    );
  } catch {
    return [];
  }
}

export async function updatePendingComplaint(
  queueId: string,
  updates: Partial<PendingComplaintQueueItem>
): Promise<void> {
  const db = await openRoadWatchDB();
  const tx = db.transaction(STORE_PENDING_COMPLAINTS, 'readwrite');
  const store = tx.objectStore(STORE_PENDING_COMPLAINTS);
  const existing = await idbRequest(
    store.get(queueId) as IDBRequest<PendingComplaintQueueItem | undefined>
  );
  if (existing) {
    store.put({ ...existing, ...updates });
  }
  await idbTxComplete(tx);
  db.close();
}

export async function removePendingComplaint(queueId: string): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_PENDING_COMPLAINTS, 'readwrite');
    tx.objectStore(STORE_PENDING_COMPLAINTS).delete(queueId);
    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal
  }
}

// --- Cached Server Complaints Store (for offline read access) ---

export async function cacheServerComplaints(
  complaints: ComplaintRecord[]
): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_CACHED_COMPLAINTS, 'readwrite');
    const store = tx.objectStore(STORE_CACHED_COMPLAINTS);
    store.clear();
    for (const c of complaints) {
      store.put(c);
    }
    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal
  }
}

export async function upsertCachedComplaint(
  complaint: ComplaintRecord
): Promise<void> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_CACHED_COMPLAINTS, 'readwrite');
    tx.objectStore(STORE_CACHED_COMPLAINTS).put(complaint);
    await idbTxComplete(tx);
    db.close();
  } catch {
    // Non-fatal
  }
}

export async function getCachedComplaints(): Promise<ComplaintRecord[]> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(STORE_CACHED_COMPLAINTS, 'readonly');
    const items = await idbRequest(
      tx.objectStore(STORE_CACHED_COMPLAINTS).getAll() as IDBRequest<
        ComplaintRecord[]
      >
    );
    await idbTxComplete(tx);
    db.close();
    return items.sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
  } catch {
    return [];
  }
}

export async function getOfflineQueueStats(): Promise<OfflineQueueStats> {
  try {
    const db = await openRoadWatchDB();
    const tx = db.transaction(
      [
        STORE_DETECTIONS,
        STORE_PENDING_COMPLAINTS,
        STORE_CACHED_COMPLAINTS,
      ],
      'readonly'
    );
    const [detectionsCount, pendingComplaintsCount, cachedComplaintsCount] =
      await Promise.all([
        idbRequest(tx.objectStore(STORE_DETECTIONS).count()),
        idbRequest(tx.objectStore(STORE_PENDING_COMPLAINTS).count()),
        idbRequest(tx.objectStore(STORE_CACHED_COMPLAINTS).count()),
      ]);
    await idbTxComplete(tx);
    db.close();
    return {
      detectionsCount,
      pendingComplaintsCount,
      cachedComplaintsCount,
    };
  } catch {
    return {
      detectionsCount: 0,
      pendingComplaintsCount: 0,
      cachedComplaintsCount: 0,
    };
  }
}

export function pendingQueueItemToComplaintRecord(
  item: PendingComplaintQueueItem
): ComplaintRecord {
  const p = item.payload;
  return {
    id: item.queueId,
    trackingCode: 'OFFLINE-QUEUE',
    createdAt: item.createdAt,
    updatedAt: item.createdAt,
    title: p.title,
    description: p.description,
    severity: p.severity,
    confidence: p.confidence,
    roadSurfaceType: p.roadSurfaceType,
    estimatedDepthCm: p.estimatedDepthCm,
    depthClass: p.depthClass,
    depthEstimationDetails: p.depthEstimationDetails || null,
    lat: p.lat,
    lng: p.lng,
    accuracyMeters: p.accuracyMeters,
    address: p.address,
    city: p.city,
    state: p.state,
    countryCode: p.countryCode,
    department: p.department,
    portalName: p.portalName,
    portalUrl: p.portalUrl,
    status: 'Queued for Govt Portal',
    externalTicketId: '',
    dispatchChannel: `${p.portalName || 'Municipal Portal'} (Offline IndexedDB Queue)`,
    upstreamHttpStatus: null,
    upstreamResponseNote:
      item.lastError ||
      'Stored in IndexedDB offline queue. Will automatically submit when internet connectivity resumes.',
    autoFiled: true,
    imageDataUrl: p.imageDataUrl,
    detections: p.detections || [],
    isClusterReport: p.isClusterReport,
    clusterCount: p.clusterCount,
    clusterRadiusMeters: p.clusterRadiusMeters,
    offlineQueued: true,
    queueId: item.queueId,
  };
}
