import { cleanupOutdatedCaches, precacheAndRoute } from 'workbox-precaching';
import {
  idbRequest,
  idbTxComplete,
  openRoadWatchDB,
  STORE_CACHED_COMPLAINTS,
  STORE_DETECTIONS,
  STORE_PENDING_COMPLAINTS,
  SYNC_TAG,
} from './offline/dbConstants';
import type {
  ComplaintRecord,
  DetectedPotholeCandidate,
  PendingComplaintQueueItem,
} from './types';

interface PrecacheManifestEntry {
  url: string;
  revision: string | null;
}

interface SWExtendableEvent extends Event {
  waitUntil(promise: Promise<unknown>): void;
}

interface SWFetchEvent extends SWExtendableEvent {
  request: Request;
  respondWith(response: Promise<Response> | Response): void;
}

interface SWSyncEvent extends SWExtendableEvent {
  tag: string;
  lastChance: boolean;
}

interface SWClient {
  id: string;
  postMessage(message: unknown): void;
}

interface SWGlobalScope {
  __WB_MANIFEST?: Array<PrecacheManifestEntry | string>;
  location: Location;
  clients: {
    claim(): Promise<void>;
    matchAll(options?: {
      type?: 'window' | 'worker' | 'sharedworker' | 'all';
      includeUncontrolled?: boolean;
    }): Promise<SWClient[]>;
  };
  skipWaiting(): Promise<void>;
  addEventListener(type: 'install', listener: (ev: SWExtendableEvent) => void): void;
  addEventListener(type: 'activate', listener: (ev: SWExtendableEvent) => void): void;
  addEventListener(type: 'fetch', listener: (ev: SWFetchEvent) => void): void;
  addEventListener(type: 'sync', listener: (ev: SWSyncEvent) => void): void;
  addEventListener(
    type: 'message',
    listener: (ev: MessageEvent & SWExtendableEvent) => void
  ): void;
}

const sw = self as unknown as SWGlobalScope;

const RUNTIME_CACHE = 'roadwatch-runtime-v1';
const GEOCODE_CACHE = 'roadwatch-geocode-v1';

// Precache production build assets injected by vite-plugin-pwa
const manifestEntries = (self as unknown as SWGlobalScope).__WB_MANIFEST;
if (Array.isArray(manifestEntries) && manifestEntries.length > 0) {
  cleanupOutdatedCaches();
  precacheAndRoute(manifestEntries);
}

sw.addEventListener('install', (event) => {
  event.waitUntil(sw.skipWaiting());
});

sw.addEventListener('activate', (event) => {
  event.waitUntil(sw.clients.claim());
});

let activeSyncPromise: Promise<{
  syncedCount: number;
  remainingCount: number;
  error?: string;
}> | null = null;

async function broadcastToClients(message: unknown): Promise<void> {
  try {
    const clients = await sw.clients.matchAll({
      type: 'window',
      includeUncontrolled: true,
    });
    for (const client of clients) {
      client.postMessage(message);
    }
  } catch {
    // ignore broadcast failures
  }
}

async function syncPendingComplaintsInWorker(): Promise<{
  syncedCount: number;
  remainingCount: number;
  error?: string;
}> {
  if (activeSyncPromise) return activeSyncPromise;

  activeSyncPromise = (async () => {
    let syncedCount = 0;
    let remainingCount = 0;
    let lastError: string | undefined;

    try {
      const db = await openRoadWatchDB();
      const readTx = db.transaction(STORE_PENDING_COMPLAINTS, 'readonly');
      const allPending = await idbRequest(
        readTx.objectStore(STORE_PENDING_COMPLAINTS).getAll() as IDBRequest<
          PendingComplaintQueueItem[]
        >
      );
      await idbTxComplete(readTx);

      const ordered = allPending.sort(
        (a, b) =>
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
      );

      for (const item of ordered) {
        try {
          const markTx = db.transaction(STORE_PENDING_COMPLAINTS, 'readwrite');
          markTx.objectStore(STORE_PENDING_COMPLAINTS).put({
            ...item,
            syncStatus: 'syncing',
          });
          await idbTxComplete(markTx);

          const res = await fetch('/api/complaints', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              ...item.payload,
              autoFiled: true,
              autoRaise: true,
            }),
          });

          if (!res.ok) {
            const errText = await res.text().catch(() => '');
            throw new Error(
              `HTTP ${res.status}${errText ? `: ${errText.slice(0, 120)}` : ''}`
            );
          }

          const data = (await res.json()) as { complaint: ComplaintRecord };
          const savedComplaint = data.complaint;

          // Update IndexedDB atomically: remove from pending queue, upsert cached_complaints, update detection candidate
          const writeTx = db.transaction(
            [
              STORE_PENDING_COMPLAINTS,
              STORE_CACHED_COMPLAINTS,
              STORE_DETECTIONS,
            ],
            'readwrite'
          );
          writeTx.objectStore(STORE_PENDING_COMPLAINTS).delete(item.queueId);
          writeTx.objectStore(STORE_CACHED_COMPLAINTS).put(savedComplaint);

          if (item.candidateId) {
            const detStore = writeTx.objectStore(STORE_DETECTIONS);
            const existingDet = await idbRequest(
              detStore.get(item.candidateId) as IDBRequest<
                DetectedPotholeCandidate | undefined
              >
            );
            if (existingDet) {
              detStore.put({
                ...existingDet,
                decision: 'raised',
                raisedComplaint: savedComplaint,
                error: undefined,
              });
            }
          }

          await idbTxComplete(writeTx);
          syncedCount++;

          await broadcastToClients({
            type: 'COMPLAINT_SYNCED',
            queueId: item.queueId,
            candidateId: item.candidateId,
            complaint: savedComplaint,
          });
        } catch (err: any) {
          remainingCount++;
          lastError = err?.message || 'Network unavailable while syncing';
          const failTx = db.transaction(STORE_PENDING_COMPLAINTS, 'readwrite');
          failTx.objectStore(STORE_PENDING_COMPLAINTS).put({
            ...item,
            syncStatus: 'failed',
            retryCount: (item.retryCount || 0) + 1,
            lastError,
          });
          await idbTxComplete(failTx);
        }
      }

      db.close();
    } catch (err: any) {
      lastError = err?.message || 'IndexedDB sync error';
    } finally {
      activeSyncPromise = null;
    }

    await broadcastToClients({
      type: 'SYNC_BATCH_COMPLETE',
      syncedCount,
      remainingCount,
      error: lastError,
    });

    return { syncedCount, remainingCount, error: lastError };
  })();

  return activeSyncPromise;
}

// Background Sync API (fires automatically when device regains internet connectivity)
sw.addEventListener('sync', (event) => {
  if (event.tag === SYNC_TAG) {
    event.waitUntil(syncPendingComplaintsInWorker());
  }
});

// Client message triggers (online event / manual "Sync Now" / immediate skipWaiting)
sw.addEventListener('message', (event) => {
  const data = event.data as { type?: string } | undefined;
  if (!data?.type) return;

  if (data.type === 'SKIP_WAITING') {
    event.waitUntil(sw.skipWaiting());
  } else if (data.type === 'TRIGGER_SYNC') {
    event.waitUntil(syncPendingComplaintsInWorker());
  }
});

async function handleGetComplaints(request: Request): Promise<Response> {
  try {
    const networkRes = await fetch(request);
    if (networkRes.ok) {
      const clone = networkRes.clone();
      clone
        .json()
        .then(async (payload: { complaints?: ComplaintRecord[] }) => {
          if (!Array.isArray(payload.complaints)) return;
          const db = await openRoadWatchDB();
          const tx = db.transaction(STORE_CACHED_COMPLAINTS, 'readwrite');
          const store = tx.objectStore(STORE_CACHED_COMPLAINTS);
          store.clear();
          for (const item of payload.complaints) {
            store.put(item);
          }
          await idbTxComplete(tx);
          db.close();
        })
        .catch(() => undefined);
    }
    return networkRes;
  } catch {
    try {
      const db = await openRoadWatchDB();
      const tx = db.transaction(STORE_CACHED_COMPLAINTS, 'readonly');
      const cached = await idbRequest(
        tx.objectStore(STORE_CACHED_COMPLAINTS).getAll() as IDBRequest<
          ComplaintRecord[]
        >
      );
      await idbTxComplete(tx);
      db.close();
      cached.sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );
      return new Response(
        JSON.stringify({ complaints: cached, offlineCached: true }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    } catch {
      return new Response(
        JSON.stringify({ complaints: [], offlineCached: true }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }
}

async function handleGetGeocode(request: Request): Promise<Response> {
  const url = new URL(request.url);
  try {
    const networkRes = await fetch(request);
    if (networkRes.ok) {
      const cache = await caches.open(GEOCODE_CACHE);
      await cache.put(request, networkRes.clone());
    }
    return networkRes;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;

    const lat = Number(url.searchParams.get('lat')) || 0;
    const lng = Number(url.searchParams.get('lng')) || 0;
    return new Response(
      JSON.stringify({
        lat,
        lng,
        formattedAddress: `GPS ${lat.toFixed(5)}, ${lng.toFixed(5)} (Offline)`,
        street: `Roadway at ${lat.toFixed(5)}, ${lng.toFixed(5)}`,
        neighborhood: '',
        city: '',
        state: '',
        postcode: '',
        country: '',
        countryCode: '',
        portal: {
          portalName: 'Municipal Public Works Portal',
          department: 'Department of Transportation & Public Works',
          portalUrl: 'https://www.fixmystreet.com/',
          submissionType: 'Municipal Public Works',
          countryName: 'Offline Capture',
          regionLabel: 'Queued locally in IndexedDB',
          instructions:
            'Location resolved offline via GPS coordinates. Complaint will automatically dispatch on reconnect.',
        },
        offlineFallback: true,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

sw.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET') return;

  // Same-origin API endpoints with IndexedDB / Cache offline fallback
  if (url.origin === sw.location.origin) {
    if (url.pathname === '/api/complaints') {
      event.respondWith(handleGetComplaints(request));
      return;
    }
    if (url.pathname === '/api/geocode') {
      event.respondWith(handleGetGeocode(request));
      return;
    }

    // Navigation & same-origin assets: Network-first with runtime cache fallback
    if (
      request.mode === 'navigate' ||
      url.pathname.startsWith('/assets/') ||
      url.pathname.startsWith('/src/') ||
      url.pathname.endsWith('.png') ||
      url.pathname.endsWith('.svg') ||
      url.pathname.endsWith('.webmanifest')
    ) {
      event.respondWith(
        fetch(request)
          .then(async (res) => {
            if (res.ok) {
              const cache = await caches.open(RUNTIME_CACHE);
              await cache.put(request, res.clone());
            }
            return res;
          })
          .catch(async () => {
            const cached = await caches.match(request);
            if (cached) return cached;
            if (request.mode === 'navigate') {
              const rootCached = await caches.match('/');
              if (rootCached) return rootCached;
            }
            return new Response('Offline', { status: 503 });
          })
      );
      return;
    }
  }

  // OpenStreetMap tile & font runtime cache (stale-while-revalidate)
  if (
    url.hostname.includes('tile.openstreetmap.org') ||
    url.hostname.includes('fonts.googleapis.com') ||
    url.hostname.includes('fonts.gstatic.com')
  ) {
    event.respondWith(
      caches.open(RUNTIME_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        const networkPromise = fetch(request)
          .then((res) => {
            if (res.ok) {
              cache.put(request, res.clone());
            }
            return res;
          })
          .catch(() => undefined);
        return cached || (await networkPromise) || new Response('', { status: 504 });
      })
    );
  }
});
