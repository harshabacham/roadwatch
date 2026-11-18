export const ROADWATCH_DB_NAME = 'roadwatch-offline-db';
export const ROADWATCH_DB_VERSION = 1;

export const STORE_DETECTIONS = 'detections';
export const STORE_PENDING_COMPLAINTS = 'pending_complaints';
export const STORE_CACHED_COMPLAINTS = 'cached_complaints';

export const SYNC_TAG = 'roadwatch-sync-complaints';

/**
 * Opens (and initializes/upgrades) the RoadWatch IndexedDB database.
 * Works identically inside the main Window context and ServiceWorkerGlobalScope.
 */
export function openRoadWatchDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const idbFactory =
      typeof self !== 'undefined' && 'indexedDB' in self
        ? self.indexedDB
        : typeof indexedDB !== 'undefined'
        ? indexedDB
        : null;

    if (!idbFactory) {
      reject(new Error('IndexedDB is unavailable in this browser context.'));
      return;
    }

    const request = idbFactory.open(ROADWATCH_DB_NAME, ROADWATCH_DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;

      if (!db.objectStoreNames.contains(STORE_DETECTIONS)) {
        const detStore = db.createObjectStore(STORE_DETECTIONS, {
          keyPath: 'candidateId',
        });
        detStore.createIndex('detectedAt', 'detectedAt', { unique: false });
        detStore.createIndex('decision', 'decision', { unique: false });
      }

      if (!db.objectStoreNames.contains(STORE_PENDING_COMPLAINTS)) {
        const pendingStore = db.createObjectStore(STORE_PENDING_COMPLAINTS, {
          keyPath: 'queueId',
        });
        pendingStore.createIndex('createdAt', 'createdAt', { unique: false });
        pendingStore.createIndex('syncStatus', 'syncStatus', { unique: false });
      }

      if (!db.objectStoreNames.contains(STORE_CACHED_COMPLAINTS)) {
        const cacheStore = db.createObjectStore(STORE_CACHED_COMPLAINTS, {
          keyPath: 'id',
        });
        cacheStore.createIndex('createdAt', 'createdAt', { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error || new Error('Failed to open IndexedDB.'));
  });
}

export function idbRequest<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB request failed.'));
  });
}

export function idbTxComplete(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('IndexedDB transaction failed.'));
    tx.onabort = () => reject(tx.error || new Error('IndexedDB transaction aborted.'));
  });
}
