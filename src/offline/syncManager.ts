import { registerSW } from 'virtual:pwa-register';
import {
  ComplaintRecord,
  PendingComplaintPayload,
  PendingComplaintQueueItem,
} from '../types';
import { SYNC_TAG } from './dbConstants';
import {
  enqueuePendingComplaint,
  getPendingComplaints,
  removePendingComplaint,
  updateDetectionCandidate,
  updatePendingComplaint,
  upsertCachedComplaint,
} from './indexedDb';

export type OfflineSyncEvent =
  | {
      type: 'COMPLAINT_SYNCED';
      queueId: string;
      candidateId?: string;
      complaint: ComplaintRecord;
    }
  | {
      type: 'SYNC_BATCH_COMPLETE';
      syncedCount: number;
      remainingCount: number;
      error?: string;
    }
  | {
      type: 'QUEUE_UPDATED';
    };

type SyncListener = (event: OfflineSyncEvent) => void;

const listeners = new Set<SyncListener>();
let initialized = false;
let inPageSyncPromise: Promise<{ syncedCount: number; remainingCount: number }> | null =
  null;
let swRegistration: ServiceWorkerRegistration | null = null;

function emit(event: OfflineSyncEvent) {
  listeners.forEach((fn) => {
    try {
      fn(event);
    } catch {
      // ignore listener errors
    }
  });
}

export function subscribeToOfflineSync(listener: SyncListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Fallback queue processor used when the Service Worker controller is not yet active
 * (e.g., very first page load before claim, or browsers without Background Sync).
 */
async function flushPendingComplaintsInPage(): Promise<{
  syncedCount: number;
  remainingCount: number;
}> {
  if (inPageSyncPromise) return inPageSyncPromise;

  inPageSyncPromise = (async () => {
    let syncedCount = 0;
    let remainingCount = 0;
    let lastError: string | undefined;

    try {
      const pending = await getPendingComplaints();
      for (const item of pending) {
        if (!navigator.onLine) {
          remainingCount++;
          continue;
        }
        try {
          await updatePendingComplaint(item.queueId, { syncStatus: 'syncing' });
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
            throw new Error(`HTTP ${res.status}`);
          }
          const data = (await res.json()) as { complaint: ComplaintRecord };
          const saved = data.complaint;

          await removePendingComplaint(item.queueId);
          await upsertCachedComplaint(saved);
          if (item.candidateId) {
            await updateDetectionCandidate(item.candidateId, {
              decision: 'raised',
              raisedComplaint: saved,
              error: undefined,
            });
          }

          syncedCount++;
          emit({
            type: 'COMPLAINT_SYNCED',
            queueId: item.queueId,
            candidateId: item.candidateId,
            complaint: saved,
          });
        } catch (err: any) {
          remainingCount++;
          lastError = err?.message || 'Failed to sync offline complaint';
          await updatePendingComplaint(item.queueId, {
            syncStatus: 'failed',
            retryCount: (item.retryCount || 0) + 1,
            lastError,
          });
        }
      }
    } finally {
      inPageSyncPromise = null;
    }

    emit({
      type: 'SYNC_BATCH_COMPLETE',
      syncedCount,
      remainingCount,
      error: lastError,
    });

    return { syncedCount, remainingCount };
  })();

  return inPageSyncPromise;
}

export async function triggerOfflineSync(): Promise<void> {
  if (typeof navigator !== 'undefined' && !navigator.onLine) return;

  const pending = await getPendingComplaints();
  if (pending.length === 0) return;

  if (
    typeof navigator !== 'undefined' &&
    'serviceWorker' in navigator &&
    navigator.serviceWorker.controller
  ) {
    navigator.serviceWorker.controller.postMessage({ type: 'TRIGGER_SYNC' });
    return;
  }

  await flushPendingComplaintsInPage();
}

export async function queueComplaintForOfflineSync(
  payload: PendingComplaintPayload,
  candidateId?: string
): Promise<PendingComplaintQueueItem> {
  const item = await enqueuePendingComplaint(payload, candidateId);
  emit({ type: 'QUEUE_UPDATED' });

  // Register Background Sync API so Chrome/Android/Edge submit automatically even if tab closes
  try {
    const reg =
      swRegistration ||
      (typeof navigator !== 'undefined' && 'serviceWorker' in navigator
        ? await navigator.serviceWorker.ready
        : null);
    const syncManager = (reg as unknown as {
      sync?: { register: (tag: string) => Promise<void> };
    })?.sync;
    if (syncManager) {
      await syncManager.register(SYNC_TAG);
    }
  } catch {
    // Fallback handled by online event listener & visibility listener
  }

  return item;
}

export function initOfflineSync(): void {
  if (initialized || typeof window === 'undefined') return;
  initialized = true;

  if ('serviceWorker' in navigator) {
    try {
      registerSW({
        immediate: true,
        onRegisteredSW(_swUrl, reg) {
          if (reg) {
            swRegistration = reg;
          }
          if (navigator.onLine) {
            void triggerOfflineSync();
          }
        },
        onRegisterError() {
          // App continues with in-page IndexedDB queue fallback
        },
      });
    } catch {
      // Ignore registration issues in unsupported embedded previews
    }

    navigator.serviceWorker.addEventListener('message', (event) => {
      const data = event.data as OfflineSyncEvent | undefined;
      if (!data?.type) return;
      if (
        data.type === 'COMPLAINT_SYNCED' ||
        data.type === 'SYNC_BATCH_COMPLETE'
      ) {
        emit(data);
      }
    });
  }

  const handleOnline = () => {
    void triggerOfflineSync();
  };

  const handleVisibility = () => {
    if (document.visibilityState === 'visible' && navigator.onLine) {
      void triggerOfflineSync();
    }
  };

  window.addEventListener('online', handleOnline);
  document.addEventListener('visibilitychange', handleVisibility);

  // Sync any leftover items on startup if currently online
  if (navigator.onLine) {
    window.setTimeout(() => {
      void triggerOfflineSync();
    }, 1200);
  }
}
