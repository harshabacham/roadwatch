import { CheckCircle2, CloudOff, Database, RefreshCw } from 'lucide-react';

interface OfflineSyncBarProps {
  isOnline: boolean;
  pendingCount: number;
  savedDetectionsCount: number;
  isSyncing: boolean;
  lastSyncNotice: string | null;
  onSyncNow: () => void;
}

export function OfflineSyncBar({
  isOnline,
  pendingCount,
  savedDetectionsCount,
  isSyncing,
  lastSyncNotice,
  onSyncNow,
}: OfflineSyncBarProps) {
  // Stay hidden when online with zero queued filings and no recent sync notice
  if (isOnline && pendingCount === 0 && !isSyncing && !lastSyncNotice) {
    return null;
  }

  if (!isOnline) {
    return (
      <div
        role="status"
        className="no-print bg-amber-950 text-amber-50 border-b border-amber-800/80"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs sm:text-sm">
          <div className="flex items-center gap-2.5 min-w-0">
            <CloudOff className="w-4 h-4 text-amber-400 shrink-0" />
            <span className="font-semibold">Offline Mode Active</span>
            <span className="text-amber-200/90 truncate">
              Pothole detection continues locally. Accepted complaints are queued in IndexedDB and will submit automatically once back online.
            </span>
          </div>
          <div className="flex items-center gap-3 font-mono text-xs text-amber-200 tabular-nums shrink-0">
            <span>{pendingCount} queued</span>
            <span aria-hidden="true">·</span>
            <span>{savedDetectionsCount} detected</span>
          </div>
        </div>
      </div>
    );
  }

  if (pendingCount > 0 || isSyncing) {
    return (
      <div
        role="status"
        className="no-print bg-slate-900 text-slate-100 border-b border-slate-800"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs sm:text-sm">
          <div className="flex items-center gap-2.5 min-w-0">
            <Database className="w-4 h-4 text-amber-400 shrink-0" />
            <span className="font-semibold">
              {isSyncing
                ? 'Syncing IndexedDB complaint queue...'
                : `${pendingCount} offline complaint${pendingCount === 1 ? '' : 's'} queued in IndexedDB`}
            </span>
            <span className="text-slate-300 hidden sm:inline">
              Connection restored — ready to file to government portals.
            </span>
          </div>
          <button
            type="button"
            onClick={onSyncNow}
            disabled={isSyncing}
            className="min-h-[34px] px-3 py-1 rounded-lg bg-amber-600 hover:bg-amber-700 disabled:opacity-60 text-white text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer shrink-0"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isSyncing ? 'animate-spin' : ''}`} />
            {isSyncing ? 'Syncing...' : 'Sync Now'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      role="status"
      className="no-print bg-emerald-950 text-emerald-50 border-b border-emerald-800"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2 flex items-center gap-2 text-xs sm:text-sm">
        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
        <span className="font-medium">{lastSyncNotice}</span>
      </div>
    </div>
  );
}
