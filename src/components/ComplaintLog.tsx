import { useMemo, useState } from 'react';
import { ChevronRight, ExternalLink, FileText, Search } from 'lucide-react';
import { ComplaintRecord, SeverityLevel } from '../types';
import { getSeverityMeta } from '../utils/geo';

interface ComplaintLogProps {
  complaints: ComplaintRecord[];
  loading: boolean;
  onSelectComplaint: (complaint: ComplaintRecord) => void;
  onOpenScanner: () => void;
}

export function ComplaintLog({
  complaints,
  loading,
  onSelectComplaint,
  onOpenScanner,
}: ComplaintLogProps) {
  const [search, setSearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState<'All' | SeverityLevel>('All');

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return complaints.filter((c) => {
      if (severityFilter !== 'All' && c.severity !== severityFilter) {
        return false;
      }
      if (!q) return true;
      return (
        c.trackingCode.toLowerCase().includes(q) ||
        c.title.toLowerCase().includes(q) ||
        c.address.toLowerCase().includes(q) ||
        c.portalName.toLowerCase().includes(q) ||
        c.externalTicketId.toLowerCase().includes(q)
      );
    });
  }, [complaints, search, severityFilter]);

  const counts = useMemo(() => {
    const severeCount = complaints.filter(
      (c) => c.severity === 'Severe' || c.severity === 'Critical'
    ).length;
    const submittedCount = complaints.filter(
      (c) =>
        c.status === 'Dispatched via API' ||
        c.status === 'Submitted to Govt Site' ||
        c.status === 'Acknowledged by Authority' ||
        c.status === 'Resolved'
    ).length;
    return {
      total: complaints.length,
      severe: severeCount,
      submitted: submittedCount,
    };
  }, [complaints]);

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6 pb-24 md:pb-10">
      {/* Header + Summary Metrics */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            Government Complaint Queue
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Geotagged pothole dossiers prepared for municipal 311, highway, and public works portals.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-4 sm:gap-6 border border-slate-200 bg-white rounded-xl px-4 py-3">
          <div>
            <div className="text-xs text-slate-500">Total Reports</div>
            <div className="font-mono text-lg sm:text-xl font-semibold text-slate-900">
              {counts.total}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">Severe / Critical</div>
            <div className="font-mono text-lg sm:text-xl font-semibold text-orange-700">
              {counts.severe}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">Dispatched</div>
            <div className="font-mono text-lg sm:text-xl font-semibold text-emerald-700">
              {counts.submitted}
            </div>
          </div>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search street, tracking code, or govt portal..."
            className="w-full h-11 pl-10 pr-4 rounded-xl border border-slate-200 bg-white text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-600"
          />
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-200/70 rounded-xl overflow-x-auto">
          {(['All', 'Critical', 'Severe', 'Moderate', 'Low'] as const).map(
            (sev) => (
              <button
                key={sev}
                type="button"
                onClick={() => setSeverityFilter(sev)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg whitespace-nowrap shrink-0 cursor-pointer transition-colors ${
                  severityFilter === sev
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {sev}
              </button>
            )
          )}
        </div>
      </div>

      {/* Loading Skeleton */}
      {loading && (
        <div className="bg-white border border-slate-200 rounded-2xl divide-y divide-slate-100">
          {[1, 2, 3].map((row) => (
            <div key={row} className="p-4 flex items-center gap-4 animate-pulse">
              <div className="w-16 h-12 rounded-lg bg-slate-200 shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="h-4 w-48 bg-slate-200 rounded" />
                <div className="h-3 w-72 bg-slate-100 rounded" />
              </div>
              <div className="h-4 w-24 bg-slate-200 rounded" />
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && complaints.length === 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl p-8 sm:p-12 text-center space-y-4">
          <div className="w-12 h-12 rounded-xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center mx-auto">
            <FileText className="w-6 h-6" />
          </div>
          <div className="max-w-md mx-auto space-y-1.5">
            <h2 className="font-display text-lg font-semibold text-slate-900">
              No pothole complaints filed yet
            </h2>
            <p className="text-sm text-slate-600">
              Open the Live Scanner on your phone camera—or test with a road sample—to detect potholes, capture GPS coordinates, and file reports to government portals.
            </p>
          </div>
          <button
            type="button"
            onClick={onOpenScanner}
            className="min-h-[44px] px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white text-sm font-semibold cursor-pointer transition-colors"
          >
            Open Live Scanner
          </button>
        </div>
      )}

      {/* Filtered Empty State */}
      {!loading && complaints.length > 0 && filtered.length === 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center">
          <p className="text-sm text-slate-600">
            No complaints match your current search or severity filter.
          </p>
        </div>
      )}

      {/* Desktop Data Table + Mobile Tap List */}
      {!loading && filtered.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
          {/* Desktop Table */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-xs font-semibold text-slate-600">
                  <th className="py-3 px-4">Ref / Time</th>
                  <th className="py-3 px-4">Hazard &amp; Location</th>
                  <th className="py-3 px-4">Est. Depth</th>
                  <th className="py-3 px-4">GPS Coordinates</th>
                  <th className="py-3 px-4">Severity</th>
                  <th className="py-3 px-4">Target Authority</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-sm">
                {filtered.map((item) => {
                  const sev = getSeverityMeta(item.severity);
                  const depthCm =
                    item.estimatedDepthCm ?? item.detections?.[0]?.estimatedDepthCm ?? null;
                  return (
                    <tr
                      key={item.id}
                      onClick={() => onSelectComplaint(item)}
                      className="hover:bg-slate-50/90 transition-colors cursor-pointer"
                    >
                      <td className="py-3 px-4 align-middle whitespace-nowrap">
                        <div className="font-mono font-semibold text-slate-900">
                          {item.trackingCode}
                        </div>
                        <div className="font-mono text-xs text-slate-500">
                          {new Date(item.createdAt).toLocaleDateString()}
                        </div>
                      </td>
                      <td className="py-3 px-4 align-middle max-w-xs">
                        <div className="flex items-center gap-3">
                          {item.imageDataUrl && (
                            <img
                              src={item.imageDataUrl}
                              alt={item.title}
                              referrerPolicy="no-referrer"
                              className="w-12 h-9 rounded object-cover border border-slate-200 shrink-0"
                            />
                          )}
                          <div className="min-w-0">
                            <div className="font-semibold text-slate-900 truncate">
                              {item.title}
                            </div>
                            <div className="text-xs text-slate-500 truncate">
                              {item.address}
                            </div>
                            {item.isClusterReport && (item.clusterCount || 0) > 1 && (
                              <div className="text-[11px] font-semibold text-amber-700 mt-0.5">
                                Cluster Report · {item.clusterCount} potholes within{' '}
                                {item.clusterRadiusMeters || 5}m
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4 align-middle font-mono text-xs text-slate-800 whitespace-nowrap">
                        {depthCm !== null ? `${depthCm} cm` : '—'}
                      </td>
                      <td className="py-3 px-4 align-middle font-mono text-xs text-slate-600 whitespace-nowrap">
                        {item.lat.toFixed(5)}, {item.lng.toFixed(5)}
                      </td>
                      <td className="py-3 px-4 align-middle whitespace-nowrap">
                        <span className={`font-semibold text-xs ${sev.textClass}`}>
                          {sev.label}
                        </span>
                      </td>
                      <td className="py-3 px-4 align-middle max-w-[200px]">
                        <div className="text-xs font-medium text-slate-800 truncate">
                          {item.portalName}
                        </div>
                        {item.externalTicketId && (
                          <div className="font-mono text-[11px] text-slate-500 truncate">
                            Ticket #{item.externalTicketId}
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-4 align-middle whitespace-nowrap">
                        <div className="text-xs font-medium text-slate-700">
                          {item.status}
                        </div>
                        {item.offlineQueued && (
                          <div className="text-[11px] font-medium text-amber-700">
                            IndexedDB Sync Pending
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-4 align-middle text-right whitespace-nowrap">
                        <a
                          href={item.portalUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="inline-flex items-center gap-1 text-xs font-semibold text-amber-700 hover:text-amber-800"
                        >
                          Portal
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Mobile Tap List */}
          <div className="md:hidden divide-y divide-slate-200">
            {filtered.map((item) => {
              const sev = getSeverityMeta(item.severity);
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectComplaint(item)}
                  className="w-full text-left p-4 flex items-center gap-3.5 hover:bg-slate-50 active:bg-slate-100 transition-colors cursor-pointer"
                >
                  {item.imageDataUrl ? (
                    <img
                      src={item.imageDataUrl}
                      alt={item.title}
                      referrerPolicy="no-referrer"
                      className="w-14 h-14 rounded-xl object-cover border border-slate-200 shrink-0"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-xl bg-slate-100 border border-slate-200 flex items-center justify-center shrink-0">
                      <FileText className="w-5 h-5 text-slate-400" />
                    </div>
                  )}

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 text-xs text-slate-500">
                      <span className="font-mono font-semibold text-slate-900">
                        {item.trackingCode}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span className={`font-semibold ${sev.textClass}`}>
                        {sev.label}
                      </span>
                      {typeof item.estimatedDepthCm === 'number' && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span className="font-mono font-semibold text-slate-700 tabular-nums">
                            {item.estimatedDepthCm.toFixed(1)} cm
                          </span>
                        </>
                      )}
                    </div>
                    <div className="font-semibold text-sm text-slate-900 truncate mt-0.5">
                      {item.title}
                    </div>
                    <div className="text-xs text-slate-500 truncate mt-0.5">
                      {item.address}
                    </div>
                    {item.isClusterReport && (item.clusterCount || 0) > 1 && (
                      <div className="text-[11px] font-semibold text-amber-700 mt-0.5">
                        Cluster Report · {item.clusterCount} potholes within{' '}
                        {item.clusterRadiusMeters || 5}m
                      </div>
                    )}
                  </div>

                  <ChevronRight className="w-5 h-5 text-slate-400 shrink-0" />
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
