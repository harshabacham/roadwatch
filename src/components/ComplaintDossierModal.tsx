import { useState } from 'react';
import {
  Check,
  Copy,
  Download,
  ExternalLink,
  Mail,
  MapPin,
  Printer,
  Trash2,
  X,
} from 'lucide-react';
import { ComplaintRecord, ComplaintStatus } from '../types';
import { buildFormattedComplaintText, getSeverityMeta } from '../utils/geo';

interface ComplaintDossierModalProps {
  complaint: ComplaintRecord;
  onClose: () => void;
  onUpdate: (
    id: string,
    updates: Partial<Pick<ComplaintRecord, 'status' | 'externalTicketId'>>
  ) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

const STATUS_OPTIONS: ComplaintStatus[] = [
  'Queued for Govt Portal',
  'Dispatched via API',
  'Submitted to Govt Site',
  'Acknowledged by Authority',
  'Resolved',
];

export function ComplaintDossierModal({
  complaint,
  onClose,
  onUpdate,
  onDelete,
}: ComplaintDossierModalProps) {
  const [copied, setCopied] = useState(false);
  const [ticketId, setTicketId] = useState(complaint.externalTicketId || '');
  const [status, setStatus] = useState<ComplaintStatus>(complaint.status);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const severityMeta = getSeverityMeta(complaint.severity);
  const formattedText = buildFormattedComplaintText(complaint);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(formattedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      // Fallback ignored if clipboard blocked
    }
  }

  function handleDownloadJson() {
    const payload = {
      open311_georeport_v2: {
        service_code: 'pothole',
        lat: complaint.lat,
        long: complaint.lng,
        address_string: complaint.address,
        description: `${complaint.title} — ${complaint.description}`,
      },
      roadwatch_dossier: {
        trackingCode: complaint.trackingCode,
        createdAt: complaint.createdAt,
        severity: complaint.severity,
        confidence: complaint.confidence,
        roadSurfaceType: complaint.roadSurfaceType,
        department: complaint.department,
        portalName: complaint.portalName,
        portalUrl: complaint.portalUrl,
        detections: complaint.detections,
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${complaint.trackingCode}-open311.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleSaveTracking() {
    setSaving(true);
    try {
      await onUpdate(complaint.id, {
        status,
        externalTicketId: ticketId.trim(),
      });
    } finally {
      setSaving(false);
    }
  }

  const mailtoHref = `mailto:?subject=${encodeURIComponent(
    `Road Hazard Complaint [${complaint.trackingCode}]: ${complaint.title}`
  )}&body=${encodeURIComponent(formattedText)}`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-slate-950/60 backdrop-blur-sm sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="dossier-title"
    >
      <div className="w-full max-w-3xl max-h-[92vh] overflow-y-auto bg-white rounded-t-3xl sm:rounded-2xl border border-slate-200 shadow-2xl">
        {/* Mobile grab handle */}
        <div className="sm:hidden w-10 h-1.5 bg-slate-300 rounded-full mx-auto mt-3" />

        {/* Header */}
        <div className="sticky top-0 z-10 bg-white/95 backdrop-blur-md px-5 sm:px-6 py-4 border-b border-slate-200 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span className="font-mono font-semibold text-slate-900">
                {complaint.trackingCode}
              </span>
              <span aria-hidden="true">·</span>
              <span>{new Date(complaint.createdAt).toLocaleString()}</span>
              <span aria-hidden="true">·</span>
              <span className={`font-semibold ${severityMeta.textClass}`}>
                {severityMeta.label}
              </span>
            </div>
            <h2
              id="dossier-title"
              className="font-display text-lg sm:text-xl font-bold text-slate-900 mt-1 truncate"
            >
              {complaint.title}
            </h2>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close complaint dossier"
            className="no-print min-h-[44px] min-w-[44px] -mr-2 flex items-center justify-center rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-5 sm:p-6 space-y-6">
          {/* Evidence photo with bounding box overlay */}
          <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-200 aspect-[4/3] max-h-[340px] w-full">
            {complaint.imageDataUrl ? (
              <>
                <img
                  src={complaint.imageDataUrl}
                  alt={complaint.title}
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover"
                />
                {complaint.detections.length > 0 && (
                  <svg
                    viewBox="0 0 1000 1000"
                    preserveAspectRatio="none"
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  >
                    {complaint.detections.map((d, idx) => {
                      const x = Math.max(0, d.box.xMin);
                      const y = Math.max(0, d.box.yMin);
                      const w = Math.max(20, d.box.xMax - d.box.xMin);
                      const h = Math.max(20, d.box.yMax - d.box.yMin);
                      const color = getSeverityMeta(d.severity).hex;
                      return (
                        <g key={idx}>
                          <rect
                            x={x}
                            y={y}
                            width={w}
                            height={h}
                            fill="none"
                            stroke={color}
                            strokeWidth="6"
                          />
                          <rect
                            x={x}
                            y={Math.max(0, y - 42)}
                            width={Math.min(460, Math.max(220, w))}
                            height="40"
                            fill={color}
                          />
                          <text
                            x={x + 10}
                            y={Math.max(28, y - 14)}
                            fill="#ffffff"
                            fontSize="24"
                            fontFamily="IBM Plex Mono, monospace"
                            fontWeight="600"
                          >
                            {d.severity} · {d.estimatedWidthCm}x{d.estimatedDepthCm}cm
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </>
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 p-6 text-center">
                <MapPin className="w-8 h-8 mb-2" />
                <p className="text-sm font-medium text-slate-300">
                  Snapshot not attached
                </p>
              </div>
            )}
          </div>

          {/* Location & Authority Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
              <div className="text-xs font-medium text-slate-500">
                Geotagged Location
              </div>
              <div className="text-sm font-semibold text-slate-900">
                {complaint.address}
              </div>
              <div className="font-mono text-xs text-slate-600">
                {complaint.lat.toFixed(6)}, {complaint.lng.toFixed(6)}
                {complaint.accuracyMeters
                  ? ` (±${Math.round(complaint.accuracyMeters)}m)`
                  : ''}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
              <div className="text-xs font-medium text-slate-500">
                Matched Government Authority
              </div>
              <div className="text-sm font-semibold text-slate-900">
                {complaint.portalName}
              </div>
              <div className="text-xs text-slate-600">{complaint.department}</div>
            </div>
          </div>

          {/* Unsupervised Depth Estimation (Image Entropy + Relative Scale) */}
          {(complaint.estimatedDepthCm !== null &&
            complaint.estimatedDepthCm !== undefined) ||
          complaint.depthEstimationDetails ? (
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-xs font-semibold text-slate-800">
                  Unsupervised Depth Estimation
                </div>
                <div className="text-[11px] text-slate-500">
                  {complaint.depthEstimationDetails?.method ||
                    'Shannon Image Entropy + Relative Scale'}
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div>
                  <div className="text-slate-500">Estimated Depth</div>
                  <div className="font-mono font-semibold text-slate-900 mt-0.5">
                    {complaint.estimatedDepthCm ?? '—'} cm
                    {complaint.depthClass ? ` · ${complaint.depthClass}` : ''}
                  </div>
                </div>
                {complaint.depthEstimationDetails && (
                  <>
                    <div>
                      <div className="text-slate-500">Shannon Entropy</div>
                      <div className="font-mono font-semibold text-slate-900 mt-0.5">
                        {complaint.depthEstimationDetails.shannonEntropyBits} bits (
                        {complaint.depthEstimationDetails.entropyRatio}x)
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-500">Relative Scale</div>
                      <div className="font-mono font-semibold text-slate-900 mt-0.5">
                        {complaint.depthEstimationDetails.relativeScaleIndex}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-500">Shadow Occlusion</div>
                      <div className="font-mono font-semibold text-slate-900 mt-0.5">
                        {complaint.depthEstimationDetails.shadowOcclusionPct}%
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : null}

          {/* Description & Detections */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-900">
              Inspection Summary
            </h3>
            <p className="text-sm text-slate-700 leading-relaxed">
              {complaint.description}
            </p>
            {complaint.detections.length > 0 && (
              <div className="divide-y divide-slate-200 border border-slate-200 rounded-xl overflow-hidden">
                {complaint.detections.map((d, i) => (
                  <div
                    key={i}
                    className="px-4 py-3 bg-white flex flex-wrap items-center justify-between gap-2 text-sm"
                  >
                    <div>
                      <span className="font-medium text-slate-900">{d.label}</span>
                      <span className="text-slate-500 text-xs block">
                        {d.hazardDescription}
                      </span>
                    </div>
                    <div className="font-mono text-xs text-slate-700 whitespace-nowrap">
                      {d.estimatedWidthCm} cm W · {d.estimatedDepthCm} cm D ·{' '}
                      <span className={getSeverityMeta(d.severity).textClass}>
                        {d.severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Dispatch Status & External Government Portal Submission */}
          <div className="no-print p-4 rounded-xl bg-amber-50/60 border border-amber-200 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  Submit to Government Website
                </h3>
                <p className="text-xs text-slate-600 mt-0.5">
                  {complaint.upstreamResponseNote}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                onClick={handleCopy}
                className="min-h-[44px] px-4 py-2 rounded-lg bg-white border border-slate-300 hover:bg-slate-50 text-slate-800 text-sm font-medium inline-flex items-center gap-2 cursor-pointer transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-600" />
                    Copied Report
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    Copy Official Report
                  </>
                )}
              </button>

              <a
                href={complaint.portalUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={handleCopy}
                className="min-h-[44px] px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-sm font-semibold inline-flex items-center gap-2 transition-colors"
              >
                Open {complaint.portalName}
                <ExternalLink className="w-4 h-4" />
              </a>

              <a
                href={mailtoHref}
                className="min-h-[44px] px-3.5 py-2 rounded-lg bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 text-sm font-medium inline-flex items-center gap-2 transition-colors"
              >
                <Mail className="w-4 h-4" />
                Email Report
              </a>
            </div>
          </div>

          {/* Update Official Tracking Status */}
          <div className="no-print pt-2 border-t border-slate-200 space-y-3">
            <h3 className="text-sm font-semibold text-slate-900">
              Government Ticket Tracking
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label
                  htmlFor="complaint-status-select"
                  className="block text-xs font-medium text-slate-600 mb-1"
                >
                  Dispatch Status
                </label>
                <select
                  id="complaint-status-select"
                  value={status}
                  onChange={(e) => setStatus(e.target.value as ComplaintStatus)}
                  className="w-full h-11 px-3 rounded-lg border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
                >
                  {STATUS_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="external-ticket-input"
                  className="block text-xs font-medium text-slate-600 mb-1"
                >
                  Official Govt Ticket #
                </label>
                <input
                  id="external-ticket-input"
                  type="text"
                  value={ticketId}
                  onChange={(e) => setTicketId(e.target.value)}
                  placeholder="e.g. 311-948210"
                  className="w-full h-11 px-3 rounded-lg border border-slate-300 bg-white text-sm font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
                />
              </div>

              <div className="flex items-end">
                <button
                  type="button"
                  disabled={saving}
                  onClick={handleSaveTracking}
                  className="w-full h-11 px-4 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-60 text-white text-sm font-semibold cursor-pointer transition-colors"
                >
                  {saving ? 'Saving...' : 'Save Status'}
                </button>
              </div>
            </div>
          </div>

          {/* Export & Delete Footer */}
          <div className="no-print pt-3 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleDownloadJson}
                className="min-h-[40px] px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-50 inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Open311 JSON
              </button>
              <button
                type="button"
                onClick={() => window.print()}
                className="min-h-[40px] px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-50 inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Printer className="w-3.5 h-3.5" />
                Print Dossier
              </button>
            </div>

            {!confirmDelete ? (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                className="min-h-[40px] px-3 py-1.5 rounded-lg text-xs font-medium text-red-600 hover:bg-red-50 inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete Record
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="min-h-[40px] px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(complaint.id)}
                  className="min-h-[40px] px-3 py-1.5 rounded-lg bg-red-600 text-white text-xs font-semibold cursor-pointer"
                >
                  Confirm Delete
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
