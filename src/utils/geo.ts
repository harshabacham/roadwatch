import { ComplaintRecord, SeverityLevel } from '../types';

export function haversineMeters(
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number {
  const R = 6371000;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export function getSeverityMeta(severity: SeverityLevel | string) {
  switch (severity) {
    case 'Critical':
      return {
        label: 'Critical Hazard',
        textClass: 'text-red-700',
        bgClass: 'bg-red-50',
        borderClass: 'border-red-200',
        hex: '#dc2626',
      };
    case 'Severe':
      return {
        label: 'Severe',
        textClass: 'text-orange-700',
        bgClass: 'bg-orange-50',
        borderClass: 'border-orange-200',
        hex: '#ea580c',
      };
    case 'Moderate':
      return {
        label: 'Moderate',
        textClass: 'text-amber-700',
        bgClass: 'bg-amber-50',
        borderClass: 'border-amber-200',
        hex: '#d97706',
      };
    case 'Low':
      return {
        label: 'Low',
        textClass: 'text-slate-700',
        bgClass: 'bg-slate-100',
        borderClass: 'border-slate-200',
        hex: '#475569',
      };
    default:
      return {
        label: 'Clear',
        textClass: 'text-emerald-700',
        bgClass: 'bg-emerald-50',
        borderClass: 'border-emerald-200',
        hex: '#16a34a',
      };
  }
}

export function buildFormattedComplaintText(complaint: ComplaintRecord): string {
  const mapsLink = `https://maps.google.com/?q=${complaint.lat.toFixed(6)},${complaint.lng.toFixed(6)}`;
  const dims =
    complaint.detections.length > 0
      ? complaint.detections
          .map(
            (d, i) =>
              `  - Hazard #${i + 1}: ${d.label} (${d.severity}), est. ${d.estimatedWidthCm} cm wide x ${d.estimatedDepthCm} cm deep — ${d.hazardDescription}`
          )
          .join('\n')
      : '  - Roadway asphalt cavity / surface failure documented via camera.';

  const depthLine =
    complaint.estimatedDepthCm !== null && complaint.estimatedDepthCm !== undefined
      ? `Estimated Depth: ${complaint.estimatedDepthCm} cm${
          complaint.depthClass ? ` (${complaint.depthClass})` : ''
        }${
          complaint.depthEstimationDetails
            ? ` | Entropy H=${complaint.depthEstimationDetails.shannonEntropyBits} bits, Scale=${complaint.depthEstimationDetails.relativeScaleIndex}`
            : ''
        }`
      : 'Estimated Depth: Not measured';

  return [
    `OFFICIAL ROAD HAZARD & POTHOLE REPAIR REQUEST`,
    `Reference ID: ${complaint.trackingCode}`,
    `Timestamp: ${new Date(complaint.createdAt).toLocaleString()}`,
    `Authority: ${complaint.department}`,
    ``,
    `LOCATION DETAILS`,
    `Address: ${complaint.address}`,
    `GPS Coordinates: ${complaint.lat.toFixed(6)}, ${complaint.lng.toFixed(6)}`,
    `Map Pin: ${mapsLink}`,
    ``,
    `HAZARD SUMMARY`,
    `Title: ${complaint.title}`,
    `Severity: ${complaint.severity} (Detection Confidence: ${Math.round(complaint.confidence)}%)`,
    depthLine,
    `Surface Type: ${complaint.roadSurfaceType}`,
    `Description: ${complaint.description}`,
    ``,
    `MEASURED DETECTIONS`,
    dims,
  ].join('\n');
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}
