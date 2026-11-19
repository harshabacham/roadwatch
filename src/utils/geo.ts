import {
  ComplaintRecord,
  DetectedPotholeCandidate,
  Open311Config,
  PendingComplaintPayload,
  SeverityLevel,
} from '../types';

export const COMPLAINT_CLUSTER_RADIUS_METERS = 5;

const SEVERITY_RANK: Record<string, number> = {
  None: 0,
  Low: 1,
  Moderate: 2,
  Severe: 3,
  Critical: 4,
};

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

/**
 * Groups detected pothole candidates within a given radius (default 5m) into
 * spatial clusters during complaint filing so nearby detections become one
 * consolidated cluster report rather than duplicate complaints.
 */
export function groupCandidatesByRadius(
  candidates: DetectedPotholeCandidate[],
  radiusMeters = COMPLAINT_CLUSTER_RADIUS_METERS
): DetectedPotholeCandidate[][] {
  const visited = new Array<boolean>(candidates.length).fill(false);
  const clusters: DetectedPotholeCandidate[][] = [];

  for (let i = 0; i < candidates.length; i++) {
    if (visited[i]) continue;
    visited[i] = true;
    const queue = [i];
    const cluster: DetectedPotholeCandidate[] = [];

    while (queue.length > 0) {
      const idx = queue.shift()!;
      const curr = candidates[idx];
      cluster.push(curr);

      for (let j = 0; j < candidates.length; j++) {
        if (visited[j]) continue;
        const other = candidates[j];
        if (
          haversineMeters(curr.lat, curr.lng, other.lat, other.lng) <=
          radiusMeters
        ) {
          visited[j] = true;
          queue.push(j);
        }
      }
    }

    clusters.push(cluster);
  }

  return clusters;
}

export function buildComplaintPayloadFromCluster(
  cluster: DetectedPotholeCandidate[],
  open311Config: Open311Config,
  radiusMeters = COMPLAINT_CLUSTER_RADIUS_METERS
): PendingComplaintPayload {
  const primary = cluster[0];
  if (cluster.length <= 1) {
    return {
      title: primary.detection.formalComplaintTitle,
      description: primary.detection.formalComplaintDescription,
      severity: primary.primaryItem.severity,
      confidence: primary.detection.confidence,
      roadSurfaceType: primary.detection.roadSurfaceType,
      estimatedDepthCm: primary.primaryItem.estimatedDepthCm,
      depthClass: primary.primaryItem.depthClass,
      depthEstimationDetails:
        primary.primaryItem.depthEstimationDetails || null,
      lat: primary.lat,
      lng: primary.lng,
      accuracyMeters: primary.accuracyMeters,
      address: primary.address,
      city: primary.city,
      state: primary.state,
      countryCode: primary.countryCode,
      department: primary.department,
      portalName: primary.portalName,
      portalUrl: primary.portalUrl,
      autoFiled: true,
      autoRaise: true,
      imageDataUrl: primary.snapshotDataUrl,
      detections: [primary.primaryItem],
      isClusterReport: false,
      clusterCount: 1,
      clusterRadiusMeters: radiusMeters,
      open311Config,
    };
  }

  // Pick worst severity / deepest pothole as representative evidence
  let lead = primary;
  for (const cand of cluster) {
    const candRank = SEVERITY_RANK[cand.primaryItem.severity] ?? 0;
    const leadRank = SEVERITY_RANK[lead.primaryItem.severity] ?? 0;
    if (
      candRank > leadRank ||
      (candRank === leadRank &&
        cand.primaryItem.estimatedDepthCm > lead.primaryItem.estimatedDepthCm)
    ) {
      lead = cand;
    }
  }

  const centroidLat =
    cluster.reduce((sum, c) => sum + c.lat, 0) / cluster.length;
  const centroidLng =
    cluster.reduce((sum, c) => sum + c.lng, 0) / cluster.length;
  const maxWidthCm = Math.max(
    ...cluster.map((c) => c.primaryItem.estimatedWidthCm)
  );
  const maxDepthCm = Math.max(
    ...cluster.map((c) => c.primaryItem.estimatedDepthCm)
  );
  const totalAreaSqCm = cluster.reduce(
    (sum, c) => sum + c.primaryItem.estimatedAreaSqCm,
    0
  );
  const maxConfidence = Math.max(
    ...cluster.map((c) => c.detection.confidence)
  );

  const consolidatedDetections = cluster.map((c, idx) => ({
    ...c.primaryItem,
    label: `Cluster Pothole #${idx + 1} (${c.primaryItem.severity})`,
  }));

  return {
    title: `Cluster Report (${cluster.length} Potholes within ${radiusMeters}m) — ${lead.primaryItem.severity} Road Defect`,
    description: `Consolidated cluster complaint grouping ${cluster.length} detected potholes within a ${radiusMeters}-meter radius to prevent duplicate reports. Worst hazard measures approx. ${maxWidthCm} cm wide × ${maxDepthCm} cm deep (${lead.primaryItem.depthClass}), combined damaged area ~${totalAreaSqCm} cm².`,
    severity: lead.primaryItem.severity,
    confidence: maxConfidence,
    roadSurfaceType: lead.detection.roadSurfaceType,
    estimatedDepthCm: maxDepthCm,
    depthClass: lead.primaryItem.depthClass,
    depthEstimationDetails: lead.primaryItem.depthEstimationDetails || null,
    lat: centroidLat,
    lng: centroidLng,
    accuracyMeters: lead.accuracyMeters,
    address: lead.address,
    city: lead.city,
    state: lead.state,
    countryCode: lead.countryCode,
    department: lead.department,
    portalName: lead.portalName,
    portalUrl: lead.portalUrl,
    autoFiled: true,
    autoRaise: true,
    imageDataUrl: lead.snapshotDataUrl,
    detections: consolidatedDetections,
    isClusterReport: true,
    clusterCount: cluster.length,
    clusterRadiusMeters: radiusMeters,
    open311Config,
  };
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
          complaint.depthEstimationDetails?.confidenceRangeCm
            ? ` | Est. Range: ${complaint.depthEstimationDetails.confidenceRangeCm}`
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
    ...(complaint.isClusterReport && (complaint.clusterCount || 0) > 1
      ? [
          `Cluster Report: ${complaint.clusterCount} potholes grouped within ${
            complaint.clusterRadiusMeters || COMPLAINT_CLUSTER_RADIUS_METERS
          }m radius`,
        ]
      : []),
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
