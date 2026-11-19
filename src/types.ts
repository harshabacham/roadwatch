export type SeverityLevel = 'None' | 'Low' | 'Moderate' | 'Severe' | 'Critical';

export interface BoundingBox {
  xMin: number; // 0-1000 normalized
  yMin: number; // 0-1000 normalized
  xMax: number; // 0-1000 normalized
  yMax: number; // 0-1000 normalized
}

export interface DepthEstimationDetailsField {
  method: string;
  shannonEntropyBits: number;
  baselineEntropyBits: number;
  entropyRatio: number;
  relativeScaleIndex: number;
  shadowOcclusionPct: number;
  confidenceRangeCm: string;
}

export interface PotholeDetectionItem {
  label: string;
  severity: 'Low' | 'Moderate' | 'Severe' | 'Critical';
  estimatedWidthCm: number;
  estimatedDepthCm: number;
  estimatedAreaSqCm: number;
  depthClass: 'Shallow (<5 cm)' | 'Medium (5–10 cm)' | 'Deep (10–15 cm)' | 'Severe (>15 cm)';
  depthConfidenceRange: string;
  depthEstimationDetails?: DepthEstimationDetailsField;
  patchCount: number;
  mahalanobisZ: number;
  luminanceDropPct: number;
  roughnessScorePct: number;
  classifierProbability?: number;
  hazardDescription: string;
  box: BoundingBox;
}

export interface DetectionResult {
  potholeDetected: boolean;
  overallSeverity: SeverityLevel;
  confidence: number;
  roadSurfaceType: string;
  detections: PotholeDetectionItem[];
  formalComplaintTitle: string;
  formalComplaintDescription: string;
  recommendedDepartment: string;
  analyzedAt: string;
}

export interface GeoCoordinates {
  lat: number;
  lng: number;
  accuracy: number;
  speed: number | null;
  heading: number | null;
  timestamp: number;
  source: 'gps' | 'manual';
}

export interface GovernmentPortalMatch {
  portalName: string;
  department: string;
  portalUrl: string;
  submissionType: '311 Portal' | 'National Highway / Grievance' | 'Municipal Public Works' | 'Open311 Standard';
  countryName: string;
  regionLabel: string;
  instructions: string;
  helpline?: string;
}

export interface GeocodedLocation {
  lat: number;
  lng: number;
  formattedAddress: string;
  street: string;
  neighborhood: string;
  city: string;
  state: string;
  postcode: string;
  country: string;
  countryCode: string;
  portal: GovernmentPortalMatch;
}

export type ComplaintStatus =
  | 'Queued for Govt Portal'
  | 'Dispatched via API'
  | 'Submitted to Govt Site'
  | 'Acknowledged by Authority'
  | 'Resolved';

export interface Open311Config {
  enabled: boolean;
  mode: 'open311' | 'webhook';
  endpointUrl: string;
  serviceCode: string;
  apiKey: string;
  jurisdictionId: string;
  reporterEmail: string;
  reporterName: string;
}

export interface ComplaintRecord {
  id: string;
  trackingCode: string;
  createdAt: string;
  updatedAt: string;
  title: string;
  description: string;
  severity: 'Low' | 'Moderate' | 'Severe' | 'Critical';
  confidence: number;
  roadSurfaceType: string;
  estimatedDepthCm: number | null;
  depthClass: string;
  depthEstimationDetails?: DepthEstimationDetailsField | null;
  lat: number;
  lng: number;
  accuracyMeters: number | null;
  address: string;
  city: string;
  state: string;
  countryCode: string;
  department: string;
  portalName: string;
  portalUrl: string;
  status: ComplaintStatus;
  externalTicketId: string;
  dispatchChannel: string;
  upstreamHttpStatus: number | null;
  upstreamResponseNote: string;
  autoFiled: boolean;
  imageDataUrl: string;
  detections: PotholeDetectionItem[];
  offlineQueued?: boolean;
  queueId?: string;
}

export interface DetectedPotholeCandidate {
  candidateId: string;
  detectedAt: string;
  detection: DetectionResult;
  primaryItem: PotholeDetectionItem;
  snapshotDataUrl: string;
  lat: number;
  lng: number;
  accuracyMeters: number | null;
  address: string;
  city: string;
  state: string;
  countryCode: string;
  portalName: string;
  portalUrl: string;
  department: string;
  decision: 'pending' | 'raising' | 'queued_offline' | 'raised' | 'rejected';
  queuedOfflineId?: string;
  raisedComplaint?: ComplaintRecord;
  error?: string;
}

export interface PendingComplaintPayload {
  title: string;
  description: string;
  severity: 'Low' | 'Moderate' | 'Severe' | 'Critical';
  confidence: number;
  roadSurfaceType: string;
  estimatedDepthCm: number | null;
  depthClass: string;
  depthEstimationDetails?: DepthEstimationDetailsField | null;
  lat: number;
  lng: number;
  accuracyMeters: number | null;
  address: string;
  city: string;
  state: string;
  countryCode: string;
  department: string;
  portalName: string;
  portalUrl: string;
  autoFiled: boolean;
  autoRaise: boolean;
  imageDataUrl: string;
  detections: PotholeDetectionItem[];
  open311Config: Open311Config;
}

export interface PendingComplaintQueueItem {
  queueId: string;
  candidateId?: string;
  createdAt: string;
  syncStatus: 'queued' | 'syncing' | 'failed';
  retryCount: number;
  lastError?: string;
  payload: PendingComplaintPayload;
}

export interface OfflineQueueStats {
  detectionsCount: number;
  pendingComplaintsCount: number;
  cachedComplaintsCount: number;
}

export type DatasetLabel = 'pothole' | 'no_pothole';

export type DatasetCategory =
  | 'pothole_cavity'
  | 'waterlogged_pothole'
  | 'alligator_cracking'
  | 'clean_road'
  | 'lane_marking'
  | 'shadow'
  | 'tar_joint'
  | 'wet_glare'
  | 'user_feedback';

export interface LabeledDatasetSample {
  id: string;
  label: DatasetLabel;
  category: DatasetCategory;
  title: string;
  sourceType: 'asset' | 'synthetic' | 'upload' | 'feedback';
  assetPath?: string;
  syntheticKind?: string;
  imageDataUrl?: string;
  createdAt: string;
}

export interface ModelEvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  f05Score: number;
  falsePositiveRate: number;
  rocAuc: number;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  datasetSamples: number;
  positiveSamples: number;
  negativeSamples: number;
  candidateRegionsEvaluated: number;
  positiveRegions: number;
  hardNegativeRegions: number;
  trainingEpochs: number;
}

export interface TrainedPotholeModel {
  version: number;
  trainedAt: string;
  algorithm: string;
  featureNames: string[];
  featureMeans: number[];
  featureStds: number[];
  weights: number[];
  bias: number;
  decisionThreshold: number;
  falsePositivePenalty: number;
  metrics: ModelEvaluationMetrics;
}

export interface KagglePotholeBox {
  label: string;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

export interface KagglePotholeRecord {
  id: string;
  filename: string;
  annotationFile: string;
  width: number;
  height: number;
  fileSizeBytes: number;
  boxCount: number;
  boxes: KagglePotholeBox[];
  cachedLocally?: boolean;
}

export interface KaggleDatasetResponse {
  dataset: string;
  sourceUrl: string;
  format: string;
  importedAt: string;
  totalImages: number;
  totalBoxes: number;
  avgBoxesPerImage: number;
  cachedImageCount: number;
  page: number;
  limit: number;
  totalFiltered: number;
  records: KagglePotholeRecord[];
}
