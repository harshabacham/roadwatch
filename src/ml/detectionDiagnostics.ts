import type { SensitivityPreset } from './unsupervisedDetector';

export type DetectionRoute =
  | 'server-proxy'
  | 'direct-roboflow'
  | 'offline-fallback'
  | 'skipped-capture';

export interface DetectionEnvironmentInfo {
  mode: string;
  isProd: boolean;
  origin: string;
  online: boolean;
  devicePixelRatio: number;
  serviceWorkerControlled: boolean;
  userAgent: string;
}

export interface PreprocessingDiagnostics {
  sourceType: 'video' | 'image' | 'canvas';
  isVideoStream: boolean;
  videoReadyState: number | null;
  sourceWidth: number;
  sourceHeight: number;
  targetMaxDimension: number;
  jpegQuality: number;
  scaleFactor: number;
  downsampledWidth: number;
  downsampledHeight: number;
  encodedBytes: number;
  captureMs: number;
  surfaceAnalysisMs: number;
  frameMeanLuminance: number | null;
  baselineLuminance: number | null;
  baselineEntropyBits: number | null;
  isLikelyBlankFrame: boolean;
}

export interface TransportDiagnostics {
  route: DetectionRoute;
  proxyAttempted: boolean;
  proxyStatus: number | null;
  proxyContentType: string | null;
  proxyLatencyMs: number | null;
  proxyError: string | null;
  directAttempted: boolean;
  directStatus: number | null;
  directLatencyMs: number | null;
  directError: string | null;
  fallbackReason: string | null;
}

export interface PredictionBoxDiagnostic {
  index: number;
  label: string;
  confidence: number;
  severity: string;
  widthCm: number;
  depthCm: number;
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

export interface PredictionDiagnostics {
  engine: 'roboflow' | 'offline-fallback' | 'none';
  modelId: string;
  sensitivityPreset: SensitivityPreset;
  confidenceThreshold: number;
  overlapThreshold: number;
  returnedImageWidth: number | null;
  returnedImageHeight: number | null;
  rawCount: number;
  filteredCount: number;
  droppedBelowThresholdCount: number;
  rawConfidences: number[];
  filteredConfidences: number[];
  minConfidence: number | null;
  meanConfidence: number | null;
  maxConfidence: number | null;
  boxes: PredictionBoxDiagnostic[];
}

export interface TimingDiagnostics {
  preprocessingMs: number;
  networkMs: number;
  postprocessingMs: number;
  reportedModelInferenceMs: number | null;
  totalMs: number;
}

export interface DetectionDiagnosticEntry {
  runId: number;
  timestamp: string;
  environment: DetectionEnvironmentInfo;
  preprocessing: PreprocessingDiagnostics;
  transport: TransportDiagnostics;
  predictions: PredictionDiagnostics;
  timing: TimingDiagnostics;
}

export interface DetectionDiagnosticsSummary {
  environment: DetectionEnvironmentInfo;
  totalRuns: number;
  routeCounts: Record<DetectionRoute, number>;
  avgTotalMs: number;
  avgPreprocessingMs: number;
  avgNetworkMs: number;
  avgEncodedKb: number;
  blankFrameCount: number;
  totalRawPredictions: number;
  totalFilteredPredictions: number;
  totalDroppedBelowThreshold: number;
  confidenceStats: {
    sampleCount: number;
    min: number | null;
    mean: number | null;
    max: number | null;
  };
  recentFallbackReasons: string[];
}

const MAX_LOG_ENTRIES = 120;
let runCounter = 0;
let verboseConsoleLogging = true;
const diagnosticEntries: DetectionDiagnosticEntry[] = [];

export function getDetectionEnvironment(): DetectionEnvironmentInfo {
  const metaEnv = (import.meta as unknown as { env?: Record<string, unknown> })
    .env;
  const mode = typeof metaEnv?.MODE === 'string' ? metaEnv.MODE : 'unknown';
  const isProd = Boolean(metaEnv?.PROD);
  const hasWindow = typeof window !== 'undefined';
  const hasNav = typeof navigator !== 'undefined';

  return {
    mode,
    isProd,
    origin: hasWindow ? window.location.origin : 'non-browser',
    online: hasNav ? navigator.onLine : true,
    devicePixelRatio: hasWindow ? Number((window.devicePixelRatio || 1).toFixed(2)) : 1,
    serviceWorkerControlled: Boolean(
      hasNav && 'serviceWorker' in navigator && navigator.serviceWorker.controller
    ),
    userAgent: hasNav ? navigator.userAgent.slice(0, 120) : 'non-browser',
  };
}

export function computeConfidenceStats(scores: number[]): {
  min: number | null;
  mean: number | null;
  max: number | null;
} {
  if (scores.length === 0) {
    return { min: null, mean: null, max: null };
  }
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const s of scores) {
    if (s < min) min = s;
    if (s > max) max = s;
    sum += s;
  }
  return {
    min: Number(min.toFixed(4)),
    mean: Number((sum / scores.length).toFixed(4)),
    max: Number(max.toFixed(4)),
  };
}

export function recordDetectionDiagnostic(
  entryData: Omit<DetectionDiagnosticEntry, 'runId' | 'timestamp' | 'environment'>
): DetectionDiagnosticEntry {
  const entry: DetectionDiagnosticEntry = {
    runId: ++runCounter,
    timestamp: new Date().toISOString(),
    environment: getDetectionEnvironment(),
    ...entryData,
  };

  diagnosticEntries.push(entry);
  if (diagnosticEntries.length > MAX_LOG_ENTRIES) {
    diagnosticEntries.shift();
  }

  if (verboseConsoleLogging && typeof console !== 'undefined') {
    const envTag = entry.environment.isProd ? 'PROD' : 'DEV';
    const { preprocessing: prep, transport, predictions: pred, timing } = entry;
    const confSummary =
      pred.filteredConfidences.length > 0
        ? `conf=[${pred.filteredConfidences.map((c) => `${Math.round(c * 100)}%`).join(', ')}]`
        : pred.rawConfidences.length > 0
        ? `rawBelowThresh=[${pred.rawConfidences.map((c) => `${Math.round(c * 100)}%`).join(', ')}]`
        : 'conf=none';

    const header =
      `[RoadWatch:Detect #${entry.runId} | ${envTag}] ` +
      `route=${transport.route} engine=${pred.engine} ` +
      `src=${prep.sourceType}(${prep.sourceWidth}x${prep.sourceHeight}->${prep.downsampledWidth}x${prep.downsampledHeight} q=${prep.jpegQuality} ${(prep.encodedBytes / 1024).toFixed(1)}KB) ` +
      `preds=${pred.filteredCount}/${pred.rawCount} (thr=${pred.confidenceThreshold}) ${confSummary} ` +
      `time=${timing.totalMs}ms (prep=${timing.preprocessingMs}ms net=${timing.networkMs}ms)`;

    if (
      transport.route === 'offline-fallback' ||
      transport.route === 'skipped-capture' ||
      prep.isLikelyBlankFrame
    ) {
      console.warn(header, {
        fallbackReason: transport.fallbackReason,
        blankFrame: prep.isLikelyBlankFrame,
        environment: entry.environment,
        preprocessing: prep,
        transport,
        predictions: pred,
        timing,
      });
    } else if (typeof console.groupCollapsed === 'function') {
      console.groupCollapsed(header);
      console.info('Environment:', entry.environment);
      console.info('Preprocessing:', prep);
      console.info('Transport:', transport);
      console.info('Timing:', timing);
      console.info('Confidence Summary:', {
        modelId: pred.modelId,
        sensitivityPreset: pred.sensitivityPreset,
        confidenceThreshold: pred.confidenceThreshold,
        overlapThreshold: pred.overlapThreshold,
        rawCount: pred.rawCount,
        filteredCount: pred.filteredCount,
        droppedBelowThresholdCount: pred.droppedBelowThresholdCount,
        minConfidence: pred.minConfidence,
        meanConfidence: pred.meanConfidence,
        maxConfidence: pred.maxConfidence,
        rawConfidences: pred.rawConfidences,
      });
      if (pred.boxes.length > 0 && typeof console.table === 'function') {
        console.table(pred.boxes);
      }
      console.groupEnd();
    } else {
      console.info(header, entry);
    }
  }

  return entry;
}

export function getDetectionDiagnosticsLog(): DetectionDiagnosticEntry[] {
  return [...diagnosticEntries];
}

export function clearDetectionDiagnosticsLog(): void {
  diagnosticEntries.length = 0;
}

export function setDetectionDiagnosticsVerbose(enabled: boolean): void {
  verboseConsoleLogging = Boolean(enabled);
}

export function getDetectionDiagnosticsSummary(): DetectionDiagnosticsSummary {
  const routeCounts: Record<DetectionRoute, number> = {
    'server-proxy': 0,
    'direct-roboflow': 0,
    'offline-fallback': 0,
    'skipped-capture': 0,
  };

  let sumTotalMs = 0;
  let sumPrepMs = 0;
  let sumNetMs = 0;
  let sumBytes = 0;
  let blankFrameCount = 0;
  let totalRawPredictions = 0;
  let totalFilteredPredictions = 0;
  let totalDroppedBelowThreshold = 0;
  const allFilteredConfidences: number[] = [];
  const fallbackReasons = new Set<string>();

  for (const e of diagnosticEntries) {
    routeCounts[e.transport.route] = (routeCounts[e.transport.route] || 0) + 1;
    sumTotalMs += e.timing.totalMs;
    sumPrepMs += e.timing.preprocessingMs;
    sumNetMs += e.timing.networkMs;
    sumBytes += e.preprocessing.encodedBytes;
    if (e.preprocessing.isLikelyBlankFrame) blankFrameCount += 1;
    totalRawPredictions += e.predictions.rawCount;
    totalFilteredPredictions += e.predictions.filteredCount;
    totalDroppedBelowThreshold += e.predictions.droppedBelowThresholdCount;
    for (const c of e.predictions.filteredConfidences) {
      allFilteredConfidences.push(c);
    }
    if (e.transport.fallbackReason) {
      fallbackReasons.add(e.transport.fallbackReason);
    }
  }

  const n = diagnosticEntries.length;
  const conf = computeConfidenceStats(allFilteredConfidences);

  return {
    environment: getDetectionEnvironment(),
    totalRuns: n,
    routeCounts,
    avgTotalMs: n > 0 ? Number((sumTotalMs / n).toFixed(1)) : 0,
    avgPreprocessingMs: n > 0 ? Number((sumPrepMs / n).toFixed(1)) : 0,
    avgNetworkMs: n > 0 ? Number((sumNetMs / n).toFixed(1)) : 0,
    avgEncodedKb: n > 0 ? Number((sumBytes / n / 1024).toFixed(1)) : 0,
    blankFrameCount,
    totalRawPredictions,
    totalFilteredPredictions,
    totalDroppedBelowThreshold,
    confidenceStats: {
      sampleCount: allFilteredConfidences.length,
      ...conf,
    },
    recentFallbackReasons: Array.from(fallbackReasons),
  };
}

export function exportDetectionDiagnosticsReport(): string {
  return JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      summary: getDetectionDiagnosticsSummary(),
      entries: diagnosticEntries,
    },
    null,
    2
  );
}

// Expose a DevTools-friendly handle for side-by-side local dev vs production comparison
if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__ROADWATCH_DIAGNOSTICS__ = {
    getLogs: getDetectionDiagnosticsLog,
    getSummary: getDetectionDiagnosticsSummary,
    exportReport: exportDetectionDiagnosticsReport,
    clear: clearDetectionDiagnosticsLog,
    setVerbose: setDetectionDiagnosticsVerbose,
  };
}
