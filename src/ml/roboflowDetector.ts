import type {
  BoundingBox,
  DetectionResult,
  PotholeDetectionItem,
  SeverityLevel,
} from '../types';
import {
  computeBaselineRoadEntropy,
  estimatePotholeDepthUnsupervised,
} from './depthEstimator';
import {
  analyzeFrameUnsupervised,
  type ClusterProfile,
  type SensitivityPreset,
  type UnsupervisedAnalysisOutput,
  type UnsupervisedMetrics,
  type UnsupervisedParams,
} from './unsupervisedDetector';
import {
  computeConfidenceStats,
  recordDetectionDiagnostic,
  type DetectionDiagnosticEntry,
  type DetectionRoute,
  type PreprocessingDiagnostics,
  type TransportDiagnostics,
} from './detectionDiagnostics';

export interface RoboflowPrediction {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
  class_id?: number;
  detection_id?: string;
}

export interface RoboflowProxyResponse {
  inferenceId?: string;
  inferenceMs: number;
  modelId: string;
  image: { width: number; height: number };
  predictions: RoboflowPrediction[];
}

const ANALYSIS_W = 240;
const ANALYSIS_H = 180;

export interface FrameCaptureOptions {
  isVideoStream?: boolean;
  maxDimension?: number;
  jpegQuality?: number;
}

export interface DownsampledFrameInfo {
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
}

export interface VisualFrameSignature {
  width: number;
  height: number;
  /** Zero-mean unit-norm downsampled luminance grid for fast optical-shift & scene matching. */
  data: Float32Array;
}

function buildNormalizedGrid(
  luminance: Float32Array,
  srcW: number,
  srcH: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  gridW: number,
  gridH: number
): Float32Array {
  const out = new Float32Array(gridW * gridH);
  const roiW = Math.max(1, x1 - x0);
  const roiH = Math.max(1, y1 - y0);
  let mean = 0;

  for (let gy = 0; gy < gridH; gy++) {
    const sy0 = Math.min(srcH - 1, Math.max(0, Math.floor(y0 + (gy / gridH) * roiH)));
    const sy1 = Math.min(srcH, Math.max(sy0 + 1, Math.floor(y0 + ((gy + 1) / gridH) * roiH)));
    for (let gx = 0; gx < gridW; gx++) {
      const sx0 = Math.min(srcW - 1, Math.max(0, Math.floor(x0 + (gx / gridW) * roiW)));
      const sx1 = Math.min(srcW, Math.max(sx0 + 1, Math.floor(x0 + ((gx + 1) / gridW) * roiW)));
      let sum = 0;
      let cnt = 0;
      for (let y = sy0; y < sy1; y++) {
        const row = y * srcW;
        for (let x = sx0; x < sx1; x++) {
          sum += luminance[row + x];
          cnt++;
        }
      }
      const val = cnt > 0 ? sum / cnt : 0;
      out[gy * gridW + gx] = val;
      mean += val;
    }
  }

  mean /= out.length;
  let normSq = 0;
  for (let i = 0; i < out.length; i++) {
    const centered = out[i] - mean;
    out[i] = centered;
    normSq += centered * centered;
  }
  const norm = Math.sqrt(normSq);
  if (norm > 1e-5) {
    for (let i = 0; i < out.length; i++) {
      out[i] /= norm;
    }
  }
  return out;
}

function computeFrameVisualSignature(
  luminance: Float32Array,
  srcW: number,
  srcH: number
): VisualFrameSignature {
  const width = 24;
  const height = 18;
  return {
    width,
    height,
    data: buildNormalizedGrid(luminance, srcW, srcH, 0, 0, srcW, srcH, width, height),
  };
}

function computeCropVisualSignature(
  luminance: Float32Array,
  srcW: number,
  srcH: number,
  box: BoundingBox
): Float32Array {
  const x0 = Math.max(0, Math.min(srcW - 1, Math.floor((box.xMin / 1000) * srcW)));
  const y0 = Math.max(0, Math.min(srcH - 1, Math.floor((box.yMin / 1000) * srcH)));
  const x1 = Math.max(x0 + 1, Math.min(srcW, Math.ceil((box.xMax / 1000) * srcW)));
  const y1 = Math.max(y0 + 1, Math.min(srcH, Math.ceil((box.yMax / 1000) * srcH)));
  return buildNormalizedGrid(luminance, srcW, srcH, x0, y0, x1, y1, 8, 8);
}

let reusableCaptureCanvas: HTMLCanvasElement | null = null;
let reusableAnalysisCanvas: HTMLCanvasElement | null = null;
let proxyUnavailable = false;

const DIRECT_ROBOFLOW_API_URL = (
  (import.meta as any).env?.VITE_ROBOFLOW_API_URL ||
  'https://serverless.roboflow.com'
).replace(/\/+$/, '');
const DIRECT_ROBOFLOW_MODEL_ID = (
  (import.meta as any).env?.VITE_ROBOFLOW_MODEL_ID ||
  'pothole-detection-bqu6s-dwjbo/1'
).replace(/^\/+/, '');
const DIRECT_ROBOFLOW_API_KEY = (
  (import.meta as any).env?.VITE_ROBOFLOW_API_KEY ||
  '796gBoiSCcsgn1e1nXgt'
).trim();

interface DirectRoboflowCallResult {
  payload: RoboflowProxyResponse;
  status: number;
  latencyMs: number;
}

async function inferViaDirectRoboflow(
  imageDataUrl: string,
  confidence: number,
  overlap: number
): Promise<DirectRoboflowCallResult> {
  const commaIdx = imageDataUrl.indexOf(',');
  const rawBase64 =
    commaIdx >= 0 ? imageDataUrl.slice(commaIdx + 1).trim() : imageDataUrl.trim();
  const params = new URLSearchParams({
    api_key: DIRECT_ROBOFLOW_API_KEY,
    confidence: Math.max(0.05, Math.min(0.99, confidence)).toFixed(2),
    overlap: Math.max(0.05, Math.min(0.99, overlap)).toFixed(2),
  });
  const endpoint = `${DIRECT_ROBOFLOW_API_URL}/${DIRECT_ROBOFLOW_MODEL_ID}?${params.toString()}`;

  // Use a CORS-simple POST (safelisted Content-Type, no custom Authorization header)
  // because serverless.roboflow.com omits Access-Control-Allow-Origin on OPTIONS preflights
  // while returning Access-Control-Allow-Origin properly on simple POST responses.
  const startedAt = performance.now();
  let upstream: globalThis.Response | null = null;
  let lastErr: unknown = null;

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      upstream = await fetch(endpoint, {
        method: 'POST',
        mode: 'cors',
        credentials: 'omit',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: rawBase64,
      });
      if (upstream.ok || (upstream.status < 500 && upstream.status !== 429)) {
        break;
      }
      if (attempt === 0) {
        await new Promise((r) => setTimeout(r, 300));
      }
    } catch (err) {
      lastErr = err;
      if (attempt === 0) {
        await new Promise((r) => setTimeout(r, 300));
      }
    }
  }

  const latencyMs = Number((performance.now() - startedAt).toFixed(1));

  if (!upstream) {
    const err = new Error(
      lastErr instanceof Error
        ? lastErr.message
        : 'Direct Roboflow inference request failed'
    ) as Error & { status?: number; latencyMs?: number };
    err.latencyMs = latencyMs;
    throw err;
  }

  if (!upstream.ok) {
    const errText = await upstream.text().catch(() => '');
    const err = new Error(
      `Direct Roboflow inference failed (HTTP ${upstream.status}): ${errText.slice(0, 200)}`
    ) as Error & { status?: number; latencyMs?: number };
    err.status = upstream.status;
    err.latencyMs = latencyMs;
    throw err;
  }

  const data: any = await upstream.json();
  const elapsedMs = Number(
    (
      (typeof data.time === 'number' ? data.time * 1000 : null) ??
      latencyMs
    ).toFixed(1)
  );

  return {
    status: upstream.status,
    latencyMs,
    payload: {
      inferenceId: data.inference_id || '',
      inferenceMs: elapsedMs,
      modelId: data.resolved_model?.model_id || DIRECT_ROBOFLOW_MODEL_ID,
      image: data.image || { width: 0, height: 0 },
      predictions: Array.isArray(data.predictions) ? data.predictions : [],
    },
  };
}

function getConfidenceThreshold(sensitivity: SensitivityPreset): number {
  switch (sensitivity) {
    case 'conservative':
      return 0.48;
    case 'sensitive':
      return 0.18;
    case 'balanced':
    default:
      return 0.28;
  }
}

function captureFrameForRoboflow(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  options?: FrameCaptureOptions
): {
  canvas: HTMLCanvasElement;
  dataUrl: string;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  sourceType: 'video' | 'image' | 'canvas';
  isVideoStream: boolean;
  videoReadyState: number | null;
  targetMaxDimension: number;
  jpegQuality: number;
  scaleFactor: number;
  encodedBytes: number;
  captureMs: number;
} | null {
  const captureStart = performance.now();
  const sourceType: 'video' | 'image' | 'canvas' =
    source instanceof HTMLVideoElement
      ? 'video'
      : source instanceof HTMLImageElement
      ? 'image'
      : 'canvas';
  const videoReadyState =
    source instanceof HTMLVideoElement ? source.readyState : null;

  if (source instanceof HTMLVideoElement && source.readyState < 2) {
    return null;
  }

  const srcWidth =
    source instanceof HTMLVideoElement
      ? source.videoWidth
      : source instanceof HTMLImageElement
      ? source.naturalWidth
      : source.width;
  const srcHeight =
    source instanceof HTMLVideoElement
      ? source.videoHeight
      : source instanceof HTMLImageElement
      ? source.naturalHeight
      : source.height;

  if (!srcWidth || !srcHeight) return null;

  const isVideo =
    options?.isVideoStream ?? source instanceof HTMLVideoElement;
  const maxDim = Math.max(
    240,
    Math.min(1280, options?.maxDimension ?? (isVideo ? 640 : 960))
  );
  const jpegQuality = Math.max(
    0.45,
    Math.min(0.95, options?.jpegQuality ?? (isVideo ? 0.72 : 0.85))
  );

  const scale = Math.min(1, maxDim / Math.max(srcWidth, srcHeight));
  const width = Math.max(1, Math.round(srcWidth * scale));
  const height = Math.max(1, Math.round(srcHeight * scale));

  const canvas = reusableCaptureCanvas || document.createElement('canvas');
  reusableCaptureCanvas = canvas;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = isVideo ? 'medium' : 'high';
  ctx.drawImage(source, 0, 0, width, height);

  const dataUrl = canvas.toDataURL('image/jpeg', jpegQuality);
  const commaIdx = dataUrl.indexOf(',');
  const base64Len = commaIdx >= 0 ? dataUrl.length - commaIdx - 1 : dataUrl.length;
  const encodedBytes = Math.round((base64Len * 3) / 4);
  const captureMs = Number((performance.now() - captureStart).toFixed(1));

  return {
    canvas,
    dataUrl,
    width,
    height,
    sourceWidth: srcWidth,
    sourceHeight: srcHeight,
    sourceType,
    isVideoStream: isVideo,
    videoReadyState,
    targetMaxDimension: maxDim,
    jpegQuality,
    scaleFactor: Number(scale.toFixed(4)),
    encodedBytes,
    captureMs,
  };
}

function extractFrameLuminanceAndGradient(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
): {
  luminance: Float32Array;
  gradient: Float32Array;
  baselineLuminance: number;
  baselineEntropyBits: number;
  frameMeanLuminance: number;
  isLikelyBlankFrame: boolean;
  surfaceAnalysisMs: number;
} | null {
  const analysisStart = performance.now();
  const canvas = reusableAnalysisCanvas || document.createElement('canvas');
  reusableAnalysisCanvas = canvas;
  if (canvas.width !== ANALYSIS_W || canvas.height !== ANALYSIS_H) {
    canvas.width = ANALYSIS_W;
    canvas.height = ANALYSIS_H;
  }
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(source, 0, 0, ANALYSIS_W, ANALYSIS_H);
  const data = ctx.getImageData(0, 0, ANALYSIS_W, ANALYSIS_H).data;
  const total = ANALYSIS_W * ANALYSIS_H;
  const luminance = new Float32Array(total);
  const gradient = new Float32Array(total);

  let roadLumSum = 0;
  let roadLumCount = 0;
  let allLumSum = 0;
  let allLumSqSum = 0;
  const roadStartY = Math.floor(ANALYSIS_H * 0.35);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const l =
      (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
    luminance[p] = l;
    allLumSum += l;
    allLumSqSum += l * l;
    if (p >= roadStartY * ANALYSIS_W && l < 0.72) {
      roadLumSum += l;
      roadLumCount++;
    }
  }

  const frameMeanLuminance = total > 0 ? allLumSum / total : 0;
  const frameLumVariance =
    total > 0 ? Math.max(0, allLumSqSum / total - frameMeanLuminance * frameMeanLuminance) : 0;
  const isLikelyBlankFrame = frameMeanLuminance < 0.015 || frameLumVariance < 0.00015;

  for (let y = 1; y < ANALYSIS_H - 1; y++) {
    const row = y * ANALYSIS_W;
    for (let x = 1; x < ANALYSIS_W - 1; x++) {
      const p = row + x;
      const gx =
        -luminance[p - ANALYSIS_W - 1] +
        luminance[p - ANALYSIS_W + 1] -
        2 * luminance[p - 1] +
        2 * luminance[p + 1] -
        luminance[p + ANALYSIS_W - 1] +
        luminance[p + ANALYSIS_W + 1];
      const gy =
        -luminance[p - ANALYSIS_W - 1] -
        2 * luminance[p - ANALYSIS_W] -
        luminance[p - ANALYSIS_W + 1] +
        luminance[p + ANALYSIS_W - 1] +
        2 * luminance[p + ANALYSIS_W] +
        luminance[p + ANALYSIS_W + 1];
      gradient[p] = Math.min(1, Math.sqrt(gx * gx + gy * gy) * 0.45);
    }
  }

  const patchSize = 6;
  const cols = Math.floor(ANALYSIS_W / patchSize);
  const rows = Math.floor(ANALYSIS_H / patchSize);
  const normalPatches: Array<{ gx: number; gy: number }> = [];
  for (let gy = Math.floor(rows * 0.4); gy < rows; gy += 2) {
    for (let gx = 1; gx < cols - 1; gx += 2) {
      normalPatches.push({ gx, gy });
    }
  }

  const baselineLuminance =
    roadLumCount > 0 ? roadLumSum / roadLumCount : 0.42;
  const baselineEntropyBits = computeBaselineRoadEntropy(
    luminance,
    ANALYSIS_W,
    ANALYSIS_H,
    normalPatches,
    patchSize
  );

  const surfaceAnalysisMs = Number((performance.now() - analysisStart).toFixed(1));

  return {
    luminance,
    gradient,
    baselineLuminance,
    baselineEntropyBits,
    frameMeanLuminance: Number(frameMeanLuminance.toFixed(4)),
    isLikelyBlankFrame,
    surfaceAnalysisMs,
  };
}

function estimateSeverity(
  widthCm: number,
  depthCm: number,
  areaFrac: number,
  confidence: number
): 'Low' | 'Moderate' | 'Severe' | 'Critical' {
  const index = areaFrac * 18 + depthCm * 0.1 + confidence * 0.8;
  if (depthCm >= 16 || widthCm >= 95 || index >= 3.6) return 'Critical';
  if (depthCm >= 11 || widthCm >= 55 || index >= 2.5) return 'Severe';
  if (depthCm >= 6 || widthCm >= 30 || index >= 1.7) return 'Moderate';
  return 'Low';
}

export interface RoboflowDetectionOutput extends UnsupervisedAnalysisOutput {
  engine: 'roboflow' | 'offline-fallback';
  modelId: string;
  /** Exact downsampled frame snapshot captured at inference time (prevents motion drift on video streams). */
  snapshotDataUrl: string;
  downsampledSize: DownsampledFrameInfo;
  /** Lightweight visual signature of the analyzed frame for camera-motion & duplicate tracking. */
  frameSignature?: VisualFrameSignature;
  /** Per-detection visual crop descriptors aligned with detection.detections indices. */
  cropSignatures?: Float32Array[];
  /** Structured diagnostic trace for comparing dev vs production preprocessing & confidence scores. */
  diagnostics?: DetectionDiagnosticEntry;
}

/**
 * Primary pothole detector backed by the server-side Roboflow Serverless Inference proxy
 * (`/api/detect/roboflow`, using header-based auth) enriched with physical width/depth
 * estimation via Shannon image entropy and perspective geometry.
 * Explicitly downsamples video stream frames onto a frozen canvas before inference so
 * depth analysis, bounding boxes, and thumbnails remain locked to the exact analyzed frame.
 * Automatically falls back to on-device unsupervised detection when offline.
 */
export async function detectPotholesWithRoboflow(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  params: UnsupervisedParams,
  options?: FrameCaptureOptions
): Promise<RoboflowDetectionOutput | null> {
  const overallStart = performance.now();
  const confidenceThreshold = getConfidenceThreshold(params.sensitivity);
  const overlapThreshold = 0.45;

  const frame = captureFrameForRoboflow(source, options);
  if (!frame) {
    const sourceType: 'video' | 'image' | 'canvas' =
      source instanceof HTMLVideoElement
        ? 'video'
        : source instanceof HTMLImageElement
        ? 'image'
        : 'canvas';
    recordDetectionDiagnostic({
      preprocessing: {
        sourceType,
        isVideoStream: options?.isVideoStream ?? source instanceof HTMLVideoElement,
        videoReadyState: source instanceof HTMLVideoElement ? source.readyState : null,
        sourceWidth:
          source instanceof HTMLVideoElement
            ? source.videoWidth
            : source instanceof HTMLImageElement
            ? source.naturalWidth
            : source.width,
        sourceHeight:
          source instanceof HTMLVideoElement
            ? source.videoHeight
            : source instanceof HTMLImageElement
            ? source.naturalHeight
            : source.height,
        targetMaxDimension: options?.maxDimension ?? 640,
        jpegQuality: options?.jpegQuality ?? 0.72,
        scaleFactor: 0,
        downsampledWidth: 0,
        downsampledHeight: 0,
        encodedBytes: 0,
        captureMs: Number((performance.now() - overallStart).toFixed(1)),
        surfaceAnalysisMs: 0,
        frameMeanLuminance: null,
        baselineLuminance: null,
        baselineEntropyBits: null,
        isLikelyBlankFrame: true,
      },
      transport: {
        route: 'skipped-capture',
        proxyAttempted: false,
        proxyStatus: null,
        proxyContentType: null,
        proxyLatencyMs: null,
        proxyError: null,
        directAttempted: false,
        directStatus: null,
        directLatencyMs: null,
        directError: null,
        fallbackReason:
          source instanceof HTMLVideoElement && source.readyState < 2
            ? `Video not ready (readyState=${source.readyState})`
            : 'Source media had zero intrinsic dimensions',
      },
      predictions: {
        engine: 'none',
        modelId: 'none',
        sensitivityPreset: params.sensitivity,
        confidenceThreshold,
        overlapThreshold,
        returnedImageWidth: null,
        returnedImageHeight: null,
        rawCount: 0,
        filteredCount: 0,
        droppedBelowThresholdCount: 0,
        rawConfidences: [],
        filteredConfidences: [],
        minConfidence: null,
        meanConfidence: null,
        maxConfidence: null,
        boxes: [],
      },
      timing: {
        preprocessingMs: Number((performance.now() - overallStart).toFixed(1)),
        networkMs: 0,
        postprocessingMs: 0,
        reportedModelInferenceMs: null,
        totalMs: Number((performance.now() - overallStart).toFixed(1)),
      },
    });
    return null;
  }

  const downsampledSize: DownsampledFrameInfo = {
    width: frame.width,
    height: frame.height,
    sourceWidth: frame.sourceWidth,
    sourceHeight: frame.sourceHeight,
  };

  // Compute surface features & optional diagnostic overlay on the exact frozen frame BEFORE async fetch
  const surface = extractFrameLuminanceAndGradient(frame.canvas);
  const frameSignature = surface
    ? computeFrameVisualSignature(surface.luminance, ANALYSIS_W, ANALYSIS_H)
    : undefined;
  const localOverlayPass =
    params.overlayMode !== 'detections'
      ? analyzeFrameUnsupervised(frame.canvas, params)
      : null;

  const preprocessingDiag: PreprocessingDiagnostics = {
    sourceType: frame.sourceType,
    isVideoStream: frame.isVideoStream,
    videoReadyState: frame.videoReadyState,
    sourceWidth: frame.sourceWidth,
    sourceHeight: frame.sourceHeight,
    targetMaxDimension: frame.targetMaxDimension,
    jpegQuality: frame.jpegQuality,
    scaleFactor: frame.scaleFactor,
    downsampledWidth: frame.width,
    downsampledHeight: frame.height,
    encodedBytes: frame.encodedBytes,
    captureMs: frame.captureMs,
    surfaceAnalysisMs: surface?.surfaceAnalysisMs ?? 0,
    frameMeanLuminance: surface?.frameMeanLuminance ?? null,
    baselineLuminance: surface ? Number(surface.baselineLuminance.toFixed(4)) : null,
    baselineEntropyBits: surface ? Number(surface.baselineEntropyBits.toFixed(3)) : null,
    isLikelyBlankFrame: surface?.isLikelyBlankFrame ?? false,
  };

  const preprocessingMs = Number((performance.now() - overallStart).toFixed(1));

  const transportDiag: TransportDiagnostics = {
    route: 'server-proxy',
    proxyAttempted: false,
    proxyStatus: null,
    proxyContentType: null,
    proxyLatencyMs: null,
    proxyError: null,
    directAttempted: false,
    directStatus: null,
    directLatencyMs: null,
    directError: null,
    fallbackReason: null,
  };

  if (typeof navigator !== 'undefined' && !navigator.onLine) {
    const postStart = performance.now();
    const offline =
      localOverlayPass || analyzeFrameUnsupervised(frame.canvas, params);
    if (!offline) return null;
    const offlineCrops = surface
      ? offline.detection.detections.map((d) =>
          computeCropVisualSignature(surface.luminance, ANALYSIS_W, ANALYSIS_H, d.box)
        )
      : undefined;
    const offlineScores = offline.detection.detections.map((d) =>
      Number((d.classifierProbability ?? offline.detection.confidence / 100).toFixed(4))
    );
    const offlineStats = computeConfidenceStats(offlineScores);
    const postMs = Number((performance.now() - postStart).toFixed(1));
    const totalEndMs = Number((performance.now() - overallStart).toFixed(1));
    const diagEntry = recordDetectionDiagnostic({
      preprocessing: preprocessingDiag,
      transport: {
        ...transportDiag,
        route: 'offline-fallback',
        fallbackReason: 'navigator.onLine is false',
      },
      predictions: {
        engine: 'offline-fallback',
        modelId: 'offline-local-cv',
        sensitivityPreset: params.sensitivity,
        confidenceThreshold,
        overlapThreshold,
        returnedImageWidth: frame.width,
        returnedImageHeight: frame.height,
        rawCount: offline.detection.detections.length,
        filteredCount: offline.detection.detections.length,
        droppedBelowThresholdCount: 0,
        rawConfidences: offlineScores,
        filteredConfidences: offlineScores,
        minConfidence: offlineStats.min,
        meanConfidence: offlineStats.mean,
        maxConfidence: offlineStats.max,
        boxes: offline.detection.detections.map((d, i) => ({
          index: i + 1,
          label: d.label,
          confidence: offlineScores[i] ?? 0,
          severity: d.severity,
          widthCm: d.estimatedWidthCm,
          depthCm: d.estimatedDepthCm,
          xMin: d.box.xMin,
          yMin: d.box.yMin,
          xMax: d.box.xMax,
          yMax: d.box.yMax,
        })),
      },
      timing: {
        preprocessingMs,
        networkMs: 0,
        postprocessingMs: postMs,
        reportedModelInferenceMs: offline.metrics.inferenceMs,
        totalMs: totalEndMs,
      },
    });
    return {
      ...offline,
      engine: 'offline-fallback',
      modelId: 'offline-local-cv',
      snapshotDataUrl: frame.dataUrl,
      downsampledSize,
      frameSignature,
      cropSignatures: offlineCrops,
      diagnostics: diagEntry,
    };
  }

  const startedAt = performance.now();

  try {
    let payload: RoboflowProxyResponse | null = null;
    let activeRoute: DetectionRoute = 'server-proxy';

    // Try server-side proxy first (when full-stack server.ts is active)
    if (!proxyUnavailable) {
      transportDiag.proxyAttempted = true;
      const proxyStart = performance.now();
      let res: globalThis.Response | null = null;
      const reqBody = JSON.stringify({
        imageDataUrl: frame.dataUrl,
        confidence: confidenceThreshold,
        overlap: overlapThreshold,
      });

      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          res = await fetch('/api/detect/roboflow', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: reqBody,
          });
          if (res.ok || (res.status !== 502 && res.status !== 503 && res.status !== 504)) {
            break;
          }
          if (attempt === 0) {
            await new Promise((r) => setTimeout(r, 300));
          }
        } catch (err) {
          transportDiag.proxyError =
            err instanceof Error ? err.message : String(err);
          if (attempt === 0) {
            await new Promise((r) => setTimeout(r, 300));
          }
        }
      }

      transportDiag.proxyLatencyMs = Number(
        (performance.now() - proxyStart).toFixed(1)
      );

      if (res) {
        transportDiag.proxyStatus = res.status;
        transportDiag.proxyContentType = res.headers.get('content-type') || null;
      }

      if (res && res.ok) {
        const contentType = (res.headers.get('content-type') || '').toLowerCase();
        if (contentType.includes('application/json')) {
          const parsed = (await res.json().catch(() => null)) as RoboflowProxyResponse | null;
          if (parsed && Array.isArray(parsed.predictions)) {
            payload = parsed;
            activeRoute = 'server-proxy';
          } else {
            transportDiag.proxyError = 'JSON response missing predictions array';
            proxyUnavailable = true;
          }
        } else {
          // Static SPA host returned index.html instead of JSON API response
          transportDiag.proxyError = `Non-JSON content-type (${contentType || 'empty'})`;
          proxyUnavailable = true;
        }
      } else if (
        res &&
        (res.status === 401 ||
          res.status === 403 ||
          res.status === 404 ||
          res.status === 405)
      ) {
        // Static / shared deploy without backend proxy — switch directly to Roboflow
        transportDiag.proxyError = `Proxy returned HTTP ${res.status}`;
        proxyUnavailable = true;
      } else if (res) {
        transportDiag.proxyError = `Proxy returned HTTP ${res.status}`;
      }
    }

    // Fallback to direct Roboflow Serverless inference when backend proxy is absent/unavailable
    if (!payload) {
      transportDiag.directAttempted = true;
      try {
        const directRes = await inferViaDirectRoboflow(
          frame.dataUrl,
          confidenceThreshold,
          overlapThreshold
        );
        transportDiag.directStatus = directRes.status;
        transportDiag.directLatencyMs = directRes.latencyMs;
        payload = directRes.payload;
        activeRoute = 'direct-roboflow';
      } catch (directErr: any) {
        transportDiag.directStatus = directErr?.status ?? null;
        transportDiag.directLatencyMs = directErr?.latencyMs ?? null;
        transportDiag.directError =
          directErr instanceof Error ? directErr.message : String(directErr);
        throw directErr;
      }
    }

    transportDiag.route = activeRoute;
    const networkMs = Number((performance.now() - startedAt).toFixed(1));
    const postStart = performance.now();

    const imgW = payload.image?.width || frame.width;
    const imgH = payload.image?.height || frame.height;
    const rawPredictions = Array.isArray(payload.predictions)
      ? payload.predictions
      : [];

    const rawConfidences = rawPredictions
      .map((p) => Number((p.confidence ?? 0).toFixed(4)))
      .sort((a, b) => b - a);

    const filteredPredictions = rawPredictions
      .filter((p) => (p.confidence ?? 0) >= confidenceThreshold)
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));

    const filteredConfidences = filteredPredictions.map((p) =>
      Number((p.confidence ?? 0).toFixed(4))
    );

    const detections: PotholeDetectionItem[] = [];

    for (let idx = 0; idx < filteredPredictions.length; idx++) {
      const pred = filteredPredictions[idx];
      const leftPx = pred.x - pred.width / 2;
      const topPx = pred.y - pred.height / 2;
      const rightPx = pred.x + pred.width / 2;
      const bottomPx = pred.y + pred.height / 2;

      const box: BoundingBox = {
        xMin: Math.max(0, Math.min(1000, Math.round((leftPx / imgW) * 1000))),
        yMin: Math.max(0, Math.min(1000, Math.round((topPx / imgH) * 1000))),
        xMax: Math.max(0, Math.min(1000, Math.round((rightPx / imgW) * 1000))),
        yMax: Math.max(0, Math.min(1000, Math.round((bottomPx / imgH) * 1000))),
      };

      const normW = Math.max(0.01, (box.xMax - box.xMin) / 1000);
      const normH = Math.max(0.01, (box.yMax - box.yMin) / 1000);
      const areaFrac = normW * normH;
      const centerNormY = (box.yMin + box.yMax) / 2000;
      const perspectiveFactor = 0.7 + (1 - centerNormY) * 1.1;

      const estimatedWidthCm = Math.round(
        Math.max(18, Math.min(180, normW * 145 * perspectiveFactor))
      );
      const estimatedLengthCm = Math.round(
        Math.max(15, Math.min(160, normH * 125 * perspectiveFactor))
      );
      const estimatedAreaSqCm = Math.round(
        estimatedWidthCm * estimatedLengthCm * 0.75
      );

      let depthResult = surface
        ? estimatePotholeDepthUnsupervised({
            luminance: surface.luminance,
            gradient: surface.gradient,
            frameWidth: ANALYSIS_W,
            frameHeight: ANALYSIS_H,
            box,
            baselineLuminance: surface.baselineLuminance,
            baselineEntropyBits: surface.baselineEntropyBits,
            fillDensity: 0.72,
          })
        : null;

      // Compute ROI luminance drop & surface roughness for summary cards
      let lumDropPct = 18;
      let roughnessPct = 36;
      if (surface) {
        const x0 = Math.max(0, Math.floor((box.xMin / 1000) * ANALYSIS_W));
        const y0 = Math.max(0, Math.floor((box.yMin / 1000) * ANALYSIS_H));
        const x1 = Math.min(ANALYSIS_W, Math.ceil((box.xMax / 1000) * ANALYSIS_W));
        const y1 = Math.min(ANALYSIS_H, Math.ceil((box.yMax / 1000) * ANALYSIS_H));
        let sumL = 0;
        let sumG = 0;
        let cnt = 0;
        for (let y = y0; y < y1; y++) {
          const row = y * ANALYSIS_W;
          for (let x = x0; x < x1; x++) {
            sumL += surface.luminance[row + x];
            sumG += surface.gradient[row + x];
            cnt++;
          }
        }
        if (cnt > 0) {
          const meanL = sumL / cnt;
          const meanG = sumG / cnt;
          lumDropPct = Math.round(
            Math.max(0, surface.baselineLuminance - meanL) * 100
          );
          roughnessPct = Math.round(Math.min(100, meanG * 140));
        }
      }

      if (!depthResult) {
        const fallbackDepth = Number(
          Math.max(3, Math.min(20, 4 + areaFrac * 45)).toFixed(1)
        );
        depthResult = {
          estimatedDepthCm: fallbackDepth,
          depthClass:
            fallbackDepth < 5
              ? 'Shallow (<5 cm)'
              : fallbackDepth <= 10
              ? 'Medium (5–10 cm)'
              : fallbackDepth <= 15
              ? 'Deep (10–15 cm)'
              : 'Severe (>15 cm)',
          details: {
            method: 'Perspective Scale Analysis',
            shannonEntropyBits: 2.4,
            baselineEntropyBits: 1.6,
            entropyRatio: 1.5,
            relativeScaleIndex: Number((areaFrac * 6).toFixed(2)),
            shadowOcclusionPct: lumDropPct,
            confidenceRangeCm: `${Math.max(2, Math.round(fallbackDepth - 2))}–${Math.round(
              fallbackDepth + 2
            )} cm`,
          },
        };
      }

      const confPct = Math.round((pred.confidence ?? 0) * 100);
      const severity = estimateSeverity(
        estimatedWidthCm,
        depthResult.estimatedDepthCm,
        areaFrac,
        pred.confidence ?? 0.5
      );

      detections.push({
        label: `Pothole #${idx + 1} (${confPct}% confidence)`,
        severity,
        estimatedWidthCm,
        estimatedDepthCm: depthResult.estimatedDepthCm,
        estimatedAreaSqCm,
        depthClass: depthResult.depthClass,
        depthConfidenceRange: depthResult.details.confidenceRangeCm,
        depthEstimationDetails: depthResult.details,
        patchCount: Math.max(6, Math.round(areaFrac * 800)),
        mahalanobisZ: Number((2.2 + (pred.confidence ?? 0) * 2.5).toFixed(2)),
        luminanceDropPct: lumDropPct,
        roughnessScorePct: roughnessPct,
        classifierProbability: Number((pred.confidence ?? 0).toFixed(3)),
        hazardDescription: `Detected roadway pothole (${confPct}% model confidence) approx. ${estimatedWidthCm} cm wide and ${depthResult.estimatedDepthCm} cm deep (${depthResult.details.confidenceRangeCm}), ${roughnessPct}% surface roughness.`,
        box,
      });
    }

    const severityOrder: Record<SeverityLevel, number> = {
      None: 0,
      Low: 1,
      Moderate: 2,
      Severe: 3,
      Critical: 4,
    };
    let overallSeverity: SeverityLevel = 'None';
    for (const d of detections) {
      if (severityOrder[d.severity] > severityOrder[overallSeverity]) {
        overallSeverity = d.severity;
      }
    }

    const potholeDetected = detections.length > 0;
    const maxConf = potholeDetected
      ? Math.round(
          Math.max(...filteredPredictions.map((p) => (p.confidence ?? 0) * 100))
        )
      : 98;
    const maxWidth = potholeDetected
      ? Math.max(...detections.map((d) => d.estimatedWidthCm))
      : 0;
    const maxDepth = potholeDetected
      ? Math.max(...detections.map((d) => d.estimatedDepthCm))
      : 0;

    const baseL = surface?.baselineLuminance ?? 0.42;
    const roadSurfaceType =
      baseL < 0.35
        ? 'Dark Asphalt Pavement'
        : baseL < 0.55
        ? 'Weathered Asphalt Roadway'
        : 'Light Concrete / Wet Surface';

    const detection: DetectionResult = {
      potholeDetected,
      overallSeverity,
      confidence: maxConf,
      roadSurfaceType,
      detections,
      formalComplaintTitle: potholeDetected
        ? `${overallSeverity} Roadway Pothole Detected (${detections.length} Hazard${
            detections.length > 1 ? 's' : ''
          }, ~${maxWidth} cm)`
        : 'Clear Roadway Segment',
      formalComplaintDescription: potholeDetected
        ? `Automated road inspection detected ${detections.length} pavement hazard${
            detections.length > 1 ? 's' : ''
          } on ${roadSurfaceType.toLowerCase()}. Largest pothole measures approx. ${maxWidth} cm wide by ${maxDepth} cm deep.`
        : 'No roadway potholes detected in this segment.',
      recommendedDepartment: 'Department of Transportation — Roadway Maintenance',
      analyzedAt: new Date().toISOString(),
    };

    const totalMs = Number(
      (payload.inferenceMs || performance.now() - startedAt).toFixed(1)
    );

    const metrics: UnsupervisedMetrics = localOverlayPass?.metrics
      ? {
          ...localOverlayPass.metrics,
          inferenceMs: totalMs,
          dbscanClustersFound: detections.length,
        }
      : {
          inferenceMs: totalMs,
          gridCols: 40,
          gridRows: 30,
          totalPatches: 1200,
          roadPatches: 800,
          kClusters: params.kClusters,
          kmeansIterations: 1,
          wcss: 0,
          silhouetteScore: potholeDetected ? Number((maxConf / 100).toFixed(2)) : 0.98,
          anomalyThresholdZ: confidenceThreshold,
          anomalousPatchCount: detections.reduce((s, d) => s + d.patchCount, 0),
          dbscanClustersFound: detections.length,
          noisePatchesPruned: 0,
          baselineLuminanceMean: Number(baseL.toFixed(2)),
          baselineRoughnessMean: 0.08,
          trainedFrames: 0,
          gmmLogLikelihood: 0,
          isolationScoreMean: Number((maxConf / 100).toFixed(2)),
        };

    const clusters: ClusterProfile[] = localOverlayPass?.clusters || [
      {
        clusterId: 0,
        role: 'Normal Road Surface',
        patchCount: Math.max(0, 800 - metrics.anomalousPatchCount),
        percentage: Math.max(
          0,
          100 - (metrics.anomalousPatchCount / 800) * 100
        ),
        meanLuminance: baseL,
        meanRoughness: 0.08,
        colorHex: '#1e40af',
      },
      ...(potholeDetected
        ? [
            {
              clusterId: 1,
              role: 'Detected Pothole Regions',
              patchCount: metrics.anomalousPatchCount,
              percentage: Math.min(100, (metrics.anomalousPatchCount / 800) * 100),
              meanLuminance: Math.max(0.1, baseL - 0.16),
              meanRoughness: 0.35,
              colorHex: '#dc2626',
            },
          ]
        : []),
    ];

    const cropSignatures = surface
      ? detections.map((d) =>
          computeCropVisualSignature(surface.luminance, ANALYSIS_W, ANALYSIS_H, d.box)
        )
      : undefined;

    const postprocessingMs = Number((performance.now() - postStart).toFixed(1));
    const overallTotalMs = Number((performance.now() - overallStart).toFixed(1));
    const confStats = computeConfidenceStats(filteredConfidences);
    const resolvedModelId = payload.modelId || 'pothole-detection-bqu6s-dwjbo/1';

    const diagEntry = recordDetectionDiagnostic({
      preprocessing: preprocessingDiag,
      transport: transportDiag,
      predictions: {
        engine: 'roboflow',
        modelId: resolvedModelId,
        sensitivityPreset: params.sensitivity,
        confidenceThreshold,
        overlapThreshold,
        returnedImageWidth: payload.image?.width ?? null,
        returnedImageHeight: payload.image?.height ?? null,
        rawCount: rawPredictions.length,
        filteredCount: filteredPredictions.length,
        droppedBelowThresholdCount: Math.max(
          0,
          rawPredictions.length - filteredPredictions.length
        ),
        rawConfidences,
        filteredConfidences,
        minConfidence: confStats.min,
        meanConfidence: confStats.mean,
        maxConfidence: confStats.max,
        boxes: detections.map((d, i) => ({
          index: i + 1,
          label: d.label,
          confidence: filteredConfidences[i] ?? d.classifierProbability ?? 0,
          severity: d.severity,
          widthCm: d.estimatedWidthCm,
          depthCm: d.estimatedDepthCm,
          xMin: d.box.xMin,
          yMin: d.box.yMin,
          xMax: d.box.xMax,
          yMax: d.box.yMax,
        })),
      },
      timing: {
        preprocessingMs,
        networkMs,
        postprocessingMs,
        reportedModelInferenceMs: payload.inferenceMs ?? null,
        totalMs: overallTotalMs,
      },
    });

    return {
      detection,
      metrics,
      clusters,
      overlayDataUrl: localOverlayPass?.overlayDataUrl || '',
      engine: 'roboflow',
      modelId: resolvedModelId,
      snapshotDataUrl: frame.dataUrl,
      downsampledSize,
      frameSignature,
      cropSignatures,
      diagnostics: diagEntry,
    };
  } catch (err) {
    const networkMs = Number((performance.now() - startedAt).toFixed(1));
    const fallbackStart = performance.now();
    const fallbackReason = err instanceof Error ? err.message : String(err);

    const offline =
      localOverlayPass || analyzeFrameUnsupervised(frame.canvas, params);
    if (!offline) return null;
    const offlineCrops = surface
      ? offline.detection.detections.map((d) =>
          computeCropVisualSignature(surface.luminance, ANALYSIS_W, ANALYSIS_H, d.box)
        )
      : undefined;

    const offlineScores = offline.detection.detections.map((d) =>
      Number((d.classifierProbability ?? offline.detection.confidence / 100).toFixed(4))
    );
    const offlineStats = computeConfidenceStats(offlineScores);
    const postMs = Number((performance.now() - fallbackStart).toFixed(1));
    const overallTotalMs = Number((performance.now() - overallStart).toFixed(1));

    const diagEntry = recordDetectionDiagnostic({
      preprocessing: preprocessingDiag,
      transport: {
        ...transportDiag,
        route: 'offline-fallback',
        fallbackReason,
      },
      predictions: {
        engine: 'offline-fallback',
        modelId: 'offline-local-cv',
        sensitivityPreset: params.sensitivity,
        confidenceThreshold,
        overlapThreshold,
        returnedImageWidth: frame.width,
        returnedImageHeight: frame.height,
        rawCount: offline.detection.detections.length,
        filteredCount: offline.detection.detections.length,
        droppedBelowThresholdCount: 0,
        rawConfidences: offlineScores,
        filteredConfidences: offlineScores,
        minConfidence: offlineStats.min,
        meanConfidence: offlineStats.mean,
        maxConfidence: offlineStats.max,
        boxes: offline.detection.detections.map((d, i) => ({
          index: i + 1,
          label: d.label,
          confidence: offlineScores[i] ?? 0,
          severity: d.severity,
          widthCm: d.estimatedWidthCm,
          depthCm: d.estimatedDepthCm,
          xMin: d.box.xMin,
          yMin: d.box.yMin,
          xMax: d.box.xMax,
          yMax: d.box.yMax,
        })),
      },
      timing: {
        preprocessingMs,
        networkMs,
        postprocessingMs: postMs,
        reportedModelInferenceMs: offline.metrics.inferenceMs,
        totalMs: overallTotalMs,
      },
    });

    return {
      ...offline,
      engine: 'offline-fallback',
      modelId: 'offline-local-cv',
      snapshotDataUrl: frame.dataUrl,
      downsampledSize,
      frameSignature,
      cropSignatures: offlineCrops,
      diagnostics: diagEntry,
    };
  }
}
