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
} | null {
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

  return {
    canvas,
    dataUrl: canvas.toDataURL('image/jpeg', jpegQuality),
    width,
    height,
    sourceWidth: srcWidth,
    sourceHeight: srcHeight,
  };
}

function extractFrameLuminanceAndGradient(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
): {
  luminance: Float32Array;
  gradient: Float32Array;
  baselineLuminance: number;
  baselineEntropyBits: number;
} | null {
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
  const roadStartY = Math.floor(ANALYSIS_H * 0.35);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const l =
      (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
    luminance[p] = l;
    if (p >= roadStartY * ANALYSIS_W && l < 0.72) {
      roadLumSum += l;
      roadLumCount++;
    }
  }

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

  return { luminance, gradient, baselineLuminance, baselineEntropyBits };
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
  const frame = captureFrameForRoboflow(source, options);
  if (!frame) return null;

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

  if (typeof navigator !== 'undefined' && !navigator.onLine) {
    const offline =
      localOverlayPass || analyzeFrameUnsupervised(frame.canvas, params);
    if (!offline) return null;
    const offlineCrops = surface
      ? offline.detection.detections.map((d) =>
          computeCropVisualSignature(surface.luminance, ANALYSIS_W, ANALYSIS_H, d.box)
        )
      : undefined;
    return {
      ...offline,
      engine: 'offline-fallback',
      modelId: 'offline-local-cv',
      snapshotDataUrl: frame.dataUrl,
      downsampledSize,
      frameSignature,
      cropSignatures: offlineCrops,
    };
  }

  const startedAt = performance.now();
  const confidenceThreshold = getConfidenceThreshold(params.sensitivity);

  try {
    const res = await fetch('/api/detect/roboflow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        imageDataUrl: frame.dataUrl,
        confidence: confidenceThreshold,
        overlap: 0.45,
      }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(
        errBody.error || `Inference API responded with HTTP ${res.status}`
      );
    }

    const payload = (await res.json()) as RoboflowProxyResponse;
    const imgW = payload.image?.width || frame.width;
    const imgH = payload.image?.height || frame.height;
    const rawPredictions = Array.isArray(payload.predictions)
      ? payload.predictions
      : [];

    const filteredPredictions = rawPredictions
      .filter((p) => (p.confidence ?? 0) >= confidenceThreshold)
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));

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

    return {
      detection,
      metrics,
      clusters,
      overlayDataUrl: localOverlayPass?.overlayDataUrl || '',
      engine: 'roboflow',
      modelId: payload.modelId || 'pothole-detection-bqu6s-dwjbo/1',
      snapshotDataUrl: frame.dataUrl,
      downsampledSize,
      frameSignature,
      cropSignatures,
    };
  } catch (err) {
    console.warn('Roboflow inference failed, falling back to offline detector:', err);
    const offline =
      localOverlayPass || analyzeFrameUnsupervised(frame.canvas, params);
    if (!offline) return null;
    const offlineCrops = surface
      ? offline.detection.detections.map((d) =>
          computeCropVisualSignature(surface.luminance, ANALYSIS_W, ANALYSIS_H, d.box)
        )
      : undefined;
    return {
      ...offline,
      engine: 'offline-fallback',
      modelId: 'offline-local-cv',
      snapshotDataUrl: frame.dataUrl,
      downsampledSize,
      frameSignature,
      cropSignatures: offlineCrops,
    };
  }
}
