import { DetectionResult, PotholeDetectionItem, SeverityLevel } from '../types';
import {
  computeBaselineRoadEntropy,
  estimatePotholeDepthUnsupervised,
} from './depthEstimator';

export type DiagnosticOverlayMode =
  | 'detections'
  | 'kmeans'
  | 'heatmap'
  | 'dbscan';

export type SensitivityPreset = 'conservative' | 'balanced' | 'sensitive';

export interface UnsupervisedParams {
  kClusters: number; // 3, 4, or 5
  sensitivity: SensitivityPreset;
  dbscanMinPts: number; // 3, 4, or 6
  overlayMode: DiagnosticOverlayMode;
}

export interface ClusterProfile {
  clusterId: number;
  role: string;
  patchCount: number;
  percentage: number;
  meanLuminance: number;
  meanRoughness: number;
  colorHex: string;
}

export interface UnsupervisedMetrics {
  inferenceMs: number;
  gridCols: number;
  gridRows: number;
  totalPatches: number;
  roadPatches: number;
  kClusters: number;
  kmeansIterations: number;
  wcss: number;
  silhouetteScore: number;
  anomalyThresholdZ: number;
  anomalousPatchCount: number;
  dbscanClustersFound: number;
  noisePatchesPruned: number;
  baselineLuminanceMean: number;
  baselineRoughnessMean: number;
}

export interface UnsupervisedAnalysisOutput {
  detection: DetectionResult;
  metrics: UnsupervisedMetrics;
  clusters: ClusterProfile[];
  overlayDataUrl: string;
}

interface PatchFeature {
  gx: number;
  gy: number;
  idx: number;
  inRoadZone: boolean;
  meanL: number;
  stdL: number;
  meanGrad: number;
  peakGrad: number;
  chromaShift: number;
  warmth: number;
  bgL: number;
  bgGrad: number;
  bgStd: number;
  lumDepression: number;
  lumContrast: number;
  textureExcess: number;
  clusterId: number;
  anomalyZ: number;
  isAnomaly: boolean;
  dbscanClusterId: number;
}

const ANALYSIS_WIDTH = 240;
const ANALYSIS_HEIGHT = 180;
const PATCH_SIZE = 6; // 40 x 30 = 1200 patches

const CLUSTER_COLORS: Array<[number, number, number, string]> = [
  [30, 64, 175, '#1e40af'], // Blue - Asphalt Baseline
  [217, 119, 6, '#d97706'], // Amber - Surface Transition
  [220, 38, 38, '#dc2626'], // Red - Cavity / Rough Anomaly
  [100, 116, 139, '#64748b'], // Slate - Bright / Markings
  [13, 148, 136, '#0d9488'], // Teal - Specular / Water
];

function getSensitivityThreshold(preset: SensitivityPreset): {
  zThreshold: number;
  minDepression: number;
  minRoughExcess: number;
} {
  switch (preset) {
    case 'conservative':
      return { zThreshold: 2.9, minDepression: 0.075, minRoughExcess: 0.065 };
    case 'sensitive':
      return { zThreshold: 1.95, minDepression: 0.035, minRoughExcess: 0.035 };
    case 'balanced':
    default:
      return { zThreshold: 2.35, minDepression: 0.05, minRoughExcess: 0.048 };
  }
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function robustStats(values: number[], minScale: number): { med: number; mad: number } {
  if (values.length === 0) return { med: 0, mad: minScale };
  const med = median(values);
  const devs = values.map((v) => Math.abs(v - med));
  const rawMad = 1.4826 * median(devs);
  return { med, mad: Math.max(minScale, rawMad) };
}

/**
 * Deterministic K-Means++ initialization + Lloyd's iterations on standardized 5D patch vectors.
 */
function runKMeansPlusPlus(
  vectors: number[][],
  k: number,
  maxIter = 20
): { assignments: number[]; centroids: number[][]; iterations: number; wcss: number } {
  const n = vectors.length;
  if (n === 0) {
    return { assignments: [], centroids: [], iterations: 0, wcss: 0 };
  }
  const dims = vectors[0].length;
  const effectiveK = Math.min(k, n);

  const norms = vectors.map((v, idx) => ({
    idx,
    norm: v.reduce((acc, val) => acc + val * val, 0),
  }));
  norms.sort((a, b) => a.norm - b.norm);
  const centroids: number[][] = [vectors[norms[Math.floor(n * 0.35)].idx].slice()];

  const minDistSq = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < effectiveK; c++) {
    const lastCentroid = centroids[c - 1];
    let farthestIdx = 0;
    let farthestDist = -1;
    for (let i = 0; i < n; i++) {
      let d2 = 0;
      for (let d = 0; d < dims; d++) {
        const diff = vectors[i][d] - lastCentroid[d];
        d2 += diff * diff;
      }
      if (d2 < minDistSq[i]) minDistSq[i] = d2;
      if (minDistSq[i] > farthestDist) {
        farthestDist = minDistSq[i];
        farthestIdx = i;
      }
    }
    centroids.push(vectors[farthestIdx].slice());
  }

  const assignments = new Int32Array(n).fill(0);
  let iterations = 0;
  let wcss = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;
    let changed = 0;
    wcss = 0;

    for (let i = 0; i < n; i++) {
      let bestC = 0;
      let bestD2 = Infinity;
      const v = vectors[i];
      for (let c = 0; c < effectiveK; c++) {
        const cent = centroids[c];
        let d2 = 0;
        for (let d = 0; d < dims; d++) {
          const diff = v[d] - cent[d];
          d2 += diff * diff;
        }
        if (d2 < bestD2) {
          bestD2 = d2;
          bestC = c;
        }
      }
      wcss += bestD2;
      if (assignments[i] !== bestC) {
        assignments[i] = bestC;
        changed++;
      }
    }

    const sums = Array.from({ length: effectiveK }, () => new Float64Array(dims));
    const counts = new Int32Array(effectiveK);
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < dims; d++) {
        sums[c][d] += vectors[i][d];
      }
    }
    for (let c = 0; c < effectiveK; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < dims; d++) {
          centroids[c][d] = sums[c][d] / counts[c];
        }
      }
    }

    if (changed === 0 && iter > 0) break;
  }

  return {
    assignments: Array.from(assignments),
    centroids,
    iterations,
    wcss,
  };
}

function computeSilhouette(
  vectors: number[][],
  assignments: number[],
  k: number
): number {
  const n = vectors.length;
  if (n < 10 || k < 2) return 0;

  const step = Math.max(1, Math.floor(n / 180));
  let totalSil = 0;
  let evaluated = 0;

  for (let i = 0; i < n; i += step) {
    const ownC = assignments[i];
    const clusterSums = new Float64Array(k);
    const clusterCounts = new Int32Array(k);

    for (let j = 0; j < n; j += step) {
      if (i === j) continue;
      let d2 = 0;
      for (let d = 0; d < vectors[i].length; d++) {
        const diff = vectors[i][d] - vectors[j][d];
        d2 += diff * diff;
      }
      const dist = Math.sqrt(d2);
      const cj = assignments[j];
      clusterSums[cj] += dist;
      clusterCounts[cj]++;
    }

    const a =
      clusterCounts[ownC] > 0 ? clusterSums[ownC] / clusterCounts[ownC] : 0;
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ownC || clusterCounts[c] === 0) continue;
      const meanDist = clusterSums[c] / clusterCounts[c];
      if (meanDist < b) b = meanDist;
    }

    if (b < Infinity && Math.max(a, b) > 0) {
      totalSil += (b - a) / Math.max(a, b);
      evaluated++;
    }
  }

  return evaluated > 0 ? Math.max(-1, Math.min(1, totalSil / evaluated)) : 0;
}

function runDbscan(
  patches: PatchFeature[],
  eps: number,
  minPts: number
): { clusterCount: number; noiseCount: number } {
  const anomalyPatches = patches.filter((p) => p.isAnomaly);
  const visited = new Set<number>();
  const epsSq = eps * eps;
  let clusterId = 0;

  function regionQuery(center: PatchFeature): PatchFeature[] {
    const neighbors: PatchFeature[] = [];
    for (let i = 0; i < anomalyPatches.length; i++) {
      const p = anomalyPatches[i];
      const dx = p.gx - center.gx;
      const dy = p.gy - center.gy;
      if (dx * dx + dy * dy <= epsSq) {
        neighbors.push(p);
      }
    }
    return neighbors;
  }

  for (let i = 0; i < anomalyPatches.length; i++) {
    const p = anomalyPatches[i];
    if (visited.has(p.idx)) continue;
    visited.add(p.idx);

    const neighbors = regionQuery(p);
    if (neighbors.length < minPts) {
      p.dbscanClusterId = -1;
      continue;
    }

    const currentCid = clusterId++;
    p.dbscanClusterId = currentCid;
    const queue = [...neighbors];
    const inQueue = new Set<number>(neighbors.map((n) => n.idx));

    while (queue.length > 0) {
      const q = queue.shift()!;
      if (!visited.has(q.idx)) {
        visited.add(q.idx);
        const qNeighbors = regionQuery(q);
        if (qNeighbors.length >= minPts) {
          for (const n of qNeighbors) {
            if (!inQueue.has(n.idx)) {
              inQueue.add(n.idx);
              queue.push(n);
            }
          }
        }
      }
      if (q.dbscanClusterId === -1) {
        q.dbscanClusterId = currentCid;
      }
    }
  }

  const noiseCount = anomalyPatches.filter((p) => p.dbscanClusterId === -1).length;
  return { clusterCount: clusterId, noiseCount };
}

export function analyzeFrameUnsupervised(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  params: UnsupervisedParams
): UnsupervisedAnalysisOutput | null {
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

  const workCanvas = document.createElement('canvas');
  workCanvas.width = ANALYSIS_WIDTH;
  workCanvas.height = ANALYSIS_HEIGHT;
  const ctx = workCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(source, 0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
  const imgData = ctx.getImageData(0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
  return analyzePixelsUnsupervised(imgData.data, params, true);
}

export function analyzePixelsUnsupervised(
  data: Uint8ClampedArray,
  params: UnsupervisedParams,
  renderOverlay = false
): UnsupervisedAnalysisOutput {
  const startTime = performance.now();

  const totalPixels = ANALYSIS_WIDTH * ANALYSIS_HEIGHT;
  const lum = new Float32Array(totalPixels);
  const chroma = new Float32Array(totalPixels);
  const warmth = new Float32Array(totalPixels);
  const grad = new Float32Array(totalPixels);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    lum[p] = 0.299 * r + 0.587 * g + 0.114 * b;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    chroma[p] = maxC - minC;
    warmth[p] = r - b;
  }

  for (let y = 1; y < ANALYSIS_HEIGHT - 1; y++) {
    const row = y * ANALYSIS_WIDTH;
    for (let x = 1; x < ANALYSIS_WIDTH - 1; x++) {
      const p = row + x;
      const tl = lum[p - ANALYSIS_WIDTH - 1];
      const t = lum[p - ANALYSIS_WIDTH];
      const tr = lum[p - ANALYSIS_WIDTH + 1];
      const l = lum[p - 1];
      const r = lum[p + 1];
      const bl = lum[p + ANALYSIS_WIDTH - 1];
      const b = lum[p + ANALYSIS_WIDTH];
      const br = lum[p + ANALYSIS_WIDTH + 1];

      const gx = -tl + tr - 2 * l + 2 * r - bl + br;
      const gy = -tl - 2 * t - tr + bl + 2 * b + br;
      grad[p] = Math.min(1, Math.sqrt(gx * gx + gy * gy) * 0.45);
    }
  }

  const cols = Math.floor(ANALYSIS_WIDTH / PATCH_SIZE);
  const rows = Math.floor(ANALYSIS_HEIGHT / PATCH_SIZE);
  const patches: PatchFeature[] = [];

  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const startX = gx * PATCH_SIZE;
      const startY = gy * PATCH_SIZE;
      const count = PATCH_SIZE * PATCH_SIZE;

      let sumL = 0;
      let sumL2 = 0;
      let sumG = 0;
      let peakG = 0;
      let sumChroma = 0;
      let sumWarmth = 0;

      for (let py = 0; py < PATCH_SIZE; py++) {
        const rowOff = (startY + py) * ANALYSIS_WIDTH + startX;
        for (let px = 0; px < PATCH_SIZE; px++) {
          const p = rowOff + px;
          const lVal = lum[p];
          const gVal = grad[p];
          sumL += lVal;
          sumL2 += lVal * lVal;
          sumG += gVal;
          if (gVal > peakG) peakG = gVal;
          sumChroma += chroma[p];
          sumWarmth += warmth[p];
        }
      }

      const meanL = sumL / count;
      const variance = Math.max(0, sumL2 / count - meanL * meanL);
      const stdL = Math.sqrt(variance);

      // Perspective roadway zone: lower 66% of frame (excludes sky, distant horizon cars/buildings)
      const normY = (gy + 0.5) / rows;
      const inRoadZone = normY >= 0.34;

      patches.push({
        gx,
        gy,
        idx: gy * cols + gx,
        inRoadZone,
        meanL,
        stdL,
        meanGrad: sumG / count,
        peakGrad: peakG,
        chromaShift: sumChroma / count,
        warmth: sumWarmth / count,
        bgL: meanL,
        bgGrad: sumG / count,
        bgStd: stdL,
        lumDepression: 0,
        lumContrast: 0,
        textureExcess: 0,
        clusterId: 0,
        anomalyZ: 0,
        isAnomaly: false,
        dbscanClusterId: -1,
      });
    }
  }

  // Wide-neighborhood local road context (radius 8 patches) using 60th-percentile-like luminance
  const bgRadius = 8;
  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const p = patches[gy * cols + gx];
      let sumBgL = 0;
      let sumBgG = 0;
      let sumBgS = 0;
      let bgCount = 0;

      const yMin = Math.max(Math.floor(rows * 0.34), gy - bgRadius);
      const yMax = Math.min(rows - 1, gy + bgRadius);
      const xMin = Math.max(0, gx - bgRadius);
      const xMax = Math.min(cols - 1, gx + bgRadius);

      for (let ny = yMin; ny <= yMax; ny++) {
        for (let nx = xMin; nx <= xMax; nx++) {
          const np = patches[ny * cols + nx];
          if (np.meanL < 0.72) {
            sumBgL += np.meanL;
            sumBgG += np.meanGrad;
            sumBgS += np.stdL;
            bgCount++;
          }
        }
      }

      if (bgCount > 0) {
        p.bgL = sumBgL / bgCount;
        p.bgGrad = sumBgG / bgCount;
        p.bgStd = sumBgS / bgCount;
      }
      p.lumDepression = Math.max(0, p.bgL - p.meanL);
      p.lumContrast = Math.abs(p.meanL - p.bgL);
      p.textureExcess = Math.max(0, p.meanGrad + p.stdL - (p.bgGrad + p.bgStd) * 0.9);
    }
  }

  const roadPatches = patches.filter((p) => p.inRoadZone);
  const rawVectors = roadPatches.map((p) => [
    p.meanL,
    p.stdL * 4.0,
    p.meanGrad * 2.6,
    p.lumDepression * 4.5,
    p.chromaShift * 2.0 + p.lumContrast * 1.4,
  ]);

  const dims = 5;
  const featMeans = new Float64Array(dims);
  const featStds = new Float64Array(dims);

  for (const v of rawVectors) {
    for (let d = 0; d < dims; d++) featMeans[d] += v[d];
  }
  for (let d = 0; d < dims; d++) featMeans[d] /= Math.max(1, rawVectors.length);

  for (const v of rawVectors) {
    for (let d = 0; d < dims; d++) {
      const diff = v[d] - featMeans[d];
      featStds[d] += diff * diff;
    }
  }
  for (let d = 0; d < dims; d++) {
    featStds[d] = Math.max(
      0.02,
      Math.sqrt(featStds[d] / Math.max(1, rawVectors.length))
    );
  }

  const stdVectors = rawVectors.map((v) =>
    v.map((val, d) => (val - featMeans[d]) / featStds[d])
  );

  const kmeans = runKMeansPlusPlus(stdVectors, params.kClusters);
  const silhouette = computeSilhouette(
    stdVectors,
    kmeans.assignments,
    params.kClusters
  );

  roadPatches.forEach((p, idx) => {
    p.clusterId = kmeans.assignments[idx];
  });

  const clusterStats = Array.from({ length: params.kClusters }, (_, cid) => {
    const members = roadPatches.filter((p) => p.clusterId === cid);
    const lowerMembers = members.filter((p) => p.gy >= rows * 0.52);
    const count = members.length;
    const meanL =
      count > 0 ? members.reduce((s, p) => s + p.meanL, 0) / count : 0;
    const meanRough =
      count > 0
        ? members.reduce((s, p) => s + p.meanGrad + p.stdL, 0) / count
        : 0;
    const meanDep =
      count > 0 ? members.reduce((s, p) => s + p.lumDepression, 0) / count : 0;

    return {
      cid,
      members,
      count,
      lowerCount: lowerMembers.length,
      meanL,
      meanRough,
      meanDep,
    };
  });

  const sortedForAsphalt = [...clusterStats]
    .filter((c) => c.count > 0)
    .sort((a, b) => {
      const aScore =
        a.lowerCount * (a.meanL < 0.65 ? 1.2 : 0.3) -
        a.meanRough * 40 -
        a.meanDep * 60;
      const bScore =
        b.lowerCount * (b.meanL < 0.65 ? 1.2 : 0.3) -
        b.meanRough * 40 -
        b.meanDep * 60;
      return bScore - aScore;
    });

  const roadClusterId = sortedForAsphalt[0]?.cid ?? 0;

  // Robust Median + MAD statistics across roadPatches (outlier-resistant unsupervised baseline)
  const statL = robustStats(
    roadPatches.map((p) => p.meanL),
    0.06
  );
  const statStd = robustStats(
    roadPatches.map((p) => p.stdL),
    0.025
  );
  const statGrad = robustStats(
    roadPatches.map((p) => p.meanGrad),
    0.038
  );
  const statDep = robustStats(
    roadPatches.map((p) => p.lumDepression),
    0.028
  );
  const statContrast = robustStats(
    roadPatches.map((p) => p.lumContrast),
    0.035
  );

  const anomalyCluster = [...clusterStats]
    .filter((c) => c.cid !== roadClusterId && c.count > 0 && c.meanL < 0.65)
    .sort((a, b) => b.meanRough + b.meanDep * 2 - (a.meanRough + a.meanDep * 2))[0];
  const anomalyClusterId = anomalyCluster?.cid ?? -1;

  const clusters: ClusterProfile[] = clusterStats.map((c, idx) => {
    let role = 'Surface Transition';
    if (c.cid === roadClusterId) role = 'Dominant Asphalt Baseline';
    else if (c.cid === anomalyClusterId) role = 'Cavity & Roughness Anomaly';
    else if (c.meanL > 0.6) role = 'Bright Markings / Reflections';
    else if (c.meanRough > statGrad.med + statStd.med)
      role = 'Cracked Rim / Aggregate';

    const color =
      c.cid === roadClusterId
        ? CLUSTER_COLORS[0][3]
        : c.cid === anomalyClusterId
        ? CLUSTER_COLORS[2][3]
        : CLUSTER_COLORS[(idx + 1) % CLUSTER_COLORS.length][3];

    return {
      clusterId: c.cid,
      role,
      patchCount: c.count,
      percentage: roadPatches.length > 0 ? (c.count / roadPatches.length) * 100 : 0,
      meanLuminance: c.meanL,
      meanRoughness: c.meanRough,
      colorHex: color,
    };
  });

  const { zThreshold, minDepression, minRoughExcess } = getSensitivityThreshold(
    params.sensitivity
  );

  for (const p of roadPatches) {
    // Reject clean white lane stripes
    const isCleanBrightPaint =
      p.meanL > Math.max(0.68, statL.med + 0.24) &&
      p.lumDepression < 0.02 &&
      p.stdL < 0.085;
    if (isCleanBrightPaint) {
      p.anomalyZ = 0;
      p.isAnomaly = false;
      continue;
    }

    const zDep = Math.max(0, (p.lumDepression - statDep.med) / statDep.mad);
    const zStd = Math.max(0, (p.stdL - statStd.med) / statStd.mad);
    const zGrad = Math.max(0, (p.meanGrad - statGrad.med) / statGrad.mad);
    const zContrast = Math.max(
      0,
      (p.lumContrast - statContrast.med) / statContrast.mad
    );
    const zDarkShift = Math.max(0, (statL.med - p.meanL) / statL.mad);

    // Weighted Robust Mahalanobis distance
    let mDist = Math.sqrt(
      0.36 * zDep * zDep +
        0.24 * zStd * zStd +
        0.2 * zGrad * zGrad +
        0.12 * zContrast * zContrast +
        0.08 * zDarkShift * zDarkShift
    );

    if (p.clusterId === anomalyClusterId) {
      mDist *= 1.15;
    }

    p.anomalyZ = mDist;

    // Require distinct physical depression, sharp roughness excess, or reflective water-pool anomaly
    const hasCavityOrRoughness =
      p.lumDepression >= minDepression ||
      p.textureExcess >= minRoughExcess ||
      (p.lumContrast >= 0.095 && p.meanL > statL.med + 0.08 && p.gy > rows * 0.42);

    p.isAnomaly = mDist >= zThreshold && hasCavityOrRoughness;
  }

  // Tight 1-step morphological closure for smooth puddle interiors surrounded by cracked rims
  const closedSet = new Set<number>();
  for (let gy = 1; gy < rows - 1; gy++) {
    for (let gx = 1; gx < cols - 1; gx++) {
      const p = patches[gy * cols + gx];
      if (!p.inRoadZone || p.isAnomaly) continue;
      let hits = 0;
      for (let d = 1; d <= 2; d++) {
        if (gx - d >= 0 && patches[gy * cols + (gx - d)].isAnomaly) {
          hits++;
          break;
        }
      }
      for (let d = 1; d <= 2; d++) {
        if (gx + d < cols && patches[gy * cols + (gx + d)].isAnomaly) {
          hits++;
          break;
        }
      }
      for (let d = 1; d <= 2; d++) {
        if (gy - d >= 0 && patches[(gy - d) * cols + gx].isAnomaly) {
          hits++;
          break;
        }
      }
      for (let d = 1; d <= 2; d++) {
        if (gy + d < rows && patches[(gy + d) * cols + gx].isAnomaly) {
          hits++;
          break;
        }
      }
      if (hits >= 3) {
        closedSet.add(p.idx);
      }
    }
  }
  for (const idx of closedSet) {
    patches[idx].isAnomaly = true;
    patches[idx].anomalyZ = Math.max(patches[idx].anomalyZ, zThreshold + 0.25);
  }

  // Compute unsupervised baseline road Shannon entropy from non-anomalous road patches
  const normalRoadPatches = roadPatches.filter((p) => !p.isAnomaly);
  const baselineEntropyBits = computeBaselineRoadEntropy(
    lum,
    ANALYSIS_WIDTH,
    ANALYSIS_HEIGHT,
    normalRoadPatches,
    PATCH_SIZE
  );

  // Tight spatial DBSCAN (eps = 1.65 grid units keeps separate potholes distinct)
  const dbscanRes = runDbscan(patches, 1.65, params.dbscanMinPts);

  const detections: PotholeDetectionItem[] = [];

  for (let cid = 0; cid < dbscanRes.clusterCount; cid++) {
    const members = patches.filter((p) => p.dbscanClusterId === cid);
    if (members.length < params.dbscanMinPts) continue;

    let minGx = cols;
    let maxGx = 0;
    let minGy = rows;
    let maxGy = 0;
    let sumZ = 0;
    let sumDep = 0;
    let sumRough = 0;

    for (const m of members) {
      if (m.gx < minGx) minGx = m.gx;
      if (m.gx > maxGx) maxGx = m.gx;
      if (m.gy < minGy) minGy = m.gy;
      if (m.gy > maxGy) maxGy = m.gy;
      sumZ += m.anomalyZ;
      sumDep += m.lumDepression;
      sumRough += m.meanGrad + m.stdL;
    }

    const boxWPatches = maxGx - minGx + 1;
    const boxHPatches = maxGy - minGy + 1;
    const boxAreaPatches = boxWPatches * boxHPatches;
    const areaFrac = members.length / roadPatches.length;
    const aspectRatio = boxWPatches / Math.max(1, boxHPatches);
    const fillDensity = members.length / Math.max(1, boxAreaPatches);
    const meanZ = sumZ / members.length;
    const meanDep = sumDep / members.length;
    const meanRough = sumRough / members.length;

    // Filter out tiny specks and narrow linear shadows/curbs
    if (areaFrac < 0.006 || members.length < params.dbscanMinPts) continue;
    if (aspectRatio > 5.5 && boxHPatches <= 2) continue;
    if (fillDensity < 0.22 && members.length < 12) continue;

    const xMin = Math.round((minGx / cols) * 1000);
    const yMin = Math.round((minGy / rows) * 1000);
    const xMax = Math.round(((maxGx + 1) / cols) * 1000);
    const yMax = Math.round(((maxGy + 1) / rows) * 1000);
    const box = { xMin, yMin, xMax, yMax };

    // Estimate pothole depth using Shannon Image Entropy & Relative Scale Analysis
    const depthResult = estimatePotholeDepthUnsupervised({
      luminance: lum,
      gradient: grad,
      frameWidth: ANALYSIS_WIDTH,
      frameHeight: ANALYSIS_HEIGHT,
      box,
      baselineLuminance: statL.med,
      baselineEntropyBits,
      fillDensity,
    });

    const centerNormY = (minGy + maxGy + 1) / (2 * rows);
    const perspectiveFactor = 0.7 + (1 - centerNormY) * 1.1;
    const estimatedWidthCm = Math.round(
      Math.max(18, Math.min(180, (boxWPatches / cols) * 145 * perspectiveFactor))
    );
    const estimatedLengthCm = Math.round(
      Math.max(15, Math.min(160, (boxHPatches / rows) * 125 * perspectiveFactor))
    );
    const estimatedAreaSqCm = Math.round(
      estimatedWidthCm * estimatedLengthCm * Math.max(0.45, fillDensity)
    );

    let severity: 'Low' | 'Moderate' | 'Severe' | 'Critical' = 'Moderate';
    const severityIndex =
      areaFrac * 18 +
      meanZ * 0.35 +
      depthResult.estimatedDepthCm * 0.09;
    if (severityIndex >= 3.4 || areaFrac >= 0.09 || depthResult.estimatedDepthCm >= 17) {
      severity = 'Critical';
    } else if (
      severityIndex >= 2.35 ||
      areaFrac >= 0.04 ||
      depthResult.estimatedDepthCm >= 12
    ) {
      severity = 'Severe';
    } else if (severityIndex >= 1.6) {
      severity = 'Moderate';
    } else {
      severity = 'Low';
    }

    const clusterConfidence = Math.round(
      Math.min(98, Math.max(64, 58 + (meanZ - zThreshold) * 16 + fillDensity * 18))
    );

    const luminanceDropPct = Math.round(meanDep * 100);
    const roughnessScorePct = Math.round(meanRough * 100);

    detections.push({
      label: `Cluster #${detections.length + 1} (Z=${meanZ.toFixed(1)}, ${clusterConfidence}%)`,
      severity,
      estimatedWidthCm,
      estimatedDepthCm: depthResult.estimatedDepthCm,
      estimatedAreaSqCm,
      depthClass: depthResult.depthClass,
      depthConfidenceRange: depthResult.details.confidenceRangeCm,
      depthEstimationDetails: depthResult.details,
      patchCount: members.length,
      mahalanobisZ: Number(meanZ.toFixed(2)),
      luminanceDropPct,
      roughnessScorePct,
      hazardDescription: `DBSCAN cluster (${members.length} patches, Mahalanobis Z=${meanZ.toFixed(
        2
      )}), Shannon Entropy H=${depthResult.details.shannonEntropyBits} bits (${depthResult.details.entropyRatio}x baseline), Relative Scale=${depthResult.details.relativeScaleIndex}.`,
      box,
    });
  }

  detections.sort((a, b) => {
    const areaA = (a.box.xMax - a.box.xMin) * (a.box.yMax - a.box.yMin);
    const areaB = (b.box.xMax - b.box.xMin) * (b.box.yMax - b.box.yMin);
    return areaB - areaA;
  });
  const topDetections = detections.slice(0, 5);

  let overlayDataUrl = '';
  if (renderOverlay && typeof document !== 'undefined' && params.overlayMode !== 'detections') {
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = ANALYSIS_WIDTH;
    overlayCanvas.height = ANALYSIS_HEIGHT;
    const oCtx = overlayCanvas.getContext('2d');

    if (oCtx) {
      const outImg = oCtx.createImageData(ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
      const outData = outImg.data;

      for (const p of patches) {
        if (!p.inRoadZone) continue;

        let r = 0;
        let g = 0;
        let b = 0;
        let alpha = 0;

        if (params.overlayMode === 'kmeans') {
          const colorTuple =
            p.clusterId === roadClusterId
              ? CLUSTER_COLORS[0]
              : p.clusterId === anomalyClusterId
              ? CLUSTER_COLORS[2]
              : CLUSTER_COLORS[(p.clusterId + 1) % CLUSTER_COLORS.length];
          r = colorTuple[0];
          g = colorTuple[1];
          b = colorTuple[2];
          alpha = p.clusterId === roadClusterId ? 55 : 135;
        } else if (params.overlayMode === 'heatmap') {
          const norm = Math.min(1, Math.max(0, (p.anomalyZ - 0.8) / 3.4));
          if (norm > 0.15) {
            r = Math.round(255 * Math.min(1, norm * 1.6));
            g = Math.round(210 * Math.max(0, 1 - Math.abs(norm - 0.5) * 1.8));
            b = Math.round(60 * (1 - norm));
            alpha = Math.round(45 + norm * 165);
          }
        } else if (params.overlayMode === 'dbscan') {
          if (p.dbscanClusterId >= 0) {
            r = 234;
            g = 88;
            b = 12;
            alpha = 155;
          } else if (p.isAnomaly) {
            r = 148;
            g = 163;
            b = 184;
            alpha = 95;
          }
        }

        if (alpha > 0) {
          const startX = p.gx * PATCH_SIZE;
          const startY = p.gy * PATCH_SIZE;
          for (let py = 0; py < PATCH_SIZE; py++) {
            for (let px = 0; px < PATCH_SIZE; px++) {
              const x = startX + px;
              const y = startY + py;
              const idx = (y * ANALYSIS_WIDTH + x) * 4;
              if (params.overlayMode === 'dbscan' && p.dbscanClusterId < 0) {
                const edgeBoost = Math.round(grad[y * ANALYSIS_WIDTH + x] * 255);
                outData[idx] = Math.max(r, edgeBoost);
                outData[idx + 1] = Math.max(g, edgeBoost);
                outData[idx + 2] = Math.max(b, edgeBoost);
                outData[idx + 3] = Math.max(alpha, edgeBoost > 60 ? 150 : 40);
              } else {
                outData[idx] = r;
                outData[idx + 1] = g;
                outData[idx + 2] = b;
                outData[idx + 3] = alpha;
              }
            }
          }
        }
      }

      oCtx.putImageData(outImg, 0, 0);
      overlayDataUrl = overlayCanvas.toDataURL('image/png');
    }
  }

  const potholeDetected = topDetections.length > 0;
  const severityOrder: Record<SeverityLevel, number> = {
    None: 0,
    Low: 1,
    Moderate: 2,
    Severe: 3,
    Critical: 4,
  };

  let overallSeverity: SeverityLevel = 'None';
  for (const d of topDetections) {
    if (severityOrder[d.severity] > severityOrder[overallSeverity]) {
      overallSeverity = d.severity;
    }
  }

  const maxAnomZ = roadPatches.reduce((m, p) => Math.max(m, p.anomalyZ), 0);
  const overallConfidence = potholeDetected
    ? Math.round(Math.min(97, Math.max(66, 60 + (maxAnomZ - zThreshold) * 14)))
    : Math.round(Math.min(96, Math.max(76, 93 - Math.max(0, maxAnomZ - 1.3) * 10)));

  const roadSurfaceType =
    statL.med < 0.35
      ? 'Dark Asphalt Pavement'
      : statL.med < 0.55
      ? 'Weathered Asphalt Roadway'
      : 'Light Concrete / Wet Surface';

  const maxWidth = potholeDetected
    ? Math.max(...topDetections.map((d) => d.estimatedWidthCm))
    : 0;
  const maxDepth = potholeDetected
    ? Math.max(...topDetections.map((d) => d.estimatedDepthCm))
    : 0;

  const formalComplaintTitle = potholeDetected
    ? `${overallSeverity} Roadway Pothole Detected (${topDetections.length} Cluster${
        topDetections.length > 1 ? 's' : ''
      }, ~${maxWidth} cm)`
    : 'Clear Roadway Segment';

  const formalComplaintDescription = potholeDetected
    ? `Unsupervised computer vision (K-Means++ K=${params.kClusters}, Mahalanobis anomaly Z>=${zThreshold.toFixed(
        2
      )}, DBSCAN spatial clustering) isolated ${topDetections.length} pavement failure cluster${
        topDetections.length > 1 ? 's' : ''
      } against the learned ${roadSurfaceType.toLowerCase()} baseline. Largest defect spans approx. ${maxWidth} cm wide by ${maxDepth} cm deep.`
    : `No anomalous pavement depression clusters exceeded Mahalanobis threshold Z=${zThreshold.toFixed(
        2
      )}.`;

  const inferenceMs = Number((performance.now() - startTime).toFixed(1));
  const anomalousPatchCount = roadPatches.filter((p) => p.isAnomaly).length;

  return {
    detection: {
      potholeDetected,
      overallSeverity,
      confidence: overallConfidence,
      roadSurfaceType,
      detections: topDetections,
      formalComplaintTitle,
      formalComplaintDescription,
      recommendedDepartment: 'Department of Transportation — Roadway Maintenance',
      analyzedAt: new Date().toISOString(),
    },
    metrics: {
      inferenceMs,
      gridCols: cols,
      gridRows: rows,
      totalPatches: patches.length,
      roadPatches: roadPatches.length,
      kClusters: params.kClusters,
      kmeansIterations: kmeans.iterations,
      wcss: Number(kmeans.wcss.toFixed(1)),
      silhouetteScore: Number(silhouette.toFixed(2)),
      anomalyThresholdZ: zThreshold,
      anomalousPatchCount,
      dbscanClustersFound: topDetections.length,
      noisePatchesPruned: dbscanRes.noiseCount,
      baselineLuminanceMean: Number(statL.med.toFixed(2)),
      baselineRoughnessMean: Number((statGrad.med + statStd.med).toFixed(2)),
    },
    clusters,
    overlayDataUrl,
  };
}
