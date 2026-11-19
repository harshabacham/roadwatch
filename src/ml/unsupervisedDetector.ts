import type {
  BoundingBox,
  DetectionResult,
  PotholeDetectionItem,
  SeverityLevel,
  TrainedPotholeModel,
} from '../types.ts';
import {
  computeBaselineRoadEntropy,
  estimatePotholeDepthUnsupervised,
} from './depthEstimator.ts';

export const POTHOLE_FEATURE_NAMES = [
  'meanZ',
  'logRimRoughRatio',
  'logRimCornerRatio',
  'interiorLumDrop',
  'brightPaintExcess',
  'isotropy',
  'entropyRatio',
  'meanIsolation',
  'fillDensity',
  'cavityInteraction',
] as const;

export interface CandidateRegionSample {
  features: number[];
  box: BoundingBox;
  patchCount: number;
  meanZ: number;
  isWaterlogged: boolean;
}

let activeTrainedModel: TrainedPotholeModel | null = null;

export function setTrainedClassifierModel(
  model: TrainedPotholeModel | null
): void {
  activeTrainedModel = model;
}

export function getTrainedClassifierModel(): TrainedPotholeModel | null {
  return activeTrainedModel;
}

export function scoreFeatureVectorWithModel(
  features: number[],
  model: TrainedPotholeModel
): number {
  let logit = model.bias;
  const len = Math.min(
    features.length,
    model.weights.length,
    model.featureMeans.length,
    model.featureStds.length
  );
  for (let i = 0; i < len; i++) {
    const std = Math.max(1e-5, model.featureStds[i]);
    const xNorm = (features[i] - model.featureMeans[i]) / std;
    logit += model.weights[i] * xNorm;
  }
  if (logit > 30) return 1;
  if (logit < -30) return 0;
  return 1 / (1 + Math.exp(-logit));
}

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
  trainedFrames: number;
  gmmLogLikelihood: number;
  isolationScoreMean: number;
}

export interface UnsupervisedAnalysisOutput {
  detection: DetectionResult;
  metrics: UnsupervisedMetrics;
  clusters: ClusterProfile[];
  overlayDataUrl: string;
}

/**
 * Online Unsupervised Road Surface Baseline Model.
 * Learns normal asphalt appearance (mean & diagonal covariance via Welford EMA)
 * across clean road frames so illumination/pavement shifts adapt over time.
 */
export interface LearnedRoadBaseline {
  framesTrained: number;
  meanL: number;
  stdL: number;
  meanRough: number;
  stdRough: number;
  meanCorner: number;
  stdCorner: number;
  meanChroma: number;
  entropyBits: number;
}

const DEFAULT_BASELINE: LearnedRoadBaseline = {
  framesTrained: 0,
  meanL: 0.42,
  stdL: 0.08,
  meanRough: 0.05,
  stdRough: 0.03,
  meanCorner: 0.012,
  stdCorner: 0.008,
  meanChroma: 0.05,
  entropyBits: 1.6,
};

let learnedRoadBaseline: LearnedRoadBaseline = { ...DEFAULT_BASELINE };

export function getLearnedRoadBaseline(): LearnedRoadBaseline {
  return { ...learnedRoadBaseline };
}

export function resetLearnedRoadBaseline(): LearnedRoadBaseline {
  learnedRoadBaseline = { ...DEFAULT_BASELINE };
  return getLearnedRoadBaseline();
}

interface PatchFeature {
  gx: number;
  gy: number;
  idx: number;
  inRoadZone: boolean;
  roadLikelihood: number;
  meanL: number;
  stdL: number;
  meanGrad: number;
  peakGrad: number;
  coherence: number; // Structure tensor anisotropy 0..1 (1 = straight line, 0 = isotropic rubble)
  cornerEnergy: number; // Minor eigenvalue of structure tensor (Shi-Tomasi cornerness)
  isotropicRough: number; // meanGrad * (1 - 0.78 * coherence) + stdL
  chromaShift: number;
  greenExcess: number;
  isVegetation: boolean;
  warmth: number;
  rg: number;
  yb: number;
  bgL: number;
  bgRough: number;
  bgCorner: number;
  bgStd: number;
  bgRg: number;
  bgYb: number;
  lumDepression: number;
  lumContrast: number;
  dogConcavity: number;
  textureExcess: number;
  cornerRatio: number;
  colorAnomaly: number;
  clusterId: number;
  gmmPosteriorRoad: number;
  mahalanobisZ: number;
  isolationScore: number;
  anomalyZ: number;
  isAnomaly: boolean;
  isWaterSeed: boolean;
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

interface SensitivityConfig {
  zThreshold: number;
  minPatchIsolation: number;
  minClusterMeanZ: number;
  minClusterPeakZ: number;
  minLumDepression: number;
  minRoughRatio: number;
  minCornerEnergy: number;
  minRimRoughnessRatio: number;
}

function getSensitivityConfig(preset: SensitivityPreset): SensitivityConfig {
  switch (preset) {
    case 'conservative':
      return {
        zThreshold: 2.6,
        minPatchIsolation: 0.495,
        minClusterMeanZ: 2.8,
        minClusterPeakZ: 3.3,
        minLumDepression: 0.072,
        minRoughRatio: 1.32,
        minCornerEnergy: 0.017,
        minRimRoughnessRatio: 1.25,
      };
    case 'sensitive':
      return {
        zThreshold: 2.15,
        minPatchIsolation: 0.48,
        minClusterMeanZ: 2.35,
        minClusterPeakZ: 2.85,
        minLumDepression: 0.055,
        minRoughRatio: 1.28,
        minCornerEnergy: 0.013,
        minRimRoughnessRatio: 1.25,
      };
    case 'balanced':
    default:
      return {
        zThreshold: 2.15,
        minPatchIsolation: 0.48,
        minClusterMeanZ: 2.3,
        minClusterPeakZ: 2.85,
        minLumDepression: 0.055,
        minRoughRatio: 1.28,
        minCornerEnergy: 0.013,
        minRimRoughnessRatio: 1.22,
      };
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

function quantile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) return 0;
  const pos = (sortedValues.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sortedValues[base + 1] !== undefined) {
    return (
      sortedValues[base] + rest * (sortedValues[base + 1] - sortedValues[base])
    );
  }
  return sortedValues[base];
}

/**
 * Deterministic PRNG (Mulberry32) so unsupervised Isolation Forest & EM
 * produce stable, reproducible scores on identical frames.
 */
function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic K-Means++ initialization + Lloyd's iterations on standardized vectors.
 */
function runKMeansPlusPlus(
  vectors: number[][],
  k: number,
  maxIter = 20
): {
  assignments: number[];
  centroids: number[][];
  iterations: number;
  wcss: number;
} {
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
  const centroids: number[][] = [
    vectors[norms[Math.floor(n * 0.3)].idx].slice(),
  ];

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

/**
 * Unsupervised Diagonal Gaussian Mixture Model (EM refinement initialized from K-Means++).
 * Computes soft posterior probabilities gamma[i][k] and average log-likelihood.
 */
function runGmmEm(
  vectors: number[][],
  initAssignments: number[],
  initCentroids: number[][],
  k: number,
  emSteps = 6
): {
  assignments: number[];
  responsibilities: number[][];
  means: number[][];
  variances: number[][];
  weights: number[];
  avgLogLikelihood: number;
} {
  const n = vectors.length;
  if (n === 0 || k === 0) {
    return {
      assignments: [],
      responsibilities: [],
      means: [],
      variances: [],
      weights: [],
      avgLogLikelihood: 0,
    };
  }
  const dims = vectors[0].length;
  const effectiveK = Math.min(k, initCentroids.length);
  const means = initCentroids.map((c) => c.slice());
  const variances = Array.from({ length: effectiveK }, () =>
    new Array<number>(dims).fill(0.6)
  );
  const weights = new Array<number>(effectiveK).fill(1 / effectiveK);

  // Initialize variances and mixing weights from K-Means++ partition
  const counts = new Int32Array(effectiveK);
  const varAcc = Array.from({ length: effectiveK }, () => new Float64Array(dims));
  for (let i = 0; i < n; i++) {
    const c = initAssignments[i];
    counts[c]++;
    for (let d = 0; d < dims; d++) {
      const diff = vectors[i][d] - means[c][d];
      varAcc[c][d] += diff * diff;
    }
  }
  for (let c = 0; c < effectiveK; c++) {
    weights[c] = Math.max(0.02, counts[c] / n);
    for (let d = 0; d < dims; d++) {
      variances[c][d] = Math.max(0.18, varAcc[c][d] / Math.max(1, counts[c]));
    }
  }

  const responsibilities: number[][] = Array.from({ length: n }, () =>
    new Array<number>(effectiveK).fill(0)
  );
  const logProbs = new Float64Array(effectiveK);
  let avgLogLikelihood = 0;

  for (let step = 0; step < emSteps; step++) {
    let totalLL = 0;
    // E-Step
    for (let i = 0; i < n; i++) {
      const v = vectors[i];
      let maxLogP = -Infinity;
      for (let c = 0; c < effectiveK; c++) {
        let logDet = 0;
        let mahal = 0;
        for (let d = 0; d < dims; d++) {
          const varD = variances[c][d];
          logDet += Math.log(2 * Math.PI * varD);
          const diff = v[d] - means[c][d];
          mahal += (diff * diff) / varD;
        }
        const lp = Math.log(weights[c] + 1e-8) - 0.5 * (logDet + mahal);
        logProbs[c] = lp;
        if (lp > maxLogP) maxLogP = lp;
      }

      let sumExp = 0;
      for (let c = 0; c < effectiveK; c++) {
        sumExp += Math.exp(logProbs[c] - maxLogP);
      }
      const logNorm = maxLogP + Math.log(sumExp + 1e-12);
      totalLL += logNorm;

      for (let c = 0; c < effectiveK; c++) {
        responsibilities[i][c] = Math.exp(logProbs[c] - logNorm);
      }
    }
    avgLogLikelihood = totalLL / n;

    // M-Step
    for (let c = 0; c < effectiveK; c++) {
      let nk = 0;
      const newMean = new Float64Array(dims);
      for (let i = 0; i < n; i++) {
        const gamma = responsibilities[i][c];
        nk += gamma;
        for (let d = 0; d < dims; d++) {
          newMean[d] += gamma * vectors[i][d];
        }
      }
      weights[c] = Math.max(0.015, nk / n);
      if (nk > 1e-4) {
        for (let d = 0; d < dims; d++) {
          means[c][d] = newMean[d] / nk;
        }
        const newVar = new Float64Array(dims);
        for (let i = 0; i < n; i++) {
          const gamma = responsibilities[i][c];
          for (let d = 0; d < dims; d++) {
            const diff = vectors[i][d] - means[c][d];
            newVar[d] += gamma * diff * diff;
          }
        }
        for (let d = 0; d < dims; d++) {
          variances[c][d] = Math.max(0.18, newVar[d] / nk);
        }
      }
    }
  }

  const assignments = responsibilities.map((row) => {
    let bestC = 0;
    let bestP = -1;
    for (let c = 0; c < effectiveK; c++) {
      if (row[c] > bestP) {
        bestP = row[c];
        bestC = c;
      }
    }
    return bestC;
  });

  return {
    assignments,
    responsibilities,
    means,
    variances,
    weights,
    avgLogLikelihood,
  };
}

/**
 * Unsupervised Isolation Forest (Liu, Ting, Zhou) for multi-dimensional anomaly scoring.
 * Anomalies require fewer random axis-aligned splits to isolate than dense road patches.
 */
interface ITreeNode {
  leafSize?: number;
  splitAttr?: number;
  splitVal?: number;
  left?: ITreeNode;
  right?: ITreeNode;
}

function averagePathLengthC(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n;
}

function buildITree(
  samples: number[][],
  depth: number,
  maxDepth: number,
  dims: number,
  rng: () => number
): ITreeNode {
  if (depth >= maxDepth || samples.length <= 1) {
    return { leafSize: samples.length };
  }

  const attr = Math.floor(rng() * dims);
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i][attr];
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }

  if (maxVal - minVal < 1e-6) {
    return { leafSize: samples.length };
  }

  const splitVal = minVal + rng() * (maxVal - minVal);
  const leftSamples: number[][] = [];
  const rightSamples: number[][] = [];
  for (let i = 0; i < samples.length; i++) {
    if (samples[i][attr] < splitVal) leftSamples.push(samples[i]);
    else rightSamples.push(samples[i]);
  }

  return {
    splitAttr: attr,
    splitVal,
    left: buildITree(leftSamples, depth + 1, maxDepth, dims, rng),
    right: buildITree(rightSamples, depth + 1, maxDepth, dims, rng),
  };
}

function pathLengthITree(vec: number[], node: ITreeNode, depth: number): number {
  if (node.leafSize !== undefined || node.splitAttr === undefined) {
    return depth + averagePathLengthC(node.leafSize ?? 1);
  }
  if (vec[node.splitAttr] < (node.splitVal ?? 0)) {
    return node.left ? pathLengthITree(vec, node.left, depth + 1) : depth + 1;
  }
  return node.right ? pathLengthITree(vec, node.right, depth + 1) : depth + 1;
}

function computeIsolationForestScores(vectors: number[][]): number[] {
  const n = vectors.length;
  if (n < 8) return new Array<number>(n).fill(0.45);

  const dims = vectors[0].length;
  const numTrees = 24;
  const subSampleSize = Math.min(96, n);
  const maxDepth = Math.ceil(Math.log2(subSampleSize));
  const cNorm = Math.max(1, averagePathLengthC(subSampleSize));
  const rng = createSeededRng(0x9e3779b9 ^ n);

  const trees: ITreeNode[] = [];
  for (let t = 0; t < numTrees; t++) {
    const sub: number[][] = [];
    for (let s = 0; s < subSampleSize; s++) {
      const idx = Math.floor(rng() * n);
      sub.push(vectors[idx]);
    }
    trees.push(buildITree(sub, 0, maxDepth, dims, rng));
  }

  const scores = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let sumH = 0;
    for (let t = 0; t < numTrees; t++) {
      sumH += pathLengthITree(vectors[i], trees[t], 0);
    }
    const meanH = sumH / numTrees;
    scores[i] = Math.pow(2, -meanH / cNorm);
  }
  return scores;
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

  const noiseCount = anomalyPatches.filter(
    (p) => p.dbscanClusterId === -1
  ).length;
  return { clusterCount: clusterId, noiseCount };
}

/**
 * Helper for 2D Integral Image rectangular sum over patch grid [x0..x1, y0..y1] inclusive.
 */
function buildIntegralGrid(
  values: Float64Array,
  cols: number,
  rows: number
): Float64Array {
  const stride = cols + 1;
  const sat = new Float64Array((rows + 1) * stride);
  for (let y = 0; y < rows; y++) {
    let rowSum = 0;
    const srcOff = y * cols;
    const dstOff = (y + 1) * stride;
    const prevOff = y * stride;
    for (let x = 0; x < cols; x++) {
      rowSum += values[srcOff + x];
      sat[dstOff + x + 1] = sat[prevOff + x + 1] + rowSum;
    }
  }
  return sat;
}

function queryIntegralRect(
  sat: Float64Array,
  cols: number,
  rows: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
): { sum: number; area: number } {
  const cx0 = Math.max(0, x0);
  const cy0 = Math.max(0, y0);
  const cx1 = Math.min(cols - 1, x1);
  const cy1 = Math.min(rows - 1, y1);
  if (cx1 < cx0 || cy1 < cy0) return { sum: 0, area: 0 };

  const stride = cols + 1;
  const sum =
    sat[(cy1 + 1) * stride + (cx1 + 1)] -
    sat[cy0 * stride + (cx1 + 1)] -
    sat[(cy1 + 1) * stride + cx0] +
    sat[cy0 * stride + cx0];
  const area = (cx1 - cx0 + 1) * (cy1 - cy0 + 1);
  return { sum, area };
}

/**
 * Explicitly trains / updates the online unsupervised road-surface baseline model
 * from a clean road image/video frame (used by manual "Train Normal Road" calibration
 * and automatically during clear frames).
 */
export function trainRoadBaselineFromSource(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
): LearnedRoadBaseline | null {
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
  return trainRoadBaselineFromPixels(imgData.data);
}

export function trainRoadBaselineFromPixels(
  data: Uint8ClampedArray
): LearnedRoadBaseline {
  const output = analyzePixelsUnsupervised(
    data,
    {
      kClusters: 4,
      sensitivity: 'balanced',
      dbscanMinPts: 4,
      overlayMode: 'detections',
    },
    false,
    true
  );
  return output ? getLearnedRoadBaseline() : learnedRoadBaseline;
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
  return analyzePixelsUnsupervised(imgData.data, params, true, false);
}

export function analyzePixelsUnsupervised(
  data: Uint8ClampedArray,
  params: UnsupervisedParams,
  renderOverlay = false,
  forceBaselineTrain = false
): UnsupervisedAnalysisOutput {
  return analyzePixelsInternal(data, params, renderOverlay, forceBaselineTrain)
    .output;
}

/**
 * Extracts 10D candidate region feature vectors in permissive mining mode
 * so the backend supervised trainer can learn from both true potholes and
 * hard negatives (shadows, lane markings, tar seams, clean coarse asphalt).
 */
export function extractCandidateRegionsForTraining(
  data: Uint8ClampedArray
): CandidateRegionSample[] {
  const { minedCandidates } = analyzePixelsInternal(
    data,
    {
      kClusters: 4,
      sensitivity: 'sensitive',
      dbscanMinPts: 3,
      overlayMode: 'detections',
    },
    false,
    false,
    true
  );
  return minedCandidates;
}

function analyzePixelsInternal(
  data: Uint8ClampedArray,
  params: UnsupervisedParams,
  renderOverlay = false,
  forceBaselineTrain = false,
  miningMode = false
): {
  output: UnsupervisedAnalysisOutput;
  minedCandidates: CandidateRegionSample[];
} {
  const startTime = performance.now();

  const totalPixels = ANALYSIS_WIDTH * ANALYSIS_HEIGHT;
  const lum = new Float32Array(totalPixels);
  const chroma = new Float32Array(totalPixels);
  const warmth = new Float32Array(totalPixels);
  const rgArr = new Float32Array(totalPixels);
  const ybArr = new Float32Array(totalPixels);
  const grad = new Float32Array(totalPixels);
  const gradX = new Float32Array(totalPixels);
  const gradY = new Float32Array(totalPixels);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    lum[p] = 0.299 * r + 0.587 * g + 0.114 * b;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    chroma[p] = maxC - minC;
    warmth[p] = r - b;
    rgArr[p] = r - g;
    ybArr[p] = 0.5 * (r + g) - b;
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
      gradX[p] = gx * 0.25;
      gradY[p] = gy * 0.25;
      grad[p] = Math.min(1, Math.sqrt(gx * gx + gy * gy) * 0.45);
    }
  }

  const cols = Math.floor(ANALYSIS_WIDTH / PATCH_SIZE);
  const rows = Math.floor(ANALYSIS_HEIGHT / PATCH_SIZE);
  const numPatches = cols * rows;
  const patches: PatchFeature[] = [];

  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const startX = gx * PATCH_SIZE;
      const startY = gy * PATCH_SIZE;
      const count = PATCH_SIZE * PATCH_SIZE;

      let sumL = 0;
      let sumL2 = 0;
      let sumR = 0;
      let sumGreen = 0;
      let sumB = 0;
      let sumGrad = 0;
      let peakG = 0;
      let sumChroma = 0;
      let sumWarmth = 0;
      let sumRg = 0;
      let sumYb = 0;
      let jxx = 0;
      let jyy = 0;
      let jxy = 0;

      for (let py = 0; py < PATCH_SIZE; py++) {
        const rowOff = (startY + py) * ANALYSIS_WIDTH + startX;
        for (let px = 0; px < PATCH_SIZE; px++) {
          const p = rowOff + px;
          const lVal = lum[p];
          const gVal = grad[p];
          const gxVal = gradX[p];
          const gyVal = gradY[p];
          const pByte = p * 4;

          sumL += lVal;
          sumL2 += lVal * lVal;
          sumR += data[pByte] / 255;
          sumGreen += data[pByte + 1] / 255;
          sumB += data[pByte + 2] / 255;
          sumGrad += gVal;
          if (gVal > peakG) peakG = gVal;
          sumChroma += chroma[p];
          sumWarmth += warmth[p];
          sumRg += rgArr[p];
          sumYb += ybArr[p];

          jxx += gxVal * gxVal;
          jyy += gyVal * gyVal;
          jxy += gxVal * gyVal;
        }
      }

      const meanL = sumL / count;
      const meanR = sumR / count;
      const meanGreen = sumGreen / count;
      const meanB = sumB / count;
      const greenExcess = Math.max(0, meanGreen - Math.max(meanR, meanB));
      const isVegetation =
        meanGreen - meanR > 0.022 && meanGreen - meanB > 0.025;
      const variance = Math.max(0, sumL2 / count - meanL * meanL);
      const stdL = Math.sqrt(variance);
      const meanGrad = sumGrad / count;

      // Structure tensor eigenvalues for orientation coherence & corner energy
      jxx /= count;
      jyy /= count;
      jxy /= count;
      const trace = jxx + jyy;
      const disc = Math.sqrt(Math.max(0, (jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy));
      const lambda1 = 0.5 * (trace + disc);
      const lambda2 = Math.max(0, 0.5 * (trace - disc));
      const coherence = trace > 1e-4 ? (lambda1 - lambda2) / (trace + 1e-4) : 0;
      const cornerEnergy = Math.sqrt(lambda2);

      // Isotropic roughness suppresses straight lane lines, curbs, and straight shadows
      const isotropicRough = meanGrad * (1 - 0.78 * coherence) + stdL * 0.85;

      const normY = (gy + 0.5) / rows;
      const inRoadZone = normY >= 0.32 && !isVegetation;

      patches.push({
        gx,
        gy,
        idx: gy * cols + gx,
        inRoadZone,
        roadLikelihood: inRoadZone ? 1 : 0,
        meanL,
        stdL,
        meanGrad,
        peakGrad: peakG,
        coherence,
        cornerEnergy,
        isotropicRough,
        chromaShift: sumChroma / count,
        greenExcess,
        isVegetation,
        warmth: sumWarmth / count,
        rg: sumRg / count,
        yb: sumYb / count,
        bgL: meanL,
        bgRough: isotropicRough,
        bgCorner: cornerEnergy,
        bgStd: stdL,
        bgRg: sumRg / count,
        bgYb: sumYb / count,
        lumDepression: 0,
        lumContrast: 0,
        dogConcavity: 0,
        textureExcess: 0,
        cornerRatio: 1,
        colorAnomaly: 0,
        clusterId: 0,
        gmmPosteriorRoad: 0.5,
        mahalanobisZ: 0,
        isolationScore: 0.45,
        anomalyZ: 0,
        isAnomaly: false,
        isWaterSeed: false,
        dbscanClusterId: -1,
      });
    }
  }

  // Build Summed-Area Tables for O(1) Annular Ring Context (outer=11 minus inner=4)
  // Using an annulus prevents large potholes from contaminating their own background baseline!
  const maskGrid = new Float64Array(numPatches);
  const lumGrid = new Float64Array(numPatches);
  const roughGrid = new Float64Array(numPatches);
  const cornerGrid = new Float64Array(numPatches);
  const stdGrid = new Float64Array(numPatches);
  const rgGrid = new Float64Array(numPatches);
  const ybGrid = new Float64Array(numPatches);

  for (let i = 0; i < numPatches; i++) {
    const p = patches[i];
    // Exclude bright white/glare paint and green vegetation from road background
    const isRoadCandidatePixel =
      p.inRoadZone && !p.isVegetation && p.meanL < 0.72;
    const w = isRoadCandidatePixel ? 1 : 0;
    maskGrid[i] = w;
    lumGrid[i] = w * p.meanL;
    roughGrid[i] = w * p.isotropicRough;
    cornerGrid[i] = w * p.cornerEnergy;
    stdGrid[i] = w * p.stdL;
    rgGrid[i] = w * p.rg;
    ybGrid[i] = w * p.yb;
  }

  const satMask = buildIntegralGrid(maskGrid, cols, rows);
  const satLum = buildIntegralGrid(lumGrid, cols, rows);
  const satRough = buildIntegralGrid(roughGrid, cols, rows);
  const satCorner = buildIntegralGrid(cornerGrid, cols, rows);
  const satStd = buildIntegralGrid(stdGrid, cols, rows);
  const satRg = buildIntegralGrid(rgGrid, cols, rows);
  const satYb = buildIntegralGrid(ybGrid, cols, rows);

  const outerR = 11;
  const innerR = 4;
  const roadStartGy = Math.floor(rows * 0.32);

  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const p = patches[gy * cols + gx];

      const outM = queryIntegralRect(
        satMask,
        cols,
        rows,
        gx - outerR,
        Math.max(roadStartGy, gy - outerR),
        gx + outerR,
        gy + outerR
      );
      const inM = queryIntegralRect(
        satMask,
        cols,
        rows,
        gx - innerR,
        Math.max(roadStartGy, gy - innerR),
        gx + innerR,
        gy + innerR
      );

      let ringCount = outM.sum - inM.sum;
      let useInnerFallback = false;
      if (ringCount < 12) {
        ringCount = outM.sum;
        useInnerFallback = true;
      }

      if (ringCount > 0) {
        const sub = (sat: Float64Array) => {
          const o = queryIntegralRect(
            sat,
            cols,
            rows,
            gx - outerR,
            Math.max(roadStartGy, gy - outerR),
            gx + outerR,
            gy + outerR
          ).sum;
          if (useInnerFallback) return o / ringCount;
          const inn = queryIntegralRect(
            sat,
            cols,
            rows,
            gx - innerR,
            Math.max(roadStartGy, gy - innerR),
            gx + innerR,
            gy + innerR
          ).sum;
          return (o - inn) / ringCount;
        };

        p.bgL = sub(satLum);
        p.bgRough = sub(satRough);
        p.bgCorner = sub(satCorner);
        p.bgStd = sub(satStd);
        p.bgRg = sub(satRg);
        p.bgYb = sub(satYb);
      }

      p.lumDepression = Math.max(0, p.bgL - p.meanL);
      p.lumContrast = Math.abs(p.meanL - p.bgL);
      p.textureExcess = Math.max(0, p.isotropicRough - p.bgRough);
      p.cornerRatio = (p.cornerEnergy + 0.008) / (p.bgCorner + 0.008);
      p.colorAnomaly = Math.hypot(p.rg - p.bgRg, p.yb - p.bgYb);
      // DoG concavity: dark depression paired with local edge/corner contrast
      p.dogConcavity =
        p.lumDepression * (1 + Math.min(2.2, p.cornerEnergy * 35));
    }
  }

  // Unsupervised Road-Plane Estimation from bottom-center asphalt seed
  const seedPatches = patches.filter(
    (p) =>
      !p.isVegetation &&
      p.gy >= Math.floor(rows * 0.62) &&
      p.gx >= Math.floor(cols * 0.2) &&
      p.gx <= Math.floor(cols * 0.8) &&
      p.meanL < 0.72
  );
  const seedL = seedPatches.length > 0 ? median(seedPatches.map((p) => p.meanL)) : 0.4;
  const seedChroma =
    seedPatches.length > 0 ? median(seedPatches.map((p) => p.chromaShift)) : 0.05;

  for (const p of patches) {
    if (!p.inRoadZone || p.isVegetation) {
      p.roadLikelihood = 0;
      continue;
    }
    const yNorm = (p.gy + 0.5) / rows;
    const colorDist = Math.abs(p.chromaShift - seedChroma);
    const lumDist = Math.abs(p.meanL - seedL);
    // Upper zone (yNorm < 0.56) on edges or colorful roadside structures is penalized
    const horizonPenalty =
      (yNorm < 0.44 && (colorDist > 0.14 || lumDist > 0.25)) ||
      (yNorm < 0.56 && p.greenExcess > 0.012)
        ? 0.55
        : 0;
    p.roadLikelihood = Math.max(
      0,
      Math.min(1, 1 - colorDist * 1.4 - horizonPenalty)
    );
  }

  const roadPatches = patches.filter((p) => p.inRoadZone);

  // Build 6D standardized feature matrix for Unsupervised Clustering & Isolation Forest
  const rawVectors = roadPatches.map((p) => [
    p.meanL,
    Math.log1p(12 * p.isotropicRough),
    Math.log1p(40 * p.cornerEnergy),
    p.lumDepression * 4.5,
    p.dogConcavity * 3.5,
    p.colorAnomaly * 3.0 + p.chromaShift * 1.2,
  ]);

  const dims = 6;
  const featMeans = new Float64Array(dims);
  const featStds = new Float64Array(dims);

  for (const v of rawVectors) {
    for (let d = 0; d < dims; d++) featMeans[d] += v[d];
  }
  for (let d = 0; d < dims; d++) {
    featMeans[d] /= Math.max(1, rawVectors.length);
  }

  for (const v of rawVectors) {
    for (let d = 0; d < dims; d++) {
      const diff = v[d] - featMeans[d];
      featStds[d] += diff * diff;
    }
  }
  for (let d = 0; d < dims; d++) {
    featStds[d] = Math.max(
      0.04,
      Math.sqrt(featStds[d] / Math.max(1, rawVectors.length))
    );
  }

  const stdVectors = rawVectors.map((v) =>
    v.map((val, d) => (val - featMeans[d]) / featStds[d])
  );

  // 1. K-Means++ Clustering
  const kmeans = runKMeansPlusPlus(stdVectors, params.kClusters);

  // 2. Diagonal GMM (Expectation-Maximization refinement)
  const gmm = runGmmEm(
    stdVectors,
    kmeans.assignments,
    kmeans.centroids,
    params.kClusters,
    5
  );

  // 3. Unsupervised Isolation Forest Anomaly Scoring
  const iForestScores = computeIsolationForestScores(stdVectors);

  const silhouette = computeSilhouette(
    stdVectors,
    gmm.assignments,
    params.kClusters
  );

  roadPatches.forEach((p, idx) => {
    p.clusterId = gmm.assignments[idx];
    p.isolationScore = iForestScores[idx];
  });

  const clusterStats = Array.from({ length: params.kClusters }, (_, cid) => {
    const members = roadPatches.filter((p) => p.clusterId === cid);
    const lowerMembers = members.filter((p) => p.gy >= rows * 0.52);
    const count = members.length;
    const meanL =
      count > 0 ? members.reduce((s, p) => s + p.meanL, 0) / count : 0;
    const meanRough =
      count > 0
        ? members.reduce((s, p) => s + p.isotropicRough, 0) / count
        : 0;
    const meanDep =
      count > 0 ? members.reduce((s, p) => s + p.lumDepression, 0) / count : 0;
    const meanCorner =
      count > 0 ? members.reduce((s, p) => s + p.cornerEnergy, 0) / count : 0;

    return {
      cid,
      members,
      count,
      lowerCount: lowerMembers.length,
      meanL,
      meanRough,
      meanDep,
      meanCorner,
    };
  });

  // Identify dominant planar asphalt cluster (high lower-frame presence, low roughness, low depression)
  const sortedForAsphalt = [...clusterStats]
    .filter((c) => c.count > 0)
    .sort((a, b) => {
      const aScore =
        a.lowerCount * (a.meanL < 0.68 ? 1.25 : 0.35) -
        a.meanRough * 55 -
        a.meanDep * 75 -
        a.meanCorner * 120;
      const bScore =
        b.lowerCount * (b.meanL < 0.68 ? 1.25 : 0.35) -
        b.meanRough * 55 -
        b.meanDep * 75 -
        b.meanCorner * 120;
      return bScore - aScore;
    });

  const roadClusterId = sortedForAsphalt[0]?.cid ?? 0;

  // Store GMM posterior probability of belonging to the normal road cluster
  roadPatches.forEach((p, idx) => {
    p.gmmPosteriorRoad = gmm.responsibilities[idx]?.[roadClusterId] ?? 0.5;
  });

  // Trimmed MCD-style Robust Baseline Statistics:
  // Fit ONLY on the cleanest 35% of road patches (lowest isotropic roughness + depression)
  // so even close-up large potholes (35-45% of frame) cannot contaminate the normal-road baseline!
  const rankedClean = [...roadPatches]
    .filter((p) => p.meanL < 0.72)
    .sort(
      (a, b) =>
        a.isotropicRough +
        a.lumDepression * 1.5 -
        (b.isotropicRough + b.lumDepression * 1.5)
    );
  const corePool =
    rankedClean.length >= 20
      ? rankedClean.slice(0, Math.max(20, Math.floor(rankedClean.length * 0.35)))
      : roadPatches;

  const coreL = corePool.map((p) => p.meanL).sort((a, b) => a - b);
  const coreRough = corePool.map((p) => p.isotropicRough).sort((a, b) => a - b);
  const coreCorner = corePool.map((p) => p.cornerEnergy).sort((a, b) => a - b);
  const coreDep = corePool.map((p) => p.lumDepression).sort((a, b) => a - b);
  const coreColor = corePool.map((p) => p.colorAnomaly).sort((a, b) => a - b);

  const frameBaselineL = quantile(coreL, 0.5);
  const frameBaselineRough = quantile(coreRough, 0.5);
  const frameBaselineCorner = quantile(coreCorner, 0.5);
  const frameBaselineDep = quantile(coreDep, 0.5);
  const frameBaselineColor = quantile(coreColor, 0.5);

  // Blend in learned online road baseline if previously calibrated (15% prior weight)
  const priorWeight = learnedRoadBaseline.framesTrained > 0 ? 0.15 : 0;
  const baseL =
    (1 - priorWeight) * frameBaselineL + priorWeight * learnedRoadBaseline.meanL;
  const baseRough =
    (1 - priorWeight) * frameBaselineRough +
    priorWeight * learnedRoadBaseline.meanRough;
  const baseCorner =
    (1 - priorWeight) * frameBaselineCorner +
    priorWeight * learnedRoadBaseline.meanCorner;

  // Calibrated sensor-noise scale floors prevent clean uniform roads from blowing up Z-scores!
  const scaleL = Math.max(
    0.095,
    (quantile(coreL, 0.84) - quantile(coreL, 0.16)) * 0.9
  );
  const scaleRough = Math.max(
    0.036,
    (quantile(coreRough, 0.84) - quantile(coreRough, 0.16)) * 1.05
  );
  const scaleCorner = Math.max(
    0.011,
    (quantile(coreCorner, 0.84) - quantile(coreCorner, 0.16)) * 1.05
  );
  const scaleDep = Math.max(
    0.038,
    (quantile(coreDep, 0.84) - quantile(coreDep, 0.16)) * 1.15
  );
  const scaleColor = Math.max(
    0.042,
    (quantile(coreColor, 0.84) - quantile(coreColor, 0.16)) * 1.15
  );

  const anomalyCluster = [...clusterStats]
    .filter((c) => c.cid !== roadClusterId && c.count > 0 && c.meanL < 0.65)
    .sort(
      (a, b) =>
        b.meanRough +
        b.meanDep * 2 +
        b.meanCorner * 3 -
        (a.meanRough + a.meanDep * 2 + a.meanCorner * 3)
    )[0];
  const anomalyClusterId = anomalyCluster?.cid ?? -1;

  const clusters: ClusterProfile[] = clusterStats.map((c, idx) => {
    let role = 'Surface Transition';
    if (c.cid === roadClusterId) role = 'Normal Road Surface';
    else if (c.cid === anomalyClusterId) role = 'Damaged / Cavity Area';
    else if (c.meanL > 0.6) role = 'Lane Lines / Glare';
    else if (c.meanRough > baseRough + scaleRough)
      role = 'Cracked Pavement Edges';

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

  const cfg = getSensitivityConfig(params.sensitivity);

  // Patch-level Unsupervised Ensemble Scoring (Trimmed Mahalanobis + iForest + GMM NLL)
  for (const p of roadPatches) {
    // 1. Reject clean painted white lane markings / bright glare lines (achromatic & smooth)
    const isBrightLaneMark =
      p.meanL > Math.max(0.68, baseL + 0.22) &&
      p.chromaShift < 0.1 &&
      p.lumDepression < 0.03 &&
      p.cornerEnergy < 0.024 &&
      (p.coherence > 0.42 || p.stdL < 0.085);

    if (isBrightLaneMark) {
      p.anomalyZ = 0;
      p.isAnomaly = false;
      continue;
    }

    // 2. Reject clean straight linear features (curb joints, straight lane edges, straight shadows)
    const isStraightLinearEdge =
      p.coherence > 0.72 &&
      p.stdL < 0.11 &&
      p.cornerEnergy < 0.018 &&
      p.lumDepression < 0.09;

    const zDep = Math.max(0, (p.lumDepression - frameBaselineDep) / scaleDep);
    const zRough = Math.max(0, (p.isotropicRough - baseRough) / scaleRough);
    const zCorner = Math.max(0, (p.cornerEnergy - baseCorner) / scaleCorner);
    const zColor = Math.max(
      0,
      (p.colorAnomaly - frameBaselineColor) / scaleColor
    );
    const zDarkShift = Math.max(0, (baseL - p.meanL) / scaleL);

    // Weighted Multivariate Mahalanobis Distance from trimmed road baseline
    const mDist = Math.sqrt(
      0.34 * zDep * zDep +
        0.26 * zRough * zRough +
        0.22 * zCorner * zCorner +
        0.1 * zDarkShift * zDarkShift +
        0.08 * zColor * zColor
    );
    p.mahalanobisZ = mDist;

    // Combine with Isolation Forest anomaly boost and GMM non-road posterior
    const iForestBoost =
      p.isolationScore > 0.51 ? 1 + (p.isolationScore - 0.51) * 1.45 : 0.92;
    const gmmNoveltyBoost = 1 + (1 - p.gmmPosteriorRoad) * 0.18;
    let ensembleZ = mDist * iForestBoost * gmmNoveltyBoost;

    if (p.clusterId === anomalyClusterId) {
      ensembleZ *= 1.08;
    }
    if (isStraightLinearEdge) {
      ensembleZ *= 0.62;
    }

    p.anomalyZ = ensembleZ;

    // Physical cavity / broken-aggregate gate:
    // Smooth uniform shadows have lumDepression > 0, BUT NO corner/roughness excess!
    // Real potholes have either:
    // (a) Broken rough aggregate / spalled edge (elevated isotropic rough + corner energy), OR
    // (b) Deep dark cavity occlusion accompanied by rim corner energy or texture contrast, OR
    // (c) Broken water-puddle lip (high local variance + step contrast + corner energy).
    const hasBrokenAggregate =
      p.isotropicRough > baseRough * cfg.minRoughRatio &&
      p.textureExcess >= 0.02 &&
      p.cornerEnergy >= cfg.minCornerEnergy;

    const hasDarkCavityWithEdge =
      p.lumDepression >= cfg.minLumDepression &&
      (p.cornerEnergy >= cfg.minCornerEnergy * 0.8 ||
        p.textureExcess >= 0.015 ||
        p.stdL >= 0.12 ||
        p.lumDepression >= cfg.minLumDepression * 1.75);

    const hasPuddleRimBreak =
      p.stdL >= 0.17 &&
      p.meanGrad >= 0.22 &&
      p.cornerEnergy >= cfg.minCornerEnergy * 1.15 &&
      p.lumContrast >= 0.11;

    const hasPhysicalEvidence =
      hasBrokenAggregate || hasDarkCavityWithEdge || hasPuddleRimBreak;

    const effectiveZThreshold = miningMode
      ? Math.min(1.65, cfg.zThreshold)
      : cfg.zThreshold;

    p.isAnomaly =
      p.roadLikelihood >= 0.45 &&
      ensembleZ >= effectiveZThreshold &&
      (miningMode ? true : hasPhysicalEvidence);
  }

  // Waterlogged Pothole Basin Closure:
  // A rain-filled pothole has a smooth reflective interior (different luminance from road)
  // bounded on >= 3 directions (or 2 opposite sides) by fractured/rough rim patches!
  const isRimBoundary = (q: PatchFeature) =>
    q.isAnomaly ||
    (q.inRoadZone &&
      q.roadLikelihood >= 0.45 &&
      q.anomalyZ >= 1.8 &&
      q.cornerEnergy >= cfg.minCornerEnergy * 0.8 &&
      q.isotropicRough >= baseRough * 1.12);

  const waterEnclosedSet = new Set<number>();
  for (let gy = roadStartGy + 1; gy < rows - 1; gy++) {
    for (let gx = 1; gx < cols - 1; gx++) {
      const p = patches[gy * cols + gx];
      if (!p.inRoadZone || p.isAnomaly || p.roadLikelihood < 0.45) continue;

      const isSmoothReflective =
        p.stdL < 0.095 &&
        p.meanGrad < 0.11 &&
        (p.lumContrast >= 0.065 || p.meanL > baseL + 0.11) &&
        p.chromaShift < 0.18;

      if (!isSmoothReflective) continue;

      let rimLeft = false;
      let rimRight = false;
      let rimUp = false;
      let rimDown = false;
      const maxSearch = 12;
      for (let d = 1; d <= maxSearch; d++) {
        if (gx - d >= 0 && isRimBoundary(patches[gy * cols + (gx - d)])) {
          rimLeft = true;
          break;
        }
      }
      for (let d = 1; d <= maxSearch; d++) {
        if (gx + d < cols && isRimBoundary(patches[gy * cols + (gx + d)])) {
          rimRight = true;
          break;
        }
      }
      for (let d = 1; d <= maxSearch; d++) {
        if (gy - d >= roadStartGy && isRimBoundary(patches[(gy - d) * cols + gx])) {
          rimUp = true;
          break;
        }
      }
      for (let d = 1; d <= maxSearch; d++) {
        if (gy + d < rows && isRimBoundary(patches[(gy + d) * cols + gx])) {
          rimDown = true;
          break;
        }
      }

      const rimDirections =
        Number(rimLeft) + Number(rimRight) + Number(rimUp) + Number(rimDown);
      const boundedAcross = (rimLeft && rimRight) || (rimUp && rimDown);
      if (rimDirections >= 3 || (rimDirections >= 2 && boundedAcross)) {
        waterEnclosedSet.add(p.idx);
        p.isWaterSeed = true;
      }
    }
  }
  if (waterEnclosedSet.size >= 2) {
    // Also promote adjacent fractured rim patches touching the enclosed puddle
    const extraRim = new Set<number>();
    for (const idx of waterEnclosedSet) {
      const wp = patches[idx];
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const nx = wp.gx + dx;
          const ny = wp.gy + dy;
          if (nx >= 0 && nx < cols && ny >= roadStartGy && ny < rows) {
            const np = patches[ny * cols + nx];
            if (!np.isAnomaly && isRimBoundary(np)) {
              extraRim.add(np.idx);
            }
          }
        }
      }
    }
    for (const idx of waterEnclosedSet) {
      patches[idx].isAnomaly = true;
      patches[idx].anomalyZ = Math.max(
        patches[idx].anomalyZ,
        cfg.zThreshold + 0.35
      );
    }
    for (const idx of extraRim) {
      patches[idx].isAnomaly = true;
      patches[idx].anomalyZ = Math.max(
        patches[idx].anomalyZ,
        cfg.zThreshold + 0.15
      );
    }
  }

  // Compute baseline road Shannon entropy from non-anomalous normal road patches
  const normalRoadPatches = roadPatches.filter((p) => !p.isAnomaly);
  const baselineEntropyBits = computeBaselineRoadEntropy(
    lum,
    ANALYSIS_WIDTH,
    ANALYSIS_HEIGHT,
    normalRoadPatches,
    PATCH_SIZE
  );

  // Spatial DBSCAN clustering on verified anomaly patches (eps=2.25 bridges 1-patch puddle/rim gaps)
  const dbscanRes = runDbscan(patches, 2.25, params.dbscanMinPts);

  // Gather candidate clusters & merge only overlapping/touching fragments
  interface RawClusterBox {
    members: PatchFeature[];
    minGx: number;
    maxGx: number;
    minGy: number;
    maxGy: number;
  }

  const rawClusters: RawClusterBox[] = [];
  for (let cid = 0; cid < dbscanRes.clusterCount; cid++) {
    const members = patches.filter((p) => p.dbscanClusterId === cid);
    if (members.length < params.dbscanMinPts) continue;
    let minGx = cols;
    let maxGx = 0;
    let minGy = rows;
    let maxGy = 0;
    for (const m of members) {
      if (m.gx < minGx) minGx = m.gx;
      if (m.gx > maxGx) maxGx = m.gx;
      if (m.gy < minGy) minGy = m.gy;
      if (m.gy > maxGy) maxGy = m.gy;
    }
    rawClusters.push({ members, minGx, maxGx, minGy, maxGy });
  }

  // Merge only directly overlapping/touching bounding boxes (gap <= 0)
  let merged = true;
  while (merged) {
    merged = false;
    for (let i = 0; i < rawClusters.length && !merged; i++) {
      for (let j = i + 1; j < rawClusters.length && !merged; j++) {
        const a = rawClusters[i];
        const b = rawClusters[j];
        const gapX = Math.max(0, Math.max(a.minGx, b.minGx) - Math.min(a.maxGx, b.maxGx) - 1);
        const gapY = Math.max(0, Math.max(a.minGy, b.minGy) - Math.min(a.maxGy, b.maxGy) - 1);
        if (gapX === 0 && gapY === 0) {
          a.members = [...a.members, ...b.members];
          a.minGx = Math.min(a.minGx, b.minGx);
          a.maxGx = Math.max(a.maxGx, b.maxGx);
          a.minGy = Math.min(a.minGy, b.minGy);
          a.maxGy = Math.max(a.maxGy, b.maxGy);
          rawClusters.splice(j, 1);
          merged = true;
        }
      }
    }
  }

  const detections: PotholeDetectionItem[] = [];
  const minedCandidates: CandidateRegionSample[] = [];

  for (const cluster of rawClusters) {
    const { members, minGx, maxGx, minGy, maxGy } = cluster;
    if (members.length < params.dbscanMinPts) continue;

    const boxWPatches = maxGx - minGx + 1;
    const boxHPatches = maxGy - minGy + 1;
    const boxAreaPatches = boxWPatches * boxHPatches;
    const areaFrac = members.length / Math.max(1, roadPatches.length);
    const aspectRatio = boxWPatches / Math.max(1, boxHPatches);
    const fillDensity = members.length / Math.max(1, boxAreaPatches);

    let sumZ = 0;
    let peakZ = 0;
    let sumDep = 0;
    let sumRough = 0;
    let sumCorner = 0;
    let sumIso = 0;
    let sumCoherence = 0;
    let waterCount = 0;
    let sumRoadLik = 0;

    for (const m of members) {
      sumZ += m.anomalyZ;
      if (m.anomalyZ > peakZ) peakZ = m.anomalyZ;
      sumDep += m.lumDepression;
      sumRough += m.isotropicRough;
      sumCorner += m.cornerEnergy;
      sumIso += m.isolationScore;
      sumCoherence += m.coherence;
      sumRoadLik += m.roadLikelihood;
      if (m.isWaterSeed) waterCount++;
    }

    const meanZ = sumZ / members.length;
    const meanDep = sumDep / members.length;
    const meanRough = sumRough / members.length;
    const meanCorner = sumCorner / members.length;
    const meanIso = sumIso / members.length;
    const meanCoherence = sumCoherence / members.length;
    const meanRoadLik = sumRoadLik / members.length;
    const isWaterlogged = waterCount >= Math.max(3, Math.floor(members.length * 0.2));

    // Geometry & plausibility filtering
    if (areaFrac < 0.007 || members.length < params.dbscanMinPts) continue;
    if (!miningMode && aspectRatio > 4.2 && boxHPatches <= 2) continue;
    if (!miningMode && aspectRatio < 0.24 && boxWPatches <= 2) continue;
    if (fillDensity < (isWaterlogged ? 0.16 : 0.24) && members.length < 12) continue;
    if (meanRoadLik < 0.55) continue;
    // Exclude full-width bottom dashcam car hood / dashboard / wiper blades
    if (boxWPatches >= cols * 0.78 && minGy / rows >= 0.76 && meanCoherence > 0.68) {
      continue;
    }

    // Region-level Outer Road Ring Verification (eliminates uniform shadows & clean-road noise!)
    const ringPad = 4;
    const outerRingPatches: PatchFeature[] = [];
    for (
      let ry = Math.max(roadStartGy, minGy - ringPad);
      ry <= Math.min(rows - 1, maxGy + ringPad);
      ry++
    ) {
      for (
        let rx = Math.max(0, minGx - ringPad);
        rx <= Math.min(cols - 1, maxGx + ringPad);
        rx++
      ) {
        const insideBox =
          rx >= minGx && rx <= maxGx && ry >= minGy && ry <= maxGy;
        if (!insideBox) {
          const rp = patches[ry * cols + rx];
          if (rp.inRoadZone && !rp.isAnomaly && rp.meanL < 0.72) {
            outerRingPatches.push(rp);
          }
        }
      }
    }

    // Use cleanest 65% of outer ring patches to avoid rim-edge contamination on large close-up potholes
    const sortedOuter = [...outerRingPatches].sort(
      (a, b) => a.isotropicRough - b.isotropicRough
    );
    const cleanOuter =
      sortedOuter.length >= 6
        ? sortedOuter.slice(0, Math.ceil(sortedOuter.length * 0.65))
        : sortedOuter;

    const outerMeanL =
      cleanOuter.length > 0
        ? cleanOuter.reduce((s, p) => s + p.meanL, 0) / cleanOuter.length
        : baseL;
    const outerMeanRough =
      cleanOuter.length > 0
        ? cleanOuter.reduce((s, p) => s + p.isotropicRough, 0) / cleanOuter.length
        : baseRough;
    const outerMeanCorner =
      cleanOuter.length > 0
        ? cleanOuter.reduce((s, p) => s + p.cornerEnergy, 0) / cleanOuter.length
        : baseCorner;

    const rimRoughRatio =
      (meanRough + 0.01) / Math.max(0.02, outerMeanRough + 0.01);
    const rimCornerRatio =
      (meanCorner + 0.005) / Math.max(0.008, outerMeanCorner + 0.005);
    const innerMeanL =
      members.reduce((s, p) => s + p.meanL, 0) / members.length;
    const interiorLumDrop = Math.max(0, outerMeanL - innerMeanL);
    const brightPaintExcess = isWaterlogged
      ? 0
      : Math.max(0, innerMeanL - outerMeanL);

    const xMin = Math.round((minGx / cols) * 1000);
    const yMin = Math.round((minGy / rows) * 1000);
    const xMax = Math.round(((maxGx + 1) / cols) * 1000);
    const yMax = Math.round(((maxGy + 1) / rows) * 1000);
    const box = { xMin, yMin, xMax, yMax };

    // Estimate depth using Shannon Image Entropy & Relative Scale Analysis
    const depthResult = estimatePotholeDepthUnsupervised({
      luminance: lum,
      gradient: grad,
      frameWidth: ANALYSIS_WIDTH,
      frameHeight: ANALYSIS_HEIGHT,
      box,
      baselineLuminance: outerMeanL,
      baselineEntropyBits,
      fillDensity,
    });

    const clampedZ = Math.min(8, meanZ);
    const logRimRough = Math.log(Math.max(0.4, rimRoughRatio));
    const logRimCorner = Math.log(Math.max(0.4, rimCornerRatio));
    const isotropy = Math.max(0, 1 - meanCoherence);
    const clampedEntropy = Math.min(2.6, depthResult.details.entropyRatio);
    const cavityInteraction =
      Math.max(0, logRimCorner) *
      (interiorLumDrop * 5 + (isWaterlogged ? 0.45 : 0));

    const featureVector: number[] = [
      clampedZ,
      logRimRough,
      logRimCorner,
      interiorLumDrop,
      brightPaintExcess,
      isotropy,
      clampedEntropy,
      meanIso,
      fillDensity,
      cavityInteraction,
    ];

    if (miningMode) {
      minedCandidates.push({
        features: featureVector,
        box,
        patchCount: members.length,
        meanZ,
        isWaterlogged,
      });
      continue;
    }

    // Must exceed ensemble Z & Isolation Forest thresholds AND pass structural contrast vs outer ring!
    // (Large craters >8% of road area exhibit iForest masking, so relax iForest floor for wide cavities)
    const effectiveMinIso = areaFrac > 0.08 ? 0.43 : cfg.minPatchIsolation;
    const effectiveMinMeanZ = isWaterlogged
      ? Math.min(cfg.minClusterMeanZ, cfg.zThreshold - 0.15)
      : cfg.minClusterMeanZ;
    if (meanZ < effectiveMinMeanZ && peakZ < cfg.minClusterPeakZ) continue;
    if (meanIso < effectiveMinIso && peakZ < cfg.minClusterPeakZ + 0.4) continue;

    const hasDeepBrokenCavity =
      meanRough >= baseRough * 1.3 &&
      meanCorner >= baseCorner * 1.35 &&
      (interiorLumDrop >= cfg.minLumDepression ||
        meanDep >= cfg.minLumDepression * 0.9);

    const passesPhysicalCavityCheck = isWaterlogged
      ? rimRoughRatio >= 1.08 || rimCornerRatio >= 1.2 || peakZ >= cfg.minClusterPeakZ
      : hasDeepBrokenCavity ||
        (rimRoughRatio >= cfg.minRimRoughnessRatio && rimCornerRatio >= 1.2) ||
        (interiorLumDrop >= cfg.minLumDepression * 1.1 && rimCornerRatio >= 1.15);

    if (!passesPhysicalCavityCheck) continue;

    // Apply backend-trained supervised classifier if loaded (specifically tuned to suppress false positives)
    let classifierProb: number | undefined;
    if (activeTrainedModel) {
      classifierProb = scoreFeatureVectorWithModel(
        featureVector,
        activeTrainedModel
      );
      if (classifierProb < activeTrainedModel.decisionThreshold) {
        continue;
      }
    }

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
      areaFrac * 16 +
      meanZ * 0.32 +
      depthResult.estimatedDepthCm * 0.085;
    if (
      severityIndex >= 3.45 ||
      areaFrac >= 0.095 ||
      depthResult.estimatedDepthCm >= 17
    ) {
      severity = 'Critical';
    } else if (
      severityIndex >= 2.4 ||
      areaFrac >= 0.042 ||
      depthResult.estimatedDepthCm >= 12
    ) {
      severity = 'Severe';
    } else if (severityIndex >= 1.65) {
      severity = 'Moderate';
    } else {
      severity = 'Low';
    }

    const baseConf =
      62 + (meanZ - cfg.zThreshold) * 14 + (meanIso - 0.5) * 45 + fillDensity * 12;
    const clusterConfidence = Math.round(
      Math.min(
        98,
        Math.max(
          68,
          classifierProb !== undefined
            ? 0.5 * baseConf + 0.5 * (classifierProb * 100)
            : baseConf
        )
      )
    );

    const luminanceDropPct = Math.round(Math.max(meanDep, interiorLumDrop) * 100);
    const roughnessScorePct = Math.round(meanRough * 100);

    detections.push({
      label: `${isWaterlogged ? 'Waterlogged ' : ''}Pothole #${
        detections.length + 1
      } (${clusterConfidence}% match)`,
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
      classifierProbability:
        classifierProb !== undefined
          ? Number(classifierProb.toFixed(3))
          : undefined,
      hazardDescription: `${
        isWaterlogged ? 'Water-filled roadway cavity' : 'Roadway cavity'
      } approx. ${estimatedWidthCm} cm wide and ${
        depthResult.estimatedDepthCm
      } cm deep (${depthResult.details.confidenceRangeCm}), ${roughnessScorePct}% surface roughness, and ${
        depthResult.details.shadowOcclusionPct
      }% interior shadow depth.`,
      box,
    });
  }

  detections.sort((a, b) => {
    const areaA = (a.box.xMax - a.box.xMin) * (a.box.yMax - a.box.yMin);
    const areaB = (b.box.xMax - b.box.xMin) * (b.box.yMax - b.box.yMin);
    return areaB - areaA;
  });
  const topDetections = detections.slice(0, 5);

  const potholeDetected = topDetections.length > 0;

  // Online Unsupervised Baseline Training (update on clean road frames or explicit calibration)
  if ((!potholeDetected && roadPatches.length > 40) || forceBaselineTrain) {
    const nPrev = learnedRoadBaseline.framesTrained;
    const alpha = nPrev === 0 ? 1 : Math.max(0.08, 1 / Math.min(20, nPrev + 1));
    learnedRoadBaseline = {
      framesTrained: nPrev + 1,
      meanL: (1 - alpha) * learnedRoadBaseline.meanL + alpha * frameBaselineL,
      stdL: (1 - alpha) * learnedRoadBaseline.stdL + alpha * scaleL,
      meanRough:
        (1 - alpha) * learnedRoadBaseline.meanRough + alpha * frameBaselineRough,
      stdRough:
        (1 - alpha) * learnedRoadBaseline.stdRough + alpha * scaleRough,
      meanCorner:
        (1 - alpha) * learnedRoadBaseline.meanCorner + alpha * frameBaselineCorner,
      stdCorner:
        (1 - alpha) * learnedRoadBaseline.stdCorner + alpha * scaleCorner,
      meanChroma:
        (1 - alpha) * learnedRoadBaseline.meanChroma + alpha * seedChroma,
      entropyBits:
        (1 - alpha) * learnedRoadBaseline.entropyBits + alpha * baselineEntropyBits,
    };
  }

  let overlayDataUrl = '';
  if (
    renderOverlay &&
    typeof document !== 'undefined' &&
    params.overlayMode !== 'detections'
  ) {
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
          const norm = Math.min(1, Math.max(0, (p.anomalyZ - 1.2) / 3.6));
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
    ? Math.round(Math.min(97, Math.max(68, 62 + (maxAnomZ - cfg.zThreshold) * 14)))
    : Math.round(Math.min(98, Math.max(80, 95 - Math.max(0, maxAnomZ - 1.5) * 8)));

  const roadSurfaceType =
    baseL < 0.35
      ? 'Dark Asphalt Pavement'
      : baseL < 0.55
      ? 'Weathered Asphalt Roadway'
      : 'Light Concrete / Wet Surface';

  const maxWidth = potholeDetected
    ? Math.max(...topDetections.map((d) => d.estimatedWidthCm))
    : 0;
  const maxDepth = potholeDetected
    ? Math.max(...topDetections.map((d) => d.estimatedDepthCm))
    : 0;

  const formalComplaintTitle = potholeDetected
    ? `${overallSeverity} Roadway Pothole Detected (${topDetections.length} Hazard${
        topDetections.length > 1 ? 's' : ''
      }, ~${maxWidth} cm)`
    : 'Clear Roadway Segment';

  const formalComplaintDescription = potholeDetected
    ? `Automated camera road inspection detected ${topDetections.length} pavement hazard${
        topDetections.length > 1 ? 's' : ''
      } on ${roadSurfaceType.toLowerCase()}. Largest pothole measures approx. ${maxWidth} cm wide by ${maxDepth} cm deep.`
    : 'No significant roadway potholes or surface cavities detected in this segment.';

  const inferenceMs = Number((performance.now() - startTime).toFixed(1));
  const anomalousPatchCount = roadPatches.filter((p) => p.isAnomaly).length;
  const meanIsoAll =
    roadPatches.length > 0
      ? roadPatches.reduce((s, p) => s + p.isolationScore, 0) / roadPatches.length
      : 0;

  return {
    minedCandidates,
    output: {
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
        anomalyThresholdZ: cfg.zThreshold,
        anomalousPatchCount,
        dbscanClustersFound: topDetections.length,
        noisePatchesPruned: dbscanRes.noiseCount,
        baselineLuminanceMean: Number(baseL.toFixed(2)),
        baselineRoughnessMean: Number(baseRough.toFixed(2)),
        trainedFrames: learnedRoadBaseline.framesTrained,
        gmmLogLikelihood: Number(gmm.avgLogLikelihood.toFixed(2)),
        isolationScoreMean: Number(meanIsoAll.toFixed(2)),
      },
      clusters,
      overlayDataUrl,
    },
  };
}
