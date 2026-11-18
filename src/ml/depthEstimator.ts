import { BoundingBox, PotholeDetectionItem } from '../types';

export interface DepthEstimationInput {
  luminance: Float32Array;
  gradient: Float32Array;
  frameWidth: number;
  frameHeight: number;
  box: BoundingBox; // 0..1000 normalized
  baselineLuminance: number;
  baselineEntropyBits: number;
  fillDensity: number;
}

export interface DepthEstimationDetails {
  method: string;
  shannonEntropyBits: number;
  baselineEntropyBits: number;
  entropyRatio: number;
  relativeScaleIndex: number;
  shadowOcclusionPct: number;
  confidenceRangeCm: string;
}

export interface DepthEstimationOutput {
  estimatedDepthCm: number;
  depthClass: PotholeDetectionItem['depthClass'];
  details: DepthEstimationDetails;
}

const HISTOGRAM_BINS = 16;

/**
 * Computes Shannon image entropy (in bits, 0..4 for 16 bins) over a rectangular pixel region.
 */
export function computeRegionShannonEntropy(
  luminance: Float32Array,
  frameWidth: number,
  frameHeight: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
): number {
  const bins = new Int32Array(HISTOGRAM_BINS);
  let count = 0;

  const startX = Math.max(0, Math.min(frameWidth - 1, Math.floor(x0)));
  const endX = Math.max(startX + 1, Math.min(frameWidth, Math.ceil(x1)));
  const startY = Math.max(0, Math.min(frameHeight - 1, Math.floor(y0)));
  const endY = Math.max(startY + 1, Math.min(frameHeight, Math.ceil(y1)));

  for (let y = startY; y < endY; y++) {
    const rowOffset = y * frameWidth;
    for (let x = startX; x < endX; x++) {
      const val = luminance[rowOffset + x];
      const bin = Math.min(
        HISTOGRAM_BINS - 1,
        Math.max(0, Math.floor(val * HISTOGRAM_BINS))
      );
      bins[bin]++;
      count++;
    }
  }

  if (count === 0) return 0;

  let entropy = 0;
  for (let b = 0; b < HISTOGRAM_BINS; b++) {
    if (bins[b] === 0) continue;
    const p = bins[b] / count;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

/**
 * Estimates baseline Shannon entropy from normal asphalt road patches (unsupervised reference).
 */
export function computeBaselineRoadEntropy(
  luminance: Float32Array,
  frameWidth: number,
  frameHeight: number,
  normalPatchCoords: Array<{ gx: number; gy: number }>,
  patchSize: number
): number {
  if (normalPatchCoords.length === 0) return 1.6;

  const sampleStep = Math.max(1, Math.floor(normalPatchCoords.length / 40));
  const entropies: number[] = [];

  for (let i = 0; i < normalPatchCoords.length; i += sampleStep) {
    const { gx, gy } = normalPatchCoords[i];
    const x0 = gx * patchSize;
    const y0 = gy * patchSize;
    const h = computeRegionShannonEntropy(
      luminance,
      frameWidth,
      frameHeight,
      x0,
      y0,
      x0 + patchSize,
      y0 + patchSize
    );
    entropies.push(h);
  }

  entropies.sort((a, b) => a - b);
  return entropies[Math.floor(entropies.length / 2)] || 1.6;
}

/**
 * Unsupervised Computer Vision Depth Estimator:
 * Combines Shannon Image Entropy (texture & aggregate cavity complexity),
 * Relative Scale Analysis (perspective-normalized angular footprint),
 * and Shadow Occlusion Ratio relative to the learned road asphalt baseline.
 */
export function estimatePotholeDepthUnsupervised(
  input: DepthEstimationInput
): DepthEstimationOutput {
  const {
    luminance,
    gradient,
    frameWidth,
    frameHeight,
    box,
    baselineLuminance,
    baselineEntropyBits,
    fillDensity,
  } = input;

  const x0 = Math.max(0, Math.floor((box.xMin / 1000) * frameWidth));
  const y0 = Math.max(0, Math.floor((box.yMin / 1000) * frameHeight));
  const x1 = Math.min(frameWidth, Math.ceil((box.xMax / 1000) * frameWidth));
  const y1 = Math.min(frameHeight, Math.ceil((box.yMax / 1000) * frameHeight));

  const roiW = Math.max(1, x1 - x0);
  const roiH = Math.max(1, y1 - y0);
  const roiArea = roiW * roiH;

  // 1. Shannon Image Entropy inside detected pothole ROI
  const roiEntropy = computeRegionShannonEntropy(
    luminance,
    frameWidth,
    frameHeight,
    x0,
    y0,
    x1,
    y1
  );
  const safeBaselineEntropy = Math.max(0.85, baselineEntropyBits);
  const entropyRatio = roiEntropy / safeBaselineEntropy;
  const entropyExcess = Math.max(0, roiEntropy - safeBaselineEntropy * 0.85);

  // 2. Shadow Occlusion & Rim Gradient Energy inside ROI
  let occludedCount = 0;
  let sumGrad = 0;
  let sumLumDrop = 0;
  const shadowCutoff = Math.max(0.08, baselineLuminance * 0.82);

  for (let y = y0; y < y1; y++) {
    const rowOff = y * frameWidth;
    for (let x = x0; x < x1; x++) {
      const idx = rowOff + x;
      const l = luminance[idx];
      sumGrad += gradient[idx];
      if (l < shadowCutoff) {
        occludedCount++;
        sumLumDrop += baselineLuminance - l;
      }
    }
  }

  const shadowOcclusionRatio = occludedCount / roiArea;
  const meanLumDrop = occludedCount > 0 ? sumLumDrop / occludedCount : 0;
  const meanGrad = sumGrad / roiArea;

  // 3. Relative Scale Analysis with Road Perspective Compensation
  const centerNormY = (y0 + y1) / (2 * frameHeight); // 0 = top, 1 = bottom (closest)
  const perspectiveMultiplier = 0.72 + (1 - Math.min(1, Math.max(0, centerNormY))) * 1.15;
  const relWidth = roiW / frameWidth;
  const relHeight = roiH / frameHeight;
  const relArea = roiArea / (frameWidth * frameHeight);

  // Perspective-normalized physical scale index (dimensionless, ~0.2 .. 2.5)
  const relativeScaleIndex =
    (Math.sqrt(relArea) * 2.2 + relWidth * 0.9 + relHeight * 0.4) *
    perspectiveMultiplier *
    Math.max(0.55, Math.min(1.0, fillDensity + 0.25));

  // Unsupervised depth model (cm):
  // - Base cavity depth: 1.8 cm
  // - Entropy complexity term: jagged aggregate & layered interior shadow spread
  // - Relative scale term: wider/longer perspective-adjusted cavities correlate with deeper structural failure
  // - Shadow occlusion term: deeper pits cast stronger interior shadow occlusion
  const entropyTerm = entropyExcess * 2.1 + Math.max(0, entropyRatio - 1) * 1.5;
  const scaleTerm = relativeScaleIndex * 4.2;
  const shadowTerm = shadowOcclusionRatio * 4.5 + meanLumDrop * 8.0 + meanGrad * 2.5;

  const rawDepthCm = 1.8 + entropyTerm + scaleTerm + shadowTerm;
  const estimatedDepthCm = Number(Math.max(2.5, Math.min(26, rawDepthCm)).toFixed(1));
  const roundedDepth = Math.round(estimatedDepthCm);

  const margin = Math.max(1, Math.round(1 + estimatedDepthCm * 0.14));
  const confidenceRangeCm = `${Math.max(2, roundedDepth - margin)}–${roundedDepth + margin} cm`;

  let depthClass: PotholeDetectionItem['depthClass'] = 'Medium (5–10 cm)';
  if (estimatedDepthCm < 5) depthClass = 'Shallow (<5 cm)';
  else if (estimatedDepthCm <= 10) depthClass = 'Medium (5–10 cm)';
  else if (estimatedDepthCm <= 15) depthClass = 'Deep (10–15 cm)';
  else depthClass = 'Severe (>15 cm)';

  return {
    estimatedDepthCm,
    depthClass,
    details: {
      method: 'Shannon Image Entropy + Relative Scale Analysis (Unsupervised)',
      shannonEntropyBits: Number(roiEntropy.toFixed(2)),
      baselineEntropyBits: Number(safeBaselineEntropy.toFixed(2)),
      entropyRatio: Number(entropyRatio.toFixed(2)),
      relativeScaleIndex: Number(relativeScaleIndex.toFixed(2)),
      shadowOcclusionPct: Math.round(shadowOcclusionRatio * 100),
      confidenceRangeCm,
    },
  };
}
