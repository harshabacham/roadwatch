import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';
import type {
  DatasetCategory,
  DatasetLabel,
  LabeledDatasetSample,
  ModelEvaluationMetrics,
  TrainedPotholeModel,
} from '../src/types.ts';
import {
  extractCandidateRegionsForTraining,
  POTHOLE_FEATURE_NAMES,
  resetLearnedRoadBaseline,
} from '../src/ml/unsupervisedDetector.ts';

const decodedRgbaCache = new Map<string, Uint8ClampedArray>();

const FRAME_W = 240;
const FRAME_H = 180;

interface MinedTrainingRow {
  sampleId: string;
  title: string;
  y: 0 | 1;
  features: number[];
}

function hash2d(x: number, y: number, seed = 1): number {
  const n = Math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453;
  return n - Math.floor(n);
}

/**
 * Resizes an RGBA buffer to 240x180 using bilinear interpolation.
 */
function resizeRgbaBilinear(
  src: Uint8Array | Uint8ClampedArray,
  srcW: number,
  srcH: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(FRAME_W * FRAME_H * 4);
  if (srcW === FRAME_W && srcH === FRAME_H) {
    out.set(src.subarray(0, out.length));
    return out;
  }

  const xRatio = (srcW - 1) / Math.max(1, FRAME_W - 1);
  const yRatio = (srcH - 1) / Math.max(1, FRAME_H - 1);

  for (let y = 0; y < FRAME_H; y++) {
    const sy = y * yRatio;
    const y0 = Math.floor(sy);
    const y1 = Math.min(srcH - 1, y0 + 1);
    const wy = sy - y0;

    for (let x = 0; x < FRAME_W; x++) {
      const sx = x * xRatio;
      const x0 = Math.floor(sx);
      const x1 = Math.min(srcW - 1, x0 + 1);
      const wx = sx - x0;

      const i00 = (y0 * srcW + x0) * 4;
      const i10 = (y0 * srcW + x1) * 4;
      const i01 = (y1 * srcW + x0) * 4;
      const i11 = (y1 * srcW + x1) * 4;
      const dst = (y * FRAME_W + x) * 4;

      for (let c = 0; c < 3; c++) {
        const top = src[i00 + c] * (1 - wx) + src[i10 + c] * wx;
        const bot = src[i01 + c] * (1 - wx) + src[i11 + c] * wx;
        out[dst + c] = Math.round(top * (1 - wy) + bot * wy);
      }
      out[dst + 3] = 255;
    }
  }
  return out;
}

/**
 * Decodes JPEG or PNG buffer (file bytes or base64 data URL) into 240x180 RGBA pixels.
 */
export function decodeImageBufferToRgba(buf: Buffer): Uint8ClampedArray {
  // PNG magic bytes: 89 50 4E 47
  if (
    buf.length >= 8 &&
    buf[0] === 0x89 &&
    buf[1] === 0x50 &&
    buf[2] === 0x4e &&
    buf[3] === 0x47
  ) {
    const png = PNG.sync.read(buf);
    return resizeRgbaBilinear(png.data, png.width, png.height);
  }
  const decoded = jpeg.decode(buf, { useTArray: true, formatAsRGBA: true });
  return resizeRgbaBilinear(decoded.data, decoded.width, decoded.height);
}

export function decodeDataUrlToRgba(dataUrl: string): Uint8ClampedArray | null {
  const match = /^data:image\/[a-zA-Z0-9+.-]+;base64,(.+)$/.exec(dataUrl.trim());
  if (!match) return null;
  const buf = Buffer.from(match[1], 'base64');
  return decodeImageBufferToRgba(buf);
}

/**
 * Generates deterministic 240x180 RGBA labeled road scenes covering diverse pothole geometries
 * and hard-negative non-pothole artifacts (lane stripes, shadows, tar seams, wet glare, coarse asphalt).
 */
export function generateSyntheticRoadImage(kind: string): Uint8ClampedArray {
  const arr = new Uint8ClampedArray(FRAME_W * FRAME_H * 4);
  const putPixel = (x: number, y: number, r: number, g: number, b: number) => {
    const i = (y * FRAME_W + x) * 4;
    arr[i] = Math.max(0, Math.min(255, Math.round(r)));
    arr[i + 1] = Math.max(0, Math.min(255, Math.round(g)));
    arr[i + 2] = Math.max(0, Math.min(255, Math.round(b)));
    arr[i + 3] = 255;
  };

  for (let y = 0; y < FRAME_H; y++) {
    const depthShade = 0.85 + (y / FRAME_H) * 0.2;
    for (let x = 0; x < FRAME_W; x++) {
      const grain = hash2d(x, y, 3) * 14 - 7;
      let r = 104 * depthShade + grain;
      let g = 105 * depthShade + grain;
      let b = 108 * depthShade + grain;

      if (kind === 'clean_coarse_chipseal') {
        const pebble = (hash2d(Math.floor(x / 2), Math.floor(y / 2), 9) - 0.5) * 32;
        r = 108 * depthShade + pebble + grain * 0.8;
        g = 107 * depthShade + pebble + grain * 0.8;
        b = 106 * depthShade + pebble + grain * 0.8;
      } else if (kind === 'white_dashed_lane_line') {
        if (y > 58 && Math.abs(x - 120) < 4 && y % 36 < 22) {
          r = 228 + grain * 0.3;
          g = 228 + grain * 0.3;
          b = 225 + grain * 0.3;
        }
      } else if (kind === 'double_yellow_centerline') {
        if (y > 55 && (Math.abs(x - 114) < 3 || Math.abs(x - 126) < 3)) {
          r = 226;
          g = 182;
          b = 48;
        }
      } else if (kind === 'crosswalk_stripes') {
        if (y > 95 && y < 145 && x % 28 < 12) {
          r = 232;
          g = 232;
          b = 230;
        }
      } else if (kind === 'tree_cast_shadow') {
        const wave = Math.sin(x * 0.08) * 8 + Math.cos(y * 0.11) * 6;
        if (y > 82 + wave && y < 138 + wave && x > 32 && x < 208) {
          r = 56 + grain * 0.5;
          g = 58 + grain * 0.5;
          b = 62 + grain * 0.5;
        }
      } else if (kind === 'vehicle_box_shadow') {
        if (y > 88 && y < 135 && x > 55 && x < 185) {
          r = 46 + grain * 0.4;
          g = 48 + grain * 0.4;
          b = 52 + grain * 0.4;
        }
      } else if (kind === 'tar_seal_crack_line') {
        const yCrack = 116 + Math.sin(x * 0.04) * 2;
        if (Math.abs(y - yCrack) <= 1.6) {
          r = 38;
          g = 38;
          b = 40;
        }
      } else if (kind === 'wet_sun_glare') {
        const dist = Math.hypot((x - 125) / 38, (y - 118) / 22);
        if (dist < 1.0) {
          const blend = 1 - dist * dist;
          r = r * (1 - blend) + 198 * blend;
          g = g * (1 - blend) + 202 * blend;
          b = b * (1 - blend) + 208 * blend;
        }
      } else if (kind === 'manhole_cover_smooth') {
        const dist = Math.hypot((x - 120) / 18, (y - 120) / 12);
        if (dist < 1.0) {
          r = 86 + (dist > 0.88 ? -20 : 4);
          g = 88 + (dist > 0.88 ? -20 : 4);
          b = 92 + (dist > 0.88 ? -20 : 4);
        }
      } else if (kind === 'curb_gutter_edge') {
        if (x < 48) {
          r = 148 + grain * 0.7;
          g = 146 + grain * 0.7;
          b = 140 + grain * 0.7;
        } else if (x < 53) {
          r = 62;
          g = 62;
          b = 64;
        }
      } else if (kind === 'indoor_blank_wall') {
        r = 186 + x * 0.04 + grain * 0.2;
        g = 182 + x * 0.04 + grain * 0.2;
        b = 176 + x * 0.04 + grain * 0.2;
      } else if (kind === 'pothole_small_asphalt') {
        const dist = Math.hypot((x - 120) / 15, (y - 124) / 10);
        if (dist < 1.0) {
          const tex = (hash2d(x * 3, y * 3, 11) - 0.5) * 95;
          const sh = y < 124 ? -56 : -30;
          r = 95 + sh + tex + (dist > 0.75 ? 16 : 0);
          g = 90 + sh + tex;
          b = 82 + sh + tex;
        }
      } else if (kind === 'pothole_medium_dark') {
        const wobble = Math.sin(Math.atan2(y - 126, x - 118) * 5) * 0.1;
        const dist = Math.hypot((x - 118) / 24, (y - 126) / 15) + wobble;
        if (dist < 1.0) {
          const tex = (hash2d(x * 2, y * 2, 17) - 0.5) * 98;
          const sh = y < 122 ? -60 : -34;
          r = 94 + sh + tex + (dist > 0.76 ? 20 : 0);
          g = 89 + sh + tex;
          b = 81 + sh + tex;
        }
      } else if (kind === 'pothole_large_spalled') {
        const wobble = Math.cos(Math.atan2(y - 120, x - 124) * 6) * 0.12;
        const dist = Math.hypot((x - 124) / 42, (y - 120) / 24) + wobble;
        if (dist < 1.0) {
          const tex = (hash2d(x * 2, y * 2, 23) - 0.5) * 105;
          const rimWall = y < 110 && dist > 0.5 ? -62 : -22;
          r = 108 + rimWall + tex;
          g = 96 + rimWall + tex;
          b = 82 + rimWall + tex;
        }
      } else if (kind === 'pothole_waterlogged_curb') {
        const wobble = Math.sin(Math.atan2(y - 122, x - 115) * 4) * 0.09;
        const dist = Math.hypot((x - 115) / 36, (y - 122) / 20) + wobble;
        if (dist < 1.0 && dist > 0.68) {
          const tex = (hash2d(x * 3, y * 3, 29) - 0.5) * 110;
          r = 52 + tex;
          g = 50 + tex;
          b = 48 + tex;
        } else if (dist <= 0.68) {
          r = 165 + Math.sin(x * 0.08) * 6;
          g = 155 + Math.sin(x * 0.08) * 6;
          b = 142 + Math.sin(x * 0.08) * 6;
        }
      } else if (kind === 'pothole_alligator_cluster') {
        const d1 = Math.hypot((x - 86) / 18, (y - 122) / 12);
        const d2 = Math.hypot((x - 152) / 20, (y - 134) / 13);
        if (d1 < 1.0 || d2 < 1.0) {
          const tex = (hash2d(x * 3, y * 3, 31) - 0.5) * 95;
          r = 62 + tex;
          g = 58 + tex;
          b = 54 + tex;
        }
      } else if (kind === 'pothole_lowlight_dusk') {
        r = 62 * depthShade + grain * 0.7;
        g = 64 * depthShade + grain * 0.7;
        b = 68 * depthShade + grain * 0.7;
        const dist = Math.hypot((x - 122) / 22, (y - 124) / 14);
        if (dist < 1.0) {
          const tex = (hash2d(x * 2, y * 2, 41) - 0.5) * 62;
          r = 30 + tex + (dist > 0.76 ? 18 : 0);
          g = 30 + tex + (dist > 0.76 ? 18 : 0);
          b = 32 + tex + (dist > 0.76 ? 18 : 0);
        }
      } else if (kind === 'pothole_concrete_joint_collapse') {
        r = 148 * depthShade + grain;
        g = 145 * depthShade + grain;
        b = 138 * depthShade + grain;
        const dist = Math.hypot((x - 120) / 26, (y - 125) / 16);
        if (dist < 1.0) {
          const tex = (hash2d(x * 3, y * 3, 47) - 0.5) * 95;
          r = 85 + tex;
          g = 80 + tex;
          b = 74 + tex;
        }
      }

      putPixel(x, y, r, g, b);
    }
  }
  return arr;
}

export function getDefaultLabeledDataset(): LabeledDatasetSample[] {
  const now = new Date().toISOString();
  const makeAsset = (
    id: string,
    label: DatasetLabel,
    title: string,
    assetPath: string,
    category: DatasetCategory
  ): LabeledDatasetSample => ({
    id,
    label,
    category,
    title,
    sourceType: 'asset',
    assetPath,
    createdAt: now,
  });

  const makeSynth = (
    id: string,
    label: DatasetLabel,
    category: DatasetCategory,
    title: string,
    syntheticKind: string
  ): LabeledDatasetSample => ({
    id,
    label,
    category,
    title,
    sourceType: 'synthetic',
    syntheticKind,
    createdAt: now,
  });

  return [
    // Real labeled road screenshots (potholes & clean road reference)
    makeAsset(
      'ds_asset_severe',
      'pothole',
      'Deep Urban Pothole (Real Sample)',
      'src/assets/images/road_sample_severe_1795005146352.jpg',
      'pothole_cavity'
    ),
    makeAsset(
      'ds_asset_moderate',
      'pothole',
      'Avenue Lane Potholes & Cracking (Real Sample)',
      'src/assets/images/road_sample_moderate_1795005146362.jpg',
      'alligator_cracking'
    ),
    makeAsset(
      'ds_asset_waterlogged',
      'pothole',
      'Waterlogged Curb Pothole (Real Sample)',
      'src/assets/images/road_sample_waterlogged_1795005147911.jpg',
      'waterlogged_pothole'
    ),
    makeAsset(
      'ds_asset_highway',
      'pothole',
      'Highway Lane Broken Pothole (Real Sample)',
      'src/assets/images/road_sample_highway_pothole_1795023499964.jpg',
      'pothole_cavity'
    ),
    makeAsset(
      'ds_asset_residential',
      'pothole',
      'Residential Curb Pothole & Cracks (Real Sample)',
      'src/assets/images/road_sample_residential_crack_1795023503887.jpg',
      'alligator_cracking'
    ),
    makeAsset(
      'ds_asset_clean_real',
      'no_pothole',
      'Clean Paved Asphalt & Shadows (Real Sample)',
      'src/assets/images/road_sample_clean_asphalt_1795023504230.jpg',
      'clean_road'
    ),
    // Labeled pothole variants
    makeSynth(
      'ds_pos_small',
      'pothole',
      'pothole_cavity',
      'Small Deep Asphalt Cavity',
      'pothole_small_asphalt'
    ),
    makeSynth(
      'ds_pos_medium',
      'pothole',
      'pothole_cavity',
      'Medium Occluded Urban Pothole',
      'pothole_medium_dark'
    ),
    makeSynth(
      'ds_pos_large',
      'pothole',
      'pothole_cavity',
      'Large Spalled Pavement Pit',
      'pothole_large_spalled'
    ),
    makeSynth(
      'ds_pos_water',
      'pothole',
      'waterlogged_pothole',
      'Rain-Filled Reflective Pothole',
      'pothole_waterlogged_curb'
    ),
    makeSynth(
      'ds_pos_alligator',
      'pothole',
      'alligator_cracking',
      'Twin Alligator-Cracked Potholes',
      'pothole_alligator_cluster'
    ),
    makeSynth(
      'ds_pos_dusk',
      'pothole',
      'pothole_cavity',
      'Low-Light Dusk Asphalt Pothole',
      'pothole_lowlight_dusk'
    ),
    makeSynth(
      'ds_pos_concrete',
      'pothole',
      'pothole_cavity',
      'Concrete Slab Joint Collapse',
      'pothole_concrete_joint_collapse'
    ),
    // Hard-negative non-pothole scenes (false-positive suppression training)
    makeSynth(
      'ds_neg_clean',
      'no_pothole',
      'clean_road',
      'Clean Uniform Asphalt Roadway',
      'clean_smooth_asphalt'
    ),
    makeSynth(
      'ds_neg_chipseal',
      'no_pothole',
      'clean_road',
      'Coarse Chipseal Intact Pavement',
      'clean_coarse_chipseal'
    ),
    makeSynth(
      'ds_neg_white_lane',
      'no_pothole',
      'lane_marking',
      'White Dashed Lane Markings',
      'white_dashed_lane_line'
    ),
    makeSynth(
      'ds_neg_yellow_line',
      'no_pothole',
      'lane_marking',
      'Double Yellow Centerline',
      'double_yellow_centerline'
    ),
    makeSynth(
      'ds_neg_crosswalk',
      'no_pothole',
      'lane_marking',
      'Pedestrian Crosswalk Stripes',
      'crosswalk_stripes'
    ),
    makeSynth(
      'ds_neg_tree_shadow',
      'no_pothole',
      'shadow',
      'Tree Canopy Cast Shadow',
      'tree_cast_shadow'
    ),
    makeSynth(
      'ds_neg_car_shadow',
      'no_pothole',
      'shadow',
      'Parked Vehicle Undercarriage Shadow',
      'vehicle_box_shadow'
    ),
    makeSynth(
      'ds_neg_tar_crack',
      'no_pothole',
      'tar_joint',
      'Sealed Linear Tar Expansion Joint',
      'tar_seal_crack_line'
    ),
    makeSynth(
      'ds_neg_wet_glare',
      'no_pothole',
      'wet_glare',
      'Intact Wet Pavement Sun Glare',
      'wet_sun_glare'
    ),
    makeSynth(
      'ds_neg_manhole',
      'no_pothole',
      'clean_road',
      'Flush Utility Manhole Cover',
      'manhole_cover_smooth'
    ),
    makeSynth(
      'ds_neg_curb',
      'no_pothole',
      'tar_joint',
      'Roadside Curb & Gutter Seam',
      'curb_gutter_edge'
    ),
    makeSynth(
      'ds_neg_indoor',
      'no_pothole',
      'clean_road',
      'Smooth Non-Road Wall / Floor',
      'indoor_blank_wall'
    ),
  ];
}

export function loadSamplePixels(
  sample: LabeledDatasetSample,
  projectRoot: string
): Uint8ClampedArray | null {
  const cacheKey = `${sample.id}:${sample.assetPath || sample.syntheticKind || ''}`;
  if (sample.sourceType === 'asset' && sample.assetPath) {
    const cached = decodedRgbaCache.get(cacheKey);
    if (cached) return cached;
    const absPath = path.resolve(projectRoot, sample.assetPath);
    if (!fs.existsSync(absPath)) return null;
    const buf = fs.readFileSync(absPath);
    const decoded = decodeImageBufferToRgba(buf);
    decodedRgbaCache.set(cacheKey, decoded);
    return decoded;
  }
  if (sample.sourceType === 'synthetic' && sample.syntheticKind) {
    const cached = decodedRgbaCache.get(cacheKey);
    if (cached) return cached;
    const generated = generateSyntheticRoadImage(sample.syntheticKind);
    decodedRgbaCache.set(cacheKey, generated);
    return generated;
  }
  if (sample.imageDataUrl) {
    return decodeDataUrlToRgba(sample.imageDataUrl);
  }
  return null;
}

function sigmoid(z: number): number {
  if (z > 30) return 1;
  if (z < -30) return 0;
  return 1 / (1 + Math.exp(-z));
}

function fitStandardizedLogisticRegression(
  rows: MinedTrainingRow[],
  falsePositivePenalty: number,
  epochs = 450
): {
  means: number[];
  stds: number[];
  weights: number[];
  bias: number;
} {
  const dims = POTHOLE_FEATURE_NAMES.length;
  const n = rows.length;
  const means = new Array<number>(dims).fill(0);
  const stds = new Array<number>(dims).fill(1);

  if (n === 0) {
    return {
      means,
      stds,
      weights: new Array<number>(dims).fill(0),
      bias: 0,
    };
  }

  for (const r of rows) {
    for (let d = 0; d < dims; d++) means[d] += r.features[d];
  }
  for (let d = 0; d < dims; d++) means[d] /= n;

  for (const r of rows) {
    for (let d = 0; d < dims; d++) {
      const diff = r.features[d] - means[d];
      stds[d] += diff * diff;
    }
  }
  for (let d = 0; d < dims; d++) {
    stds[d] = Math.max(1e-3, Math.sqrt(stds[d] / n));
  }

  const xNorm = rows.map((r) =>
    r.features.map((v, d) => (v - means[d]) / stds[d])
  );

  // Adam optimizer on cost-sensitive L2-regularized cross-entropy
  const weights = new Array<number>(dims).fill(0);
  let bias = 0;
  const mW = new Array<number>(dims).fill(0);
  const vW = new Array<number>(dims).fill(0);
  let mB = 0;
  let vB = 0;

  const lr = 0.045;
  const l2 = 0.04;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const gradW = new Array<number>(dims).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      const x = xNorm[i];
      const y = rows[i].y;
      let z = bias;
      for (let d = 0; d < dims; d++) z += weights[d] * x[d];
      const p = sigmoid(z);

      // Asymmetric cost weighting: false positives penalized falsePositivePenalty times harder
      const sampleWeight = y === 0 ? falsePositivePenalty : 1.0;
      const err = (p - y) * sampleWeight;
      gradB += err;
      for (let d = 0; d < dims; d++) {
        gradW[d] += err * x[d];
      }
    }

    const bc1 = 1 - Math.pow(beta1, epoch);
    const bc2 = 1 - Math.pow(beta2, epoch);

    gradB /= n;
    mB = beta1 * mB + (1 - beta1) * gradB;
    vB = beta2 * vB + (1 - beta2) * gradB * gradB;
    bias -= (lr * (mB / bc1)) / (Math.sqrt(vB / bc2) + eps);

    for (let d = 0; d < dims; d++) {
      const g = gradW[d] / n + l2 * weights[d];
      mW[d] = beta1 * mW[d] + (1 - beta1) * g;
      vW[d] = beta2 * vW[d] + (1 - beta2) * g * g;
      weights[d] -= (lr * (mW[d] / bc1)) / (Math.sqrt(vW[d] / bc2) + eps);
    }
  }

  return { means, stds, weights, bias };
}

function predictProbability(
  features: number[],
  model: { means: number[]; stds: number[]; weights: number[]; bias: number }
): number {
  let z = model.bias;
  for (let d = 0; d < model.weights.length; d++) {
    const xn = (features[d] - model.means[d]) / model.stds[d];
    z += model.weights[d] * xn;
  }
  return sigmoid(z);
}

function computeRocAuc(probs: number[], labels: Array<0 | 1>): number {
  const pos: number[] = [];
  const neg: number[] = [];
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) pos.push(probs[i]);
    else neg.push(probs[i]);
  }
  if (pos.length === 0 || neg.length === 0) return 1;
  let wins = 0;
  for (const p of pos) {
    for (const n of neg) {
      if (p > n) wins += 1;
      else if (p === n) wins += 0.5;
    }
  }
  return wins / (pos.length * neg.length);
}

/**
 * Full Backend ML Training Pipeline:
 * 1. Loads all labeled images from data/training_dataset.json
 * 2. Runs unsupervised candidate mining (GMM + iForest + Structure Tensor + DBSCAN)
 *    to extract 10D region features for positives and hard negatives
 * 3. Runs 5-fold grouped cross-validation with cost-sensitive L2 Logistic Regression
 * 4. Sweeps decision threshold tau to minimize False Positive Rate while preserving Recall
 * 5. Fits final weights on all labeled regions and persists data/ml_model.json
 */
export function trainBackendPotholeModel(options: {
  dataset: LabeledDatasetSample[];
  projectRoot: string;
  previousVersion?: number;
  falsePositivePenalty?: number;
}): TrainedPotholeModel {
  const { dataset, projectRoot } = options;
  const falsePositivePenalty = Math.max(
    1.0,
    Math.min(5.0, options.falsePositivePenalty ?? 2.5)
  );
  const epochs = 450;

  const minedRows: MinedTrainingRow[] = [];

  for (const sample of dataset) {
    const rgba = loadSamplePixels(sample, projectRoot);
    if (!rgba) continue;

    resetLearnedRoadBaseline();
    const candidates = extractCandidateRegionsForTraining(rgba);

    if (sample.label === 'no_pothole') {
      // Every candidate mined from a no-pothole image is a true Hard Negative!
      for (const cand of candidates) {
        minedRows.push({
          sampleId: sample.id,
          title: sample.title,
          y: 0,
          features: cand.features,
        });
      }
      // Also add clean-road baseline anchor in 10D feature space
      minedRows.push({
        sampleId: sample.id,
        title: sample.title,
        y: 0,
        features: [1.15, 0.0, 0.0, 0.015, 0.0, 0.55, 1.0, 0.44, 0.52, 0.0],
      });
    } else {
      // Labeled pothole image: keep genuine structural pothole candidates only
      // so weak background spots on positive images are not mislabeled y=1.
      const sorted = [...candidates].sort((a, b) => {
        const scoreA = a.meanZ + a.features[1] * 2.2 + a.features[2] * 2.5 + a.features[9] * 2.0;
        const scoreB = b.meanZ + b.features[1] * 2.2 + b.features[2] * 2.5 + b.features[9] * 2.0;
        return scoreB - scoreA;
      });
      const kept = sorted.filter(
        (c) =>
          c.meanZ >= 2.15 &&
          (c.features[1] > 0.08 || c.isWaterlogged) &&
          c.features[2] > 0.12 &&
          c.features[4] < 0.18
      );
      const positives = kept.length > 0 ? kept.slice(0, 2) : sorted.slice(0, 1);
      for (const cand of positives) {
        minedRows.push({
          sampleId: sample.id,
          title: sample.title,
          y: 1,
          features: cand.features,
        });
      }
    }
  }

  resetLearnedRoadBaseline();

  // Primary Trainer: Python 3.11 ML pipeline (ml/train_pothole_model.py)
  const pyScriptPath = path.resolve(projectRoot, 'ml', 'train_pothole_model.py');
  if (fs.existsSync(pyScriptPath) && minedRows.length > 0) {
    const pyPayload = {
      rows: minedRows,
      featureNames: [...POTHOLE_FEATURE_NAMES],
      falsePositivePenalty,
      previousVersion: options.previousVersion ?? 0,
      datasetStats: {
        datasetSamples: dataset.length,
        positiveSamples: dataset.filter((d) => d.label === 'pothole').length,
        negativeSamples: dataset.filter((d) => d.label === 'no_pothole').length,
      },
    };
    const pyRes = spawnSync('python3', [pyScriptPath], {
      input: JSON.stringify(pyPayload),
      encoding: 'utf-8',
      maxBuffer: 12 * 1024 * 1024,
      timeout: 30000,
    });
    if (pyRes.status === 0 && pyRes.stdout) {
      try {
        const parsed = JSON.parse(pyRes.stdout) as TrainedPotholeModel;
        if (
          Array.isArray(parsed.weights) &&
          parsed.weights.length === POTHOLE_FEATURE_NAMES.length &&
          parsed.metrics
        ) {
          return parsed;
        }
      } catch (err) {
        console.warn('Failed to parse Python ML output, falling back to TS trainer:', err);
      }
    } else if (pyRes.stderr) {
      console.warn('Python ML trainer stderr:', pyRes.stderr);
    }
  }

  // Fallback TS implementation if python3 is unavailable
  const uniqueSampleIds = Array.from(new Set(minedRows.map((r) => r.sampleId)));
  const kFolds = Math.min(5, Math.max(2, uniqueSampleIds.length));
  const oofProbs = new Array<number>(minedRows.length).fill(0.5);
  const oofLabels = minedRows.map((r) => r.y);

  for (let fold = 0; fold < kFolds; fold++) {
    const valSampleSet = new Set(
      uniqueSampleIds.filter((_, idx) => idx % kFolds === fold)
    );
    const trainRows = minedRows.filter((r) => !valSampleSet.has(r.sampleId));
    const foldModel = fitStandardizedLogisticRegression(
      trainRows.length >= 4 ? trainRows : minedRows,
      falsePositivePenalty,
      epochs
    );

    for (let i = 0; i < minedRows.length; i++) {
      if (valSampleSet.has(minedRows[i].sampleId)) {
        oofProbs[i] = predictProbability(minedRows[i].features, foldModel);
      }
    }
  }

  let bestThreshold = 0.55;
  let bestObjective = -Infinity;

  for (let t = 0.36; t <= 0.85; t += 0.01) {
    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;
    for (let i = 0; i < minedRows.length; i++) {
      const pred = oofProbs[i] >= t ? 1 : 0;
      const actual = oofLabels[i];
      if (pred === 1 && actual === 1) tp++;
      else if (pred === 1 && actual === 0) fp++;
      else if (pred === 0 && actual === 0) tn++;
      else fn++;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
    const acc = (tp + tn) / Math.max(1, tp + fp + tn + fn);
    const f05 =
      precision + recall > 0
        ? (1.25 * precision * recall) / (0.25 * precision + recall)
        : 0;
    const objective =
      f05 * 0.65 +
      acc * 0.35 -
      fpr * 0.45 -
      (recall < 0.88 ? (0.88 - recall) * 1.6 : 0);
    if (objective > bestObjective) {
      bestObjective = objective;
      bestThreshold = Number(t.toFixed(2));
    }
  }

  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let i = 0; i < minedRows.length; i++) {
    const pred = oofProbs[i] >= bestThreshold ? 1 : 0;
    const actual = oofLabels[i];
    if (pred === 1 && actual === 1) tp++;
    else if (pred === 1 && actual === 0) fp++;
    else if (pred === 0 && actual === 0) tn++;
    else fn++;
  }

  const total = Math.max(1, tp + fp + tn + fn);
  const accuracy = (tp + tn) / total;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const f05Score =
    precision + recall > 0
      ? (1.25 * precision * recall) / (0.25 * precision + recall)
      : 0;
  const falsePositiveRate = fp + tn > 0 ? fp / (fp + tn) : 0;
  const rocAuc = computeRocAuc(oofProbs, oofLabels);

  const fullModel = fitStandardizedLogisticRegression(
    minedRows,
    falsePositivePenalty,
    epochs
  );

  const positiveRegions = minedRows.filter((r) => r.y === 1).length;
  const hardNegativeRegions = minedRows.filter((r) => r.y === 0).length;

  const metrics: ModelEvaluationMetrics = {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1Score: Number(f1Score.toFixed(4)),
    f05Score: Number(f05Score.toFixed(4)),
    falsePositiveRate: Number(falsePositiveRate.toFixed(4)),
    rocAuc: Number(rocAuc.toFixed(4)),
    truePositives: tp,
    falsePositives: fp,
    trueNegatives: tn,
    falseNegatives: fn,
    datasetSamples: dataset.length,
    positiveSamples: dataset.filter((d) => d.label === 'pothole').length,
    negativeSamples: dataset.filter((d) => d.label === 'no_pothole').length,
    candidateRegionsEvaluated: minedRows.length,
    positiveRegions,
    hardNegativeRegions,
    trainingEpochs: epochs,
  };

  return {
    version: (options.previousVersion ?? 0) + 1,
    trainedAt: new Date().toISOString(),
    algorithm:
      'Cost-Sensitive L2 Logistic Regression (Adam) + Hard-Negative Mining + Grouped 5-Fold CV',
    featureNames: [...POTHOLE_FEATURE_NAMES],
    featureMeans: fullModel.means.map((v) => Number(v.toFixed(5))),
    featureStds: fullModel.stds.map((v) => Number(v.toFixed(5))),
    weights: fullModel.weights.map((v) => Number(v.toFixed(5))),
    bias: Number(fullModel.bias.toFixed(5)),
    decisionThreshold: bestThreshold,
    falsePositivePenalty,
    metrics,
  };
}
