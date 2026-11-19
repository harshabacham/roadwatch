import type { BoundingBox, PotholeDetectionItem } from '../types';
import type { VisualFrameSignature } from './roboflowDetector';

interface TrackObservation {
  timeSec: number;
  box: BoundingBox;
}

interface StoredKeyframe {
  timeSec: number;
  signature: VisualFrameSignature;
}

export interface VideoPotholeTrack {
  trackId: string;
  firstSeenTimeSec: number;
  lastSeenTimeSec: number;
  lastBox: BoundingBox;
  /** Continuously propagated by frame-to-frame optical shift even across missed frames. */
  projectedBox: BoundingBox;
  vx: number; // normalized units (0-1000) per second
  vy: number; // normalized units (0-1000) per second
  cropSignature?: Float32Array;
  observations: TrackObservation[];
}

function boxCenter(box: BoundingBox): { x: number; y: number } {
  return {
    x: (box.xMin + box.xMax) / 2,
    y: (box.yMin + box.yMax) / 2,
  };
}

function boxArea(box: BoundingBox): number {
  const w = Math.max(0, box.xMax - box.xMin);
  const h = Math.max(0, box.yMax - box.yMin);
  return w * h;
}

function boxIntersectionArea(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a.xMin, b.xMin);
  const y1 = Math.max(a.yMin, b.yMin);
  const x2 = Math.min(a.xMax, b.xMax);
  const y2 = Math.min(a.yMax, b.yMax);
  if (x2 <= x1 || y2 <= y1) return 0;
  return (x2 - x1) * (y2 - y1);
}

export function computeBoxIoU(a: BoundingBox, b: BoundingBox): number {
  const inter = boxIntersectionArea(a, b);
  if (inter <= 0) return 0;
  const union = boxArea(a) + boxArea(b) - inter;
  return union > 0 ? inter / union : 0;
}

export function computeBoxContainment(a: BoundingBox, b: BoundingBox): number {
  const inter = boxIntersectionArea(a, b);
  if (inter <= 0) return 0;
  const minArea = Math.min(boxArea(a), boxArea(b));
  return minArea > 0 ? inter / minArea : 0;
}

function boxCenterDistance(a: BoundingBox, b: BoundingBox): number {
  const ca = boxCenter(a);
  const cb = boxCenter(b);
  return Math.hypot(ca.x - cb.x, ca.y - cb.y);
}

function padBox(box: BoundingBox, padX: number, padY: number): BoundingBox {
  return {
    xMin: box.xMin - padX,
    yMin: box.yMin - padY,
    xMax: box.xMax + padX,
    yMax: box.yMax + padY,
  };
}

function boxesIntersect(a: BoundingBox, b: BoundingBox): boolean {
  return a.xMin <= b.xMax && a.xMax >= b.xMin && a.yMin <= b.yMax && a.yMax >= b.yMin;
}

function shiftBox(box: BoundingBox, dx: number, dy: number): BoundingBox {
  return {
    xMin: box.xMin + dx,
    yMin: box.yMin + dy,
    xMax: box.xMax + dx,
    yMax: box.yMax + dy,
  };
}

function interpolateBox(a: BoundingBox, b: BoundingBox, alpha: number): BoundingBox {
  const t = Math.max(0, Math.min(1, alpha));
  return {
    xMin: a.xMin + (b.xMin - a.xMin) * t,
    yMin: a.yMin + (b.yMin - a.yMin) * t,
    xMax: a.xMax + (b.xMax - a.xMax) * t,
    yMax: a.yMax + (b.yMax - a.yMax) * t,
  };
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function detectionScore(item: PotholeDetectionItem): number {
  if (typeof item.classifierProbability === 'number') {
    return item.classifierProbability;
  }
  return item.mahalanobisZ / 10;
}

function dotSimilarity(a?: Float32Array, b?: Float32Array): number {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
 * Fast 2D normalized cross-correlation on 24x18 visual frame signatures to measure
 * global camera pan / dashcam road scroll (dx, dy) in normalized 0-1000 coordinates.
 */
function estimateCameraShift(
  prev: VisualFrameSignature,
  curr: VisualFrameSignature
): {
  camDx: number;
  camDy: number;
  bestSimilarity: number;
  staticSimilarity: number;
} {
  const w = prev.width;
  const h = prev.height;
  if (w !== curr.width || h !== curr.height) {
    return { camDx: 0, camDy: 0, bestSimilarity: 0, staticSimilarity: 0 };
  }

  const staticSim = dotSimilarity(prev.data, curr.data);
  let bestSim = staticSim;
  let bestSx = 0;
  let bestSy = 0;

  // Search +/- 42% horizontal pan and -33%..+44% vertical scroll (fast forward road motion)
  const maxSx = Math.min(10, Math.floor(w * 0.42));
  const minSy = -Math.min(6, Math.floor(h * 0.34));
  const maxSy = Math.min(8, Math.floor(h * 0.45));
  const minCells = Math.floor(w * h * 0.42);

  for (let sy = minSy; sy <= maxSy; sy++) {
    const yStart = Math.max(0, -sy);
    const yEnd = Math.min(h, h - sy);
    if (yEnd <= yStart) continue;

    for (let sx = -maxSx; sx <= maxSx; sx++) {
      if (sx === 0 && sy === 0) continue;
      const xStart = Math.max(0, -sx);
      const xEnd = Math.min(w, w - sx);
      const overlapCount = (xEnd - xStart) * (yEnd - yStart);
      if (overlapCount < minCells) continue;

      let dot = 0;
      let normA = 0;
      let normB = 0;

      for (let y = yStart; y < yEnd; y++) {
        const prevRow = y * w;
        const currRow = (y + sy) * w;
        for (let x = xStart; x < xEnd; x++) {
          const va = prev.data[prevRow + x];
          const vb = curr.data[currRow + (x + sx)];
          dot += va * vb;
          normA += va * va;
          normB += vb * vb;
        }
      }

      const denom = Math.sqrt(normA * normB);
      if (denom > 1e-5) {
        const corr = dot / denom;
        if (corr > bestSim) {
          bestSim = corr;
          bestSx = sx;
          bestSy = sy;
        }
      }
    }
  }

  // Only adopt non-zero shift if it meaningfully improves over static alignment
  if (bestSim >= 0.42 && bestSim >= staticSim + 0.025) {
    return {
      camDx: (bestSx / w) * 1000,
      camDy: (bestSy / h) * 1000,
      bestSimilarity: bestSim,
      staticSimilarity: staticSim,
    };
  }

  return {
    camDx: 0,
    camDy: 0,
    bestSimilarity: Math.max(bestSim, staticSim),
    staticSimilarity: staticSim,
  };
}

/**
 * Suppresses overlapping or tightly touching sub-boxes produced within the same frame
 * for the same physical pothole while preserving genuinely distinct potholes.
 */
export function deduplicateSameFrameDetections(
  detections: PotholeDetectionItem[],
  cropSignatures?: Float32Array[]
): { items: PotholeDetectionItem[]; crops: Array<Float32Array | undefined> } {
  if (detections.length <= 1) {
    return {
      items: detections,
      crops: detections.map((_, idx) => cropSignatures?.[idx]),
    };
  }

  const indexed = detections.map((item, idx) => ({
    item,
    crop: cropSignatures?.[idx],
  }));

  indexed.sort((a, b) => {
    const scoreDiff = detectionScore(b.item) - detectionScore(a.item);
    if (Math.abs(scoreDiff) > 0.005) return scoreDiff;
    return boxArea(b.item.box) - boxArea(a.item.box);
  });

  const kept: typeof indexed = [];
  for (const candidate of indexed) {
    const paddedCand = padBox(candidate.item.box, 50, 50);
    const overlapsExisting = kept.some((existing) => {
      if (computeBoxIoU(candidate.item.box, existing.item.box) >= 0.14) return true;
      if (computeBoxContainment(candidate.item.box, existing.item.box) >= 0.35) return true;
      if (boxesIntersect(paddedCand, padBox(existing.item.box, 50, 50))) return true;
      return false;
    });
    if (!overlapsExisting) {
      kept.push(candidate);
    }
  }

  return {
    items: kept.map((k) => k.item),
    crops: kept.map((k) => k.crop),
  };
}

/**
 * Estimates where a historical track was located at a given playback timestamp
 * using linear interpolation across recorded observations (for replay / scrubbing).
 */
function sampleTrackBoxAtTime(
  track: VideoPotholeTrack,
  videoTimeSec: number
): { box: BoundingBox; timeGap: number } | null {
  const obs = track.observations;
  if (obs.length === 0) return null;

  if (videoTimeSec <= obs[0].timeSec) {
    return { box: obs[0].box, timeGap: obs[0].timeSec - videoTimeSec };
  }
  const last = obs[obs.length - 1];
  if (videoTimeSec >= last.timeSec) {
    return { box: last.box, timeGap: videoTimeSec - last.timeSec };
  }

  for (let i = 0; i < obs.length - 1; i++) {
    const curr = obs[i];
    const next = obs[i + 1];
    if (videoTimeSec >= curr.timeSec && videoTimeSec <= next.timeSec) {
      const span = next.timeSec - curr.timeSec;
      if (span <= 0.001) {
        return { box: next.box, timeGap: 0 };
      }
      const alpha = (videoTimeSec - curr.timeSec) / span;
      return {
        box: interpolateBox(curr.box, next.box, alpha),
        timeGap: 0,
      };
    }
  }

  return null;
}

/**
 * Multi-layered video pothole tracker combining:
 * 1. Frame-to-frame optical shift estimation (camera pan & forward dashcam motion),
 *    continuously propagated even across missed/blurred frames.
 * 2. Velocity-predicted + downward perspective-cone matching & visual crop descriptors.
 * 3. Secondary split/fragment suppression so a pothole broken into 2-3 sub-boxes on
 *    closer frames never spawns duplicate complaint entries.
 * 4. Trajectory interpolation & keyframe visual memory for replays, loops, and seeks.
 */
export class VideoPotholeTracker {
  private tracks: VideoPotholeTrack[] = [];
  private nextTrackNum = 1;

  private lastFrameTimeSec: number | null = null;
  private lastFrameSig: VisualFrameSignature | null = null;
  private lastStaticSimilarity = 0;
  private lastMatchedReplayTimeSec: number | null = null;
  private keyframeHistory: StoredKeyframe[] = [];

  public reset(): void {
    this.tracks = [];
    this.nextTrackNum = 1;
    this.lastFrameTimeSec = null;
    this.lastFrameSig = null;
    this.lastStaticSimilarity = 0;
    this.lastMatchedReplayTimeSec = null;
    this.keyframeHistory = [];
  }

  public getTrackCount(): number {
    return this.tracks.length;
  }

  /**
   * Call on EVERY analyzed video frame (including frames with zero detections) so
   * camera motion is tracked seamlessly across flicker/missed frames.
   */
  public observeFrame(
    videoTimeSec: number,
    frameSignature?: VisualFrameSignature
  ): void {
    if (!frameSignature) return;
    if (
      this.lastFrameTimeSec !== null &&
      Math.abs(videoTimeSec - this.lastFrameTimeSec) < 0.005
    ) {
      return;
    }

    this.lastStaticSimilarity = 0;
    this.lastMatchedReplayTimeSec = null;

    if (this.lastFrameSig && this.lastFrameTimeSec !== null) {
      const dt = videoTimeSec - this.lastFrameTimeSec;
      if (dt > 0 && dt <= 3.5) {
        const motion = estimateCameraShift(this.lastFrameSig, frameSignature);
        this.lastStaticSimilarity = motion.staticSimilarity;

        // Propagate every active track's projected box by the measured global camera shift
        if (motion.camDx !== 0 || motion.camDy !== 0) {
          for (const track of this.tracks) {
            if (videoTimeSec - track.lastSeenTimeSec <= 6.0) {
              track.projectedBox = shiftBox(
                track.projectedBox,
                motion.camDx,
                motion.camDy
              );
            }
          }
        }
      }
    }

    // Check if this frame visually repeats an earlier recorded keyframe (loop/replay/rewind)
    let bestHistSim = 0.89;
    for (const kf of this.keyframeHistory) {
      if (Math.abs(videoTimeSec - kf.timeSec) < 0.75) continue;
      const sim = dotSimilarity(frameSignature.data, kf.signature.data);
      if (sim > bestHistSim) {
        bestHistSim = sim;
        this.lastMatchedReplayTimeSec = kf.timeSec;
      }
    }

    // Record keyframe history (at most once every ~0.35s, bounded to 160 keyframes)
    const lastKf = this.keyframeHistory[this.keyframeHistory.length - 1];
    if (!lastKf || Math.abs(videoTimeSec - lastKf.timeSec) >= 0.35) {
      this.keyframeHistory.push({ timeSec: videoTimeSec, signature: frameSignature });
      if (this.keyframeHistory.length > 160) {
        this.keyframeHistory = this.keyframeHistory.filter(
          (_, idx, arr) => idx === 0 || idx === arr.length - 1 || idx % 2 === 0
        );
      }
    }

    this.lastFrameTimeSec = videoTimeSec;
    this.lastFrameSig = frameSignature;
  }

  public filterNewDetections(
    detections: PotholeDetectionItem[],
    videoTimeSec: number,
    frameSignature?: VisualFrameSignature,
    cropSignatures?: Float32Array[]
  ): PotholeDetectionItem[] {
    this.observeFrame(videoTimeSec, frameSignature);

    const { items: uniqueDetections, crops: uniqueCrops } =
      deduplicateSameFrameDetections(detections, cropSignatures);
    if (uniqueDetections.length === 0) return [];

    interface MatchCandidate {
      detIdx: number;
      trackIdx: number;
      cost: number;
      mode: 'active' | 'replay';
    }

    const candidates: MatchCandidate[] = [];

    for (let detIdx = 0; detIdx < uniqueDetections.length; detIdx++) {
      const det = uniqueDetections[detIdx];
      const detCrop = uniqueCrops[detIdx];
      const detCenter = boxCenter(det.box);

      for (let trackIdx = 0; trackIdx < this.tracks.length; trackIdx++) {
        const track = this.tracks[trackIdx];
        const dt = videoTimeSec - track.lastSeenTimeSec;
        const cropSim = dotSimilarity(detCrop, track.cropSignature);

        // Mode 1: Active continuous tracking (handles stationary, panning, dashcam, & flicker)
        if (dt >= -0.5 && dt <= 8.0) {
          const dtPos = Math.max(0, dt);
          const predDx = clamp(track.vx * dtPos, -520, 520);
          const predDy = clamp(track.vy * dtPos, -520, 520);
          const velBox = shiftBox(track.lastBox, predDx, predDy);
          const refBoxes = [track.lastBox, track.projectedBox, velBox];

          let bestIou = 0;
          let bestContainment = 0;
          let bestDist = Infinity;
          let expandedTouch = false;

          const paddedDet = padBox(det.box, 55, 55);
          for (const ref of refBoxes) {
            bestIou = Math.max(bestIou, computeBoxIoU(det.box, ref));
            bestContainment = Math.max(
              bestContainment,
              computeBoxContainment(det.box, ref)
            );
            bestDist = Math.min(bestDist, boxCenterDistance(det.box, ref));
            if (boxesIntersect(paddedDet, padBox(ref, 55, 55))) {
              expandedTouch = true;
            }
          }

          const projCenter = boxCenter(track.projectedBox);
          const dx = detCenter.x - projCenter.x;
          const dy = detCenter.y - projCenter.y;

          const strongOverlap = bestIou >= 0.12 || bestContainment >= 0.35;
          const lightOverlap =
            dt <= 4.5 &&
            (bestIou >= 0.04 || bestContainment >= 0.2 || expandedTouch);

          // Guard: if a track already exited near the bottom (y > 720) and a new box appears
          // near the top (y < 420), they are distinct road defects unless visual crop matches strongly
          const exitedBottomAndNewTop =
            !strongOverlap &&
            projCenter.y > 700 &&
            detCenter.y < 430 &&
            dy < -280 &&
            cropSim < 0.82;

          const visualCropMatch =
            !exitedBottomAndNewTop &&
            dt <= 5.5 &&
            cropSim >= 0.68 &&
            bestDist <= 560;

          const closeProximity =
            !exitedBottomAndNewTop &&
            dt <= 4.0 &&
            bestDist <= 275 + Math.min(dtPos, 2.2) * 110;

          // Forward dashcam perspective motion: road defects rush downward & expand outward
          const dashcamDownwardMotion =
            !exitedBottomAndNewTop &&
            dtPos > 0 &&
            dtPos <= 3.2 &&
            dy >= -95 &&
            dy <= 660 &&
            Math.abs(dx) <= 350;

          // Stationary / slow scene guard when video frame visually barely moved
          const sameSceneGuard =
            !exitedBottomAndNewTop &&
            this.lastStaticSimilarity >= 0.84 &&
            dt <= 6.5 &&
            bestDist <= 420;

          if (
            strongOverlap ||
            lightOverlap ||
            visualCropMatch ||
            closeProximity ||
            dashcamDownwardMotion ||
            sameSceneGuard
          ) {
            const cost =
              bestDist -
              bestIou * 400 -
              bestContainment * 180 -
              Math.max(0, cropSim) * 160 +
              dtPos * 20;
            candidates.push({ detIdx, trackIdx, cost, mode: 'active' });
            continue;
          }
        }

        // Mode 2: Replay / scrub / loop match against recorded trajectory
        const replayCheckTimes: number[] = [videoTimeSec];
        if (this.lastMatchedReplayTimeSec !== null) {
          replayCheckTimes.push(this.lastMatchedReplayTimeSec);
        }

        let bestReplayCost = Infinity;
        for (const checkTime of replayCheckTimes) {
          if (
            checkTime <= track.lastSeenTimeSec + 2.5 &&
            checkTime >= track.firstSeenTimeSec - 2.5
          ) {
            const sampled = sampleTrackBoxAtTime(track, checkTime);
            if (sampled && sampled.timeGap <= 2.5) {
              const iou = computeBoxIoU(det.box, sampled.box);
              const containment = computeBoxContainment(det.box, sampled.box);
              const dist = boxCenterDistance(det.box, sampled.box);
              const paddedTouch = boxesIntersect(
                padBox(det.box, 65, 65),
                padBox(sampled.box, 65, 65)
              );
              if (
                iou >= 0.04 ||
                containment >= 0.2 ||
                paddedTouch ||
                dist <= 320 ||
                (cropSim >= 0.68 && dist <= 520)
              ) {
                const cost =
                  dist -
                  iou * 400 -
                  containment * 160 -
                  Math.max(0, cropSim) * 150 +
                  sampled.timeGap * 55;
                if (cost < bestReplayCost) {
                  bestReplayCost = cost;
                }
              }
            }
          }
        }

        if (bestReplayCost < Infinity) {
          candidates.push({
            detIdx,
            trackIdx,
            cost: bestReplayCost,
            mode: 'replay',
          });
        }
      }
    }

    // Stage 1: Greedy one-to-one primary matching
    candidates.sort((a, b) => a.cost - b.cost);
    const matchedDetIndices = new Set<number>();
    const matchedTrackIndices = new Set<number>();

    for (const cand of candidates) {
      if (
        matchedDetIndices.has(cand.detIdx) ||
        matchedTrackIndices.has(cand.trackIdx)
      ) {
        continue;
      }
      matchedDetIndices.add(cand.detIdx);
      matchedTrackIndices.add(cand.trackIdx);

      const det = uniqueDetections[cand.detIdx];
      const detCrop = uniqueCrops[cand.detIdx];
      const track = this.tracks[cand.trackIdx];

      if (cand.mode === 'active') {
        const dt = videoTimeSec - track.lastSeenTimeSec;
        if (dt > 0.05 && dt <= 3.2) {
          const prevCenter = boxCenter(track.lastBox);
          const nextCenter = boxCenter(det.box);
          const instVx = clamp((nextCenter.x - prevCenter.x) / dt, -700, 700);
          const instVy = clamp((nextCenter.y - prevCenter.y) / dt, -700, 700);
          if (track.observations.length <= 1) {
            track.vx = instVx;
            track.vy = instVy;
          } else {
            track.vx = track.vx * 0.45 + instVx * 0.55;
            track.vy = track.vy * 0.45 + instVy * 0.55;
          }
        }

        if (videoTimeSec >= track.lastSeenTimeSec) {
          track.lastSeenTimeSec = videoTimeSec;
          track.lastBox = det.box;
          track.projectedBox = det.box;
          if (detCrop) {
            track.cropSignature = detCrop;
          }
          const lastObs = track.observations[track.observations.length - 1];
          if (!lastObs || videoTimeSec - lastObs.timeSec >= 0.15) {
            track.observations.push({ timeSec: videoTimeSec, box: det.box });
            if (track.observations.length > 140) {
              track.observations = track.observations.filter(
                (_, idx, arr) => idx === 0 || idx === arr.length - 1 || idx % 2 === 0
              );
            }
          }
        }
      }
    }

    // Stage 2: Secondary split/fragment suppression.
    // If Roboflow split one previously-tracked pothole into 2-3 sub-boxes on this frame,
    // suppress the extra sub-boxes instead of creating duplicate complaints.
    for (let detIdx = 0; detIdx < uniqueDetections.length; detIdx++) {
      if (matchedDetIndices.has(detIdx)) continue;
      const det = uniqueDetections[detIdx];
      const detCrop = uniqueCrops[detIdx];
      const paddedDet = padBox(det.box, 70, 70);

      const isSplitOfExistingTrack = this.tracks.some((track) => {
        const dt = videoTimeSec - track.lastSeenTimeSec;
        if (dt < -0.5 || dt > 5.0) return false;

        const refs = [track.lastBox, track.projectedBox];
        for (const ref of refs) {
          if (computeBoxIoU(det.box, ref) >= 0.04) return true;
          if (computeBoxContainment(det.box, ref) >= 0.22) return true;
          if (boxesIntersect(paddedDet, padBox(ref, 70, 70))) return true;
        }
        if (
          dotSimilarity(detCrop, track.cropSignature) >= 0.74 &&
          boxCenterDistance(det.box, track.projectedBox) <= 460
        ) {
          return true;
        }
        return false;
      });

      if (isSplitOfExistingTrack) {
        matchedDetIndices.add(detIdx);
      }
    }

    // Stage 3: Create new tracks ONLY for genuinely unmatched new potholes
    const newDetections: PotholeDetectionItem[] = [];
    for (let detIdx = 0; detIdx < uniqueDetections.length; detIdx++) {
      if (matchedDetIndices.has(detIdx)) continue;
      const det = uniqueDetections[detIdx];
      const detCrop = uniqueCrops[detIdx];
      newDetections.push(det);
      this.tracks.push({
        trackId: `vtrack_${this.nextTrackNum++}`,
        firstSeenTimeSec: videoTimeSec,
        lastSeenTimeSec: videoTimeSec,
        lastBox: det.box,
        projectedBox: det.box,
        vx: 0,
        vy: 0,
        cropSignature: detCrop,
        observations: [{ timeSec: videoTimeSec, box: det.box }],
      });
    }

    return newDetections;
  }
}
