import {
  detectPotholesWithRoboflow,
  type RoboflowDetectionOutput,
} from './roboflowDetector';
import type { UnsupervisedParams } from './unsupervisedDetector';

export interface VideoStreamConfig {
  /** Process 1 out of every N decoded video frames (e.g. 5 = skip 4 frames between detections). */
  frameSkipInterval: number;
  /** Maximum width/height in pixels when downsampling video frames prior to inference. */
  maxDimension: number;
  /** JPEG compression quality (0.5 - 0.95) for downsampled video frames. */
  jpegQuality: number;
  /** Minimum wall-clock gap (ms) between inference dispatches to cap high-FPS (60/120fps) streams. */
  minIntervalMs: number;
}

export interface VideoStreamStats {
  totalFramesSeen: number;
  framesProcessed: number;
  framesSkipped: number;
  droppedBackpressure: number;
  downsampledWidth: number;
  downsampledHeight: number;
  sourceWidth: number;
  sourceHeight: number;
  frameSkipInterval: number;
}

export interface VideoStreamCallbacks {
  getParams: () => UnsupervisedParams;
  getConfig: () => VideoStreamConfig;
  onResult: (
    result: RoboflowDetectionOutput,
    stats: VideoStreamStats,
    videoTimeSec: number
  ) => void;
  onStats?: (stats: VideoStreamStats) => void;
  onError?: (error: Error) => void;
  onStopped?: () => void;
}

type VideoFrameCallbackHandle = number;
type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: DOMHighResTimeStamp, metadata: { mediaTime: number; presentedFrames: number }) => void
  ) => VideoFrameCallbackHandle;
  cancelVideoFrameCallback?: (handle: VideoFrameCallbackHandle) => void;
};

/**
 * Explicit video-stream detection pipeline that:
 * 1. Synchronizes with decoded video frames via requestVideoFrameCallback (with rAF fallback).
 * 2. Applies a configurable frame-skip interval plus in-flight backpressure dropping so
 *    playback remains smooth and inference never queues stale frames.
 * 3. Downsamples frames to a target maxDimension & JPEG quality before Roboflow inference,
 *    keeping bounding boxes and review snapshots locked to the exact analyzed frame.
 */
export class VideoStreamDetectionPipeline {
  private video: VideoWithFrameCallback | null = null;
  private callbacks: VideoStreamCallbacks | null = null;
  private active = false;
  private inFlight = false;
  private generation = 0;

  private rvfcHandle: VideoFrameCallbackHandle | null = null;
  private rafHandle: number | null = null;

  private framesSinceLastProcess = 0;
  private lastDispatchTimeMs = 0;
  private lastObservedMediaTime = -1;
  private lastPresentedFrames = -1;

  private stats: VideoStreamStats = {
    totalFramesSeen: 0,
    framesProcessed: 0,
    framesSkipped: 0,
    droppedBackpressure: 0,
    downsampledWidth: 0,
    downsampledHeight: 0,
    sourceWidth: 0,
    sourceHeight: 0,
    frameSkipInterval: 5,
  };

  public isRunning(): boolean {
    return this.active;
  }

  public getStats(): VideoStreamStats {
    return { ...this.stats };
  }

  public start(
    videoElement: HTMLVideoElement,
    callbacks: VideoStreamCallbacks,
    resetStats = true
  ): void {
    this.stop(false);

    this.video = videoElement as VideoWithFrameCallback;
    this.callbacks = callbacks;
    this.active = true;
    this.inFlight = false;
    this.generation += 1;

    const cfg = callbacks.getConfig();
    if (resetStats) {
      this.stats = {
        totalFramesSeen: 0,
        framesProcessed: 0,
        framesSkipped: 0,
        droppedBackpressure: 0,
        downsampledWidth: 0,
        downsampledHeight: 0,
        sourceWidth: videoElement.videoWidth || 0,
        sourceHeight: videoElement.videoHeight || 0,
        frameSkipInterval: Math.max(1, Math.round(cfg.frameSkipInterval)),
      };
    } else {
      this.stats.frameSkipInterval = Math.max(1, Math.round(cfg.frameSkipInterval));
    }

    // Force the very first decoded frame to be processed immediately
    this.framesSinceLastProcess = this.stats.frameSkipInterval;
    this.lastDispatchTimeMs = 0;
    this.lastObservedMediaTime = -1;
    this.lastPresentedFrames = -1;

    this.scheduleNextFrame();
  }

  public stop(notify = true): void {
    const wasActive = this.active;
    this.active = false;
    this.generation += 1;

    if (
      this.rvfcHandle !== null &&
      this.video?.cancelVideoFrameCallback
    ) {
      try {
        this.video.cancelVideoFrameCallback(this.rvfcHandle);
      } catch {
        // Ignore if video element detached
      }
    }
    this.rvfcHandle = null;

    if (this.rafHandle !== null) {
      window.cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }

    const cb = this.callbacks;
    this.video = null;
    this.callbacks = null;

    if (notify && wasActive) {
      cb?.onStopped?.();
    }
  }

  private scheduleNextFrame(): void {
    if (!this.active || !this.video) return;

    if (typeof this.video.requestVideoFrameCallback === 'function') {
      this.rvfcHandle = this.video.requestVideoFrameCallback((now, metadata) => {
        this.rvfcHandle = null;
        this.handleDecodedFrame(now, metadata.mediaTime, metadata.presentedFrames);
      });
    } else {
      this.rafHandle = window.requestAnimationFrame((now) => {
        this.rafHandle = null;
        if (!this.video) return;
        const currentTime = this.video.currentTime;
        // Skip duplicate rAF ticks when mediaTime hasn't advanced on uploaded videos
        const isLiveCamera = this.video.srcObject !== null;
        if (
          !isLiveCamera &&
          Math.abs(currentTime - this.lastObservedMediaTime) < 0.004
        ) {
          this.scheduleNextFrame();
          return;
        }
        this.handleDecodedFrame(now, currentTime, this.stats.totalFramesSeen + 1);
      });
    }
  }

  private handleDecodedFrame(
    nowMs: number,
    mediaTimeSec: number,
    presentedFrames: number
  ): void {
    if (!this.active || !this.video || !this.callbacks) return;

    const video = this.video;
    const isLiveCamera = video.srcObject !== null;

    // Auto-stop if an uploaded video paused or reached the end
    if (!isLiveCamera && (video.paused || video.ended)) {
      this.stop(true);
      return;
    }

    // Compute how many video frames elapsed since the previous callback
    const frameDelta =
      this.lastPresentedFrames >= 0 && presentedFrames > this.lastPresentedFrames
        ? presentedFrames - this.lastPresentedFrames
        : 1;
    this.lastPresentedFrames = presentedFrames;
    this.lastObservedMediaTime = mediaTimeSec;

    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      this.scheduleNextFrame();
      return;
    }

    const cfg = this.callbacks.getConfig();
    const skipInterval = Math.max(1, Math.round(cfg.frameSkipInterval));
    this.stats.frameSkipInterval = skipInterval;
    this.stats.totalFramesSeen += frameDelta;
    this.stats.sourceWidth = video.videoWidth;
    this.stats.sourceHeight = video.videoHeight;

    this.framesSinceLastProcess += frameDelta;
    const elapsedSinceDispatch = nowMs - this.lastDispatchTimeMs;

    const shouldSkipByInterval =
      this.framesSinceLastProcess < skipInterval ||
      (this.lastDispatchTimeMs > 0 && elapsedSinceDispatch < cfg.minIntervalMs);

    if (shouldSkipByInterval) {
      this.stats.framesSkipped += frameDelta;
      this.callbacks.onStats?.(this.getStats());
      this.scheduleNextFrame();
      return;
    }

    if (this.inFlight) {
      this.stats.framesSkipped += frameDelta;
      this.stats.droppedBackpressure += frameDelta;
      this.callbacks.onStats?.(this.getStats());
      this.scheduleNextFrame();
      return;
    }

    // Dispatch this downsampled frame through the detection pipeline
    this.framesSinceLastProcess = 0;
    this.lastDispatchTimeMs = nowMs;
    this.inFlight = true;

    const runGen = this.generation;
    const callbacks = this.callbacks;
    const params = callbacks.getParams();
    const capturedVideoTime = Number.isFinite(video.currentTime)
      ? video.currentTime
      : mediaTimeSec;

    // Schedule next frame listener immediately so we keep accurately counting skipped frames
    this.scheduleNextFrame();

    void detectPotholesWithRoboflow(video, params, {
      isVideoStream: true,
      maxDimension: cfg.maxDimension,
      jpegQuality: cfg.jpegQuality,
    })
      .then((result) => {
        if (!this.active || runGen !== this.generation || !result) return;
        this.stats.framesProcessed += 1;
        if (result.downsampledSize) {
          this.stats.downsampledWidth = result.downsampledSize.width;
          this.stats.downsampledHeight = result.downsampledSize.height;
          this.stats.sourceWidth = result.downsampledSize.sourceWidth;
          this.stats.sourceHeight = result.downsampledSize.sourceHeight;
        }
        const snapshotStats = this.getStats();
        callbacks.onStats?.(snapshotStats);
        callbacks.onResult(result, snapshotStats, capturedVideoTime);
      })
      .catch((err: unknown) => {
        if (!this.active || runGen !== this.generation) return;
        callbacks.onError?.(
          err instanceof Error ? err : new Error(String(err))
        );
      })
      .finally(() => {
        if (runGen === this.generation) {
          this.inFlight = false;
        }
      });
  }
}
