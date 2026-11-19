import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  Check,
  CheckCheck,
  Cpu,
  Download,
  ExternalLink,
  Flashlight,
  Image as ImageIcon,
  Layers,
  MapPin,
  Pause,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Square,
  Upload,
  Video,
  X,
} from 'lucide-react';
import {
  ComplaintRecord,
  DetectedPotholeCandidate,
  DetectionResult,
  GeoCoordinates,
  GeocodedLocation,
  Open311Config,
} from '../types';
import {
  buildComplaintPayloadFromCluster,
  COMPLAINT_CLUSTER_RADIUS_METERS,
  formatDuration,
  getSeverityMeta,
  groupCandidatesByRadius,
  haversineMeters,
} from '../utils/geo';
import {
  analyzeFrameUnsupervised,
  ClusterProfile,
  DiagnosticOverlayMode,
  getLearnedRoadBaseline,
  LearnedRoadBaseline,
  resetLearnedRoadBaseline,
  SensitivityPreset,
  trainRoadBaselineFromSource,
  UnsupervisedMetrics,
  UnsupervisedParams,
} from '../ml/unsupervisedDetector';
import {
  clearFinishedDetections,
  getAllDetectionCandidates,
  pendingQueueItemToComplaintRecord,
  saveDetectionCandidate,
  updateDetectionCandidate,
  upsertCachedComplaint,
} from '../offline/indexedDb';
import {
  queueComplaintForOfflineSync,
  subscribeToOfflineSync,
} from '../offline/syncManager';

import {
  detectPotholesWithRoboflow,
  type VisualFrameSignature,
} from '../ml/roboflowDetector';
import {
  deduplicateSameFrameDetections,
  VideoPotholeTracker,
} from '../ml/videoPotholeTracker';
import {
  VideoStreamDetectionPipeline,
  type VideoStreamConfig,
  type VideoStreamStats,
} from '../ml/videoStreamProcessor';
import severeSampleUrl from '../assets/images/road_sample_severe_1795005146352.jpg';
import moderateSampleUrl from '../assets/images/road_sample_moderate_1795005146362.jpg';
import waterloggedSampleUrl from '../assets/images/road_sample_waterlogged_1795005147911.jpg';
import highwaySampleUrl from '../assets/images/road_sample_highway_pothole_1795023499964.jpg';
import residentialSampleUrl from '../assets/images/road_sample_residential_crack_1795023503887.jpg';
import cleanAsphaltSampleUrl from '../assets/images/road_sample_clean_asphalt_1795023504230.jpg';

type SourceMode = 'camera' | 'upload' | 'samples';

const ROAD_SAMPLES = [
  {
    id: 'severe',
    label: 'Deep Urban Pothole',
    subtitle: 'Exposed aggregate cavity',
    url: severeSampleUrl,
  },
  {
    id: 'highway',
    label: 'Highway Lane Pothole',
    subtitle: 'Sharp-edged broken crater',
    url: highwaySampleUrl,
  },
  {
    id: 'moderate',
    label: 'Avenue Lane Hazards',
    subtitle: 'Multiple cracks & cavities',
    url: moderateSampleUrl,
  },
  {
    id: 'residential',
    label: 'Residential Curb Hole',
    subtitle: 'Crumbling curb pavement',
    url: residentialSampleUrl,
  },
  {
    id: 'waterlogged',
    label: 'Waterlogged Curb Hole',
    subtitle: 'Rain-filled edge collapse',
    url: waterloggedSampleUrl,
  },
  {
    id: 'clean',
    label: 'Clean Asphalt Road',
    subtitle: 'Intact road (0 hazards)',
    url: cleanAsphaltSampleUrl,
  },
];

const OVERLAY_MODES: Array<{ id: DiagnosticOverlayMode; label: string }> = [
  { id: 'detections', label: 'Hazard Boxes' },
  { id: 'kmeans', label: 'Surface Zones' },
  { id: 'heatmap', label: 'Damage Heatmap' },
  { id: 'dbscan', label: 'Depth & Edges' },
];

interface LiveScannerProps {
  currentLocation: GeoCoordinates | null;
  geocoded: GeocodedLocation | null;
  gpsStatus: 'acquiring' | 'locked' | 'denied' | 'unavailable';
  gpsError: string | null;
  onRetryGps: () => void;
  onSetManualLocation: (lat: number, lng: number) => void;
  open311Config: Open311Config;
  onComplaintCreated: (complaint: ComplaintRecord) => void;
  onOpenComplaint: (complaint: ComplaintRecord) => void;
  onDetectionsChanged?: () => void;
  externalImage?: { url: string; label: string } | null;
}

export function LiveScanner({
  currentLocation,
  geocoded,
  gpsStatus,
  gpsError,
  onRetryGps,
  onSetManualLocation,
  open311Config,
  onComplaintCreated,
  onOpenComplaint,
  onDetectionsChanged,
  externalImage,
}: LiveScannerProps) {
  const [sourceMode, setSourceMode] = useState<SourceMode>('camera');
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [cameraStatus, setCameraStatus] = useState<
    'idle' | 'requesting' | 'ready' | 'error'
  >('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [torchSupported, setTorchSupported] = useState(false);
  const [torchOn, setTorchOn] = useState(false);

  // Unsupervised ML Hyperparameters & Online Road Baseline State
  const [mlParams, setMlParams] = useState<UnsupervisedParams>({
    kClusters: 4,
    sensitivity: 'balanced',
    dbscanMinPts: 4,
    overlayMode: 'detections',
  });
  const [showHyperparams, setShowHyperparams] = useState(false);
  const [learnedBaseline, setLearnedBaseline] = useState<LearnedRoadBaseline>(
    () => getLearnedRoadBaseline()
  );
  const [baselineNotice, setBaselineNotice] = useState<string | null>(null);

  // Upload & sample media state
  const [selectedSampleId, setSelectedSampleId] = useState<string>('severe');
  const [uploadedFileUrl, setUploadedFileUrl] = useState<string | null>(null);
  const [uploadedKind, setUploadedKind] = useState<'image' | 'video'>('image');
  const [uploadedVideoDims, setUploadedVideoDims] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [videoOverlayRect, setVideoOverlayRect] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

  // Video stream downsampling & frame-skip pipeline settings
  const [videoStreamConfig, setVideoStreamConfig] = useState<VideoStreamConfig>({
    frameSkipInterval: 5,
    maxDimension: 640,
    jpegQuality: 0.72,
    minIntervalMs: 160,
  });
  const [videoStreamStats, setVideoStreamStats] =
    useState<VideoStreamStats | null>(null);

  // Recording & continuous scanning state
  const [isRecording, setIsRecording] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [videoFramesScanned, setVideoFramesScanned] = useState(0);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  // Live Detection telemetry & Detected Potholes List for User Acceptance
  const [lastDetection, setLastDetection] = useState<DetectionResult | null>(null);
  const [mlMetrics, setMlMetrics] = useState<UnsupervisedMetrics | null>(null);
  const [clusterProfiles, setClusterProfiles] = useState<ClusterProfile[]>([]);
  const [overlayDataUrl, setOverlayDataUrl] = useState<string>('');
  const [detectedList, setDetectedList] = useState<DetectedPotholeCandidate[]>([]);
  const [raisingAll, setRaisingAll] = useState(false);

  // Manual GPS editor
  const [editingCoords, setEditingCoords] = useState(false);
  const [manualLat, setManualLat] = useState('37.77493');
  const [manualLng, setManualLng] = useState('-122.41942');

  const viewfinderRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const uploadedVideoRef = useRef<HTMLVideoElement | null>(null);
  const sampleImgRef = useRef<HTMLImageElement | null>(null);
  const uploadedImgRef = useRef<HTMLImageElement | null>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const stopCameraOnRecorderStopRef = useRef(false);
  const recordedChunksRef = useRef<Blob[]>([]);
  const videoPipelineRef = useRef<VideoStreamDetectionPipeline | null>(null);
  const durationTimerRef = useRef<number | null>(null);
  const consecutiveHitsRef = useRef(0);
  const scanInFlightRef = useRef(false);
  const scanGenerationRef = useRef(0);
  const sessionCapturedRef = useRef<
    Array<{
      lat: number;
      lng: number;
      timestamp: number;
    }>
  >([]);
  const videoTrackersByFileRef = useRef<Map<string, VideoPotholeTracker>>(
    new Map()
  );
  const videoTrackerRef = useRef<VideoPotholeTracker>(new VideoPotholeTracker());
  const cameraTrackerRef = useRef<VideoPotholeTracker>(new VideoPotholeTracker());
  const cameraSessionStartMsRef = useRef<number>(Date.now());

  const locationRef = useRef(currentLocation);
  locationRef.current = currentLocation;
  const geocodedRef = useRef(geocoded);
  geocodedRef.current = geocoded;
  const open311Ref = useRef(open311Config);
  open311Ref.current = open311Config;
  const mlParamsRef = useRef(mlParams);
  mlParamsRef.current = mlParams;
  const videoStreamConfigRef = useRef(videoStreamConfig);
  videoStreamConfigRef.current = videoStreamConfig;
  const sourceModeRef = useRef(sourceMode);
  sourceModeRef.current = sourceMode;
  const uploadedKindRef = useRef(uploadedKind);
  uploadedKindRef.current = uploadedKind;

  const stopCameraStream = useCallback((resetToIdle = true) => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setTorchSupported(false);
    setTorchOn(false);
    if (resetToIdle) {
      setCameraStatus('idle');
    }
  }, []);

  const startCamera = useCallback(
    async (
      preferredFacing: 'environment' | 'user'
    ): Promise<MediaStream | null> => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraStatus('error');
        setCameraError(
          'Camera API is unavailable in this browser context. Use Upload Media or Road Samples below.'
        );
        return null;
      }

      stopCameraStream(false);
      setCameraStatus('requesting');
      setCameraError(null);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: preferredFacing },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        if (sourceModeRef.current !== 'camera') {
          stream.getTracks().forEach((track) => track.stop());
          setCameraStatus('idle');
          return null;
        }
        streamRef.current = stream;

        const videoEl = videoRef.current;
        if (videoEl) {
          videoEl.srcObject = stream;
          await videoEl.play().catch(() => undefined);
          if (videoEl.videoWidth === 0) {
            await new Promise<void>((resolve) => {
              const onReady = () => {
                videoEl.removeEventListener('loadedmetadata', onReady);
                resolve();
              };
              videoEl.addEventListener('loadedmetadata', onReady, { once: true });
              window.setTimeout(() => {
                videoEl.removeEventListener('loadedmetadata', onReady);
                resolve();
              }, 1500);
            });
          }
        }

        const track = stream.getVideoTracks()[0];
        const caps = track?.getCapabilities?.() as Record<string, unknown> | undefined;
        setTorchSupported(Boolean(caps && 'torch' in caps));
        setCameraStatus('ready');
        return stream;
      } catch (err: any) {
        setCameraStatus('error');
        setCameraError(
          err?.name === 'NotAllowedError'
            ? 'Camera access was blocked. Allow camera permission in browser settings or tap Retry.'
            : err?.message ||
                'Unable to access camera. Switch to Road Samples or Upload Media to test detection.'
        );
        return null;
      }
    },
    [stopCameraStream]
  );

  // Stop any active camera stream when switching away from camera mode or unmounting
  useEffect(() => {
    if (sourceMode !== 'camera') {
      stopCameraStream(true);
    }
    return () => {
      stopCameraStream(true);
    };
  }, [sourceMode, stopCameraStream]);

  useEffect(() => {
    if (!externalImage?.url) return;
    setUploadedFileUrl((prev) => {
      if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
      return externalImage.url;
    });
    setUploadedKind('image');
    setSourceMode('upload');
    setLastDetection(null);
    setMlMetrics(null);
    setClusterProfiles([]);
    setOverlayDataUrl('');
    setScanError(null);
  }, [externalImage]);

  useEffect(() => {
    let active = true;
    void (async () => {
      const saved = await getAllDetectionCandidates();
      if (active && saved.length > 0) {
        setDetectedList(saved);
      }
    })();

    const unsubscribe = subscribeToOfflineSync((event) => {
      if (event.type === 'COMPLAINT_SYNCED' && event.candidateId) {
        setDetectedList((prev) =>
          prev.map((item) =>
            item.candidateId === event.candidateId
              ? {
                  ...item,
                  decision: 'raised',
                  raisedComplaint: event.complaint,
                  error: undefined,
                }
              : item
          )
        );
      }
    });

    return () => {
      active = false;
      unsubscribe();
      videoPipelineRef.current?.stop(false);
      if (durationTimerRef.current) window.clearInterval(durationTimerRef.current);
    };
  }, []);

  // Align SVG bounding boxes accurately over object-contain uploaded videos (portrait or landscape)
  useEffect(() => {
    if (
      sourceMode !== 'upload' ||
      uploadedKind !== 'video' ||
      !uploadedVideoDims
    ) {
      setVideoOverlayRect(null);
      return;
    }
    const container = viewfinderRef.current;
    if (!container) return;

    const updateRect = () => {
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      const { width: mw, height: mh } = uploadedVideoDims;
      if (!cw || !ch || !mw || !mh) return;
      const scale = Math.min(cw / mw, ch / mh);
      const rw = mw * scale;
      const rh = mh * scale;
      setVideoOverlayRect({
        left: (cw - rw) / 2,
        top: (ch - rh) / 2,
        width: rw,
        height: rh,
      });
    };

    updateRect();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(updateRect);
      ro.observe(container);
      return () => ro.disconnect();
    }
    window.addEventListener('resize', updateRect);
    return () => window.removeEventListener('resize', updateRect);
  }, [sourceMode, uploadedKind, uploadedVideoDims]);

  async function toggleTorch() {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) return;
    const next = !torchOn;
    try {
      await track.applyConstraints({
        advanced: [{ torch: next } as MediaTrackConstraintSet],
      });
      setTorchOn(next);
    } catch {
      // Torch unsupported
    }
  }

  function getActiveMediaElement():
    | HTMLVideoElement
    | HTMLImageElement
    | null {
    const mode = sourceModeRef.current;
    const kind = uploadedKindRef.current;
    if (
      mode === 'camera' &&
      videoRef.current &&
      videoRef.current.readyState >= 2 &&
      videoRef.current.videoWidth > 0
    ) {
      return videoRef.current;
    }
    if (mode === 'upload') {
      if (
        kind === 'video' &&
        uploadedVideoRef.current &&
        uploadedVideoRef.current.readyState >= 2 &&
        uploadedVideoRef.current.videoWidth > 0
      ) {
        return uploadedVideoRef.current;
      }
      if (
        kind === 'image' &&
        uploadedImgRef.current &&
        uploadedImgRef.current.naturalWidth > 0
      ) {
        return uploadedImgRef.current;
      }
    }
    if (
      mode === 'samples' &&
      sampleImgRef.current &&
      sampleImgRef.current.naturalWidth > 0
    ) {
      return sampleImgRef.current;
    }
    return null;
  }

  function captureSnapshotDataUrl(
    sourceEl: HTMLVideoElement | HTMLImageElement
  ): string {
    if (sourceEl instanceof HTMLVideoElement && sourceEl.readyState < 2) {
      return '';
    }
    const canvas = snapshotCanvasRef.current || document.createElement('canvas');
    snapshotCanvasRef.current = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    const width =
      sourceEl instanceof HTMLVideoElement
        ? sourceEl.videoWidth
        : sourceEl.naturalWidth;
    const height =
      sourceEl instanceof HTMLVideoElement
        ? sourceEl.videoHeight
        : sourceEl.naturalHeight;

    if (!width || !height) return '';

    const maxDim = 900;
    const scale = Math.min(1, maxDim / Math.max(width, height));
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    ctx.drawImage(sourceEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.82);
  }

  const addCandidateToReviewList = useCallback(
    (
      detection: DetectionResult,
      snapshotDataUrl: string,
      isStaticSingleScan: boolean,
      capturedVideoTimeSec?: number,
      frameSignature?: VisualFrameSignature,
      cropSignatures?: Float32Array[]
    ) => {
      if (!detection.potholeDetected || detection.detections.length === 0) return;

      const loc = locationRef.current;
      const geo = geocodedRef.current;
      const lat = loc?.lat ?? geo?.lat ?? 37.77493;
      const lng = loc?.lng ?? geo?.lng ?? -122.41942;
      const nowMs = Date.now();
      const isUploadedVideo =
        sourceModeRef.current === 'upload' && uploadedKindRef.current === 'video';
      const videoTimeSec = isUploadedVideo
        ? capturedVideoTimeSec ?? uploadedVideoRef.current?.currentTime ?? 0
        : undefined;

      let itemsToRecord = detection.detections;

      if (isUploadedVideo && videoTimeSec !== undefined) {
        // Track physical potholes continuously across video frames & replays using
        // optical shift, velocity, split-box suppression, and visual crop descriptors
        itemsToRecord = videoTrackerRef.current.filterNewDetections(
          detection.detections,
          videoTimeSec,
          frameSignature,
          cropSignatures
        );
        if (itemsToRecord.length === 0) return;
      } else if (sourceModeRef.current === 'camera' && !isStaticSingleScan) {
        const elapsedSec = (nowMs - cameraSessionStartMsRef.current) / 1000;
        itemsToRecord = cameraTrackerRef.current.filterNewDetections(
          detection.detections,
          elapsedSec,
          frameSignature,
          cropSignatures
        );
        if (itemsToRecord.length === 0) return;

        // Also guard against GPS-stationary re-logging within 12m & 8s
        const recentSameSpot = sessionCapturedRef.current.some(
          (entry) =>
            haversineMeters(entry.lat, entry.lng, lat, lng) < 12 &&
            nowMs - entry.timestamp < 8000
        );
        if (recentSameSpot) return;
        sessionCapturedRef.current.push({ lat, lng, timestamp: nowMs });
      } else {
        itemsToRecord = deduplicateSameFrameDetections(
          detection.detections,
          cropSignatures
        ).items;
        if (itemsToRecord.length === 0) return;
        sessionCapturedRef.current.push({ lat, lng, timestamp: nowMs });
      }

      const timeTag =
        videoTimeSec !== undefined
          ? ` [Video @ ${formatDuration(Math.floor(videoTimeSec))}]`
          : '';

      const newCandidates: DetectedPotholeCandidate[] = itemsToRecord.map(
        (item, idx) => ({
          candidateId: `cand_${nowMs}_${idx}_${Math.floor(Math.random() * 999)}`,
          detectedAt: new Date().toISOString(),
          detection: {
            ...detection,
            overallSeverity: item.severity,
            detections: [item],
            formalComplaintTitle: `${item.severity} Pothole (${item.estimatedWidthCm} cm × ${item.estimatedDepthCm} cm deep)${timeTag}`,
            formalComplaintDescription: `${item.hazardDescription} Estimated depth ${item.depthConfidenceRange} (${item.depthClass}), surface area ~${item.estimatedAreaSqCm} cm².`,
          },
          primaryItem: item,
          snapshotDataUrl,
          lat,
          lng,
          accuracyMeters: loc?.accuracy ?? null,
          address: geo?.formattedAddress || `GPS ${lat.toFixed(5)}, ${lng.toFixed(5)}`,
          city: geo?.city || '',
          state: geo?.state || '',
          countryCode: geo?.countryCode || '',
          portalName: geo?.portal?.portalName || 'Municipal Public Works Portal',
          portalUrl: geo?.portal?.portalUrl || 'https://www.fixmystreet.com/',
          department:
            geo?.portal?.department ||
            detection.recommendedDepartment ||
            'Department of Transportation',
          decision: 'pending',
        })
      );

      setDetectedList((prev) => [...newCandidates, ...prev]);
      void Promise.all(
        newCandidates.map((cand) => saveDetectionCandidate(cand))
      ).then(() => {
        onDetectionsChanged?.();
      });
    },
    [onDetectionsChanged]
  );

  const clearScanVisuals = useCallback(() => {
    scanGenerationRef.current += 1;
    scanInFlightRef.current = false;
    consecutiveHitsRef.current = 0;
    setIsScanning(false);
    setLastDetection(null);
    setMlMetrics(null);
    setClusterProfiles([]);
    setOverlayDataUrl('');
    setScanError(null);
  }, []);

  const runDetectionPass = useCallback(
    async (recordToList: boolean, isContinuousStream: boolean) => {
      if (scanInFlightRef.current) return;
      const mediaEl = getActiveMediaElement();
      if (!mediaEl) {
        if (!isContinuousStream) {
          setScanError(
            'Media not ready yet. Wait for camera or road image to finish loading.'
          );
        }
        return;
      }

      const gen = scanGenerationRef.current;
      scanInFlightRef.current = true;
      if (!isContinuousStream) setIsScanning(true);
      setScanError(null);

      try {
        const isVideoEl = mediaEl instanceof HTMLVideoElement;
        const capturedTime =
          isVideoEl &&
          sourceModeRef.current === 'upload' &&
          uploadedKindRef.current === 'video'
            ? mediaEl.currentTime
            : undefined;
        const result = await detectPotholesWithRoboflow(
          mediaEl,
          mlParamsRef.current,
          isVideoEl
            ? {
                isVideoStream: true,
                maxDimension: videoStreamConfigRef.current.maxDimension,
                jpegQuality: videoStreamConfigRef.current.jpegQuality,
              }
            : undefined
        );
        if (!result || gen !== scanGenerationRef.current) return;

        setLastDetection(result.detection);
        setMlMetrics(result.metrics);
        setClusterProfiles(result.clusters);
        setOverlayDataUrl(result.overlayDataUrl);
        setLearnedBaseline(getLearnedRoadBaseline());

        if (
          capturedTime !== undefined &&
          result.frameSignature &&
          sourceModeRef.current === 'upload' &&
          uploadedKindRef.current === 'video'
        ) {
          videoTrackerRef.current.observeFrame(capturedTime, result.frameSignature);
        }

        if (result.detection.potholeDetected) {
          consecutiveHitsRef.current += 1;
          const confirmed =
            !isContinuousStream ||
            result.engine === 'roboflow' ||
            consecutiveHitsRef.current >= 2;

          if (recordToList && confirmed) {
            const snap = result.snapshotDataUrl || captureSnapshotDataUrl(mediaEl);
            addCandidateToReviewList(
              result.detection,
              snap,
              !isContinuousStream,
              capturedTime,
              result.frameSignature,
              result.cropSignatures
            );
          }
        } else {
          consecutiveHitsRef.current = 0;
        }
      } catch (err: any) {
        if (gen === scanGenerationRef.current && !isContinuousStream) {
          setScanError(err?.message || 'Pothole detection request failed.');
        }
      } finally {
        scanInFlightRef.current = false;
        if (gen === scanGenerationRef.current) {
          setIsScanning(false);
        }
      }
    },
    [sourceMode, uploadedKind, addCandidateToReviewList]
  );

  // Recompute pixel overlay visualization when user switches overlay mode on static sample/image
  useEffect(() => {
    if (!isRecording && lastDetection) {
      const mediaEl = getActiveMediaElement();
      if (!mediaEl) return;
      if (mlParams.overlayMode === 'detections') {
        setOverlayDataUrl('');
        return;
      }
      const updated = analyzeFrameUnsupervised(mediaEl, mlParams);
      if (updated) {
        setOverlayDataUrl(updated.overlayDataUrl);
        setClusterProfiles(updated.clusters);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mlParams.overlayMode]);

  function stopContinuousScanLoop() {
    videoPipelineRef.current?.stop(false);
    if (durationTimerRef.current) {
      window.clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
  }

  function startVideoStreamPipeline(
    videoEl: HTMLVideoElement,
    resetStats: boolean
  ) {
    stopContinuousScanLoop();
    if (!videoPipelineRef.current) {
      videoPipelineRef.current = new VideoStreamDetectionPipeline();
    }

    const gen = scanGenerationRef.current;
    videoPipelineRef.current.start(
      videoEl,
      {
        getParams: () => mlParamsRef.current,
        getConfig: () => videoStreamConfigRef.current,
        onStats: (stats) => {
          if (gen !== scanGenerationRef.current) return;
          setVideoStreamStats(stats);
          setVideoFramesScanned(stats.framesProcessed);
        },
        onResult: (result, stats, videoTimeSec) => {
          if (gen !== scanGenerationRef.current) return;
          setLastDetection(result.detection);
          setMlMetrics(result.metrics);
          setClusterProfiles(result.clusters);
          setOverlayDataUrl(result.overlayDataUrl);
          setLearnedBaseline(getLearnedRoadBaseline());
          setVideoStreamStats(stats);
          setVideoFramesScanned(stats.framesProcessed);

          // Continuously observe frame motion even on zero-detection frames so
          // active pothole tracks stay locked across blurred/missed video frames
          if (result.frameSignature) {
            if (
              sourceModeRef.current === 'upload' &&
              uploadedKindRef.current === 'video'
            ) {
              videoTrackerRef.current.observeFrame(
                videoTimeSec,
                result.frameSignature
              );
            } else if (sourceModeRef.current === 'camera') {
              const elapsedSec =
                (Date.now() - cameraSessionStartMsRef.current) / 1000;
              cameraTrackerRef.current.observeFrame(
                elapsedSec,
                result.frameSignature
              );
            }
          }

          if (result.detection.potholeDetected) {
            consecutiveHitsRef.current += 1;
            const confirmed =
              result.engine === 'roboflow' || consecutiveHitsRef.current >= 2;
            if (confirmed) {
              const snap =
                result.snapshotDataUrl || captureSnapshotDataUrl(videoEl);
              addCandidateToReviewList(
                result.detection,
                snap,
                false,
                videoTimeSec,
                result.frameSignature,
                result.cropSignatures
              );
            }
          } else {
            consecutiveHitsRef.current = 0;
          }
        },
        onStopped: () => {
          if (durationTimerRef.current) {
            window.clearInterval(durationTimerRef.current);
            durationTimerRef.current = null;
          }
          setIsRecording(false);
        },
      },
      resetStats
    );

    durationTimerRef.current = window.setInterval(() => {
      setRecordingSeconds((s) => s + 1);
    }, 1000);
  }

  function handleUploadedVideoPlay() {
    if (sourceModeRef.current !== 'upload' || uploadedKindRef.current !== 'video') {
      return;
    }
    const vid = uploadedVideoRef.current;
    if (!vid) return;

    setScanError(null);
    const startingFromBeginning = vid.currentTime < 0.4;
    if (startingFromBeginning) {
      setVideoFramesScanned(0);
      setVideoStreamStats(null);
    }
    if (!videoPipelineRef.current?.isRunning()) {
      setIsRecording(true);
      startVideoStreamPipeline(vid, startingFromBeginning);
    }
  }

  function handleUploadedVideoPauseOrEnd() {
    if (sourceModeRef.current !== 'upload' || uploadedKindRef.current !== 'video') {
      return;
    }
    stopContinuousScanLoop();
    setIsRecording(false);
  }

  async function startRecordingSession() {
    if (isRecording || cameraStatus === 'requesting') return;
    setScanError(null);
    stopCameraOnRecorderStopRef.current = false;

    if (sourceMode === 'upload' && uploadedKind === 'video') {
      const vid = uploadedVideoRef.current;
      if (!vid) {
        setScanError('Video player is not ready yet.');
        return;
      }
      const restartFromZero =
        vid.ended || (vid.duration && vid.currentTime >= vid.duration - 0.15);
      if (restartFromZero) {
        vid.currentTime = 0;
        setVideoFramesScanned(0);
        setVideoStreamStats(null);
      }
      setRecordingSeconds(0);
      setIsRecording(true);
      try {
        await vid.play();
      } catch {
        // Fallback: start muted if browser autoplay policy requires it
        vid.muted = true;
        await vid.play().catch(() => undefined);
      }
      if (!videoPipelineRef.current?.isRunning()) {
        startVideoStreamPipeline(vid, Boolean(restartFromZero) || vid.currentTime < 0.4);
      }
      return;
    }

    if (sourceMode === 'camera' && (cameraStatus !== 'ready' || !streamRef.current)) {
      const stream = await startCamera(facingMode);
      if (!stream) return;
    }

    consecutiveHitsRef.current = 0;
    sessionCapturedRef.current = [];
    cameraTrackerRef.current.reset();
    cameraSessionStartMsRef.current = Date.now();
    setVideoFramesScanned(0);
    setVideoStreamStats(null);
    setRecordedVideoUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setRecordingSeconds(0);
    setIsRecording(true);

    if (gpsStatus !== 'locked') {
      onRetryGps();
    }

    if (
      sourceMode === 'camera' &&
      streamRef.current &&
      typeof MediaRecorder !== 'undefined'
    ) {
      try {
        recordedChunksRef.current = [];
        const recorder = new MediaRecorder(streamRef.current);
        recorder.ondataavailable = (ev) => {
          if (ev.data && ev.data.size > 0) {
            recordedChunksRef.current.push(ev.data);
          }
        };
        recorder.onstop = () => {
          if (recordedChunksRef.current.length > 0) {
            const blob = new Blob(recordedChunksRef.current, {
              type: recorder.mimeType || 'video/webm',
            });
            setRecordedVideoUrl(URL.createObjectURL(blob));
          }
          if (stopCameraOnRecorderStopRef.current) {
            stopCameraOnRecorderStopRef.current = false;
            stopCameraStream(true);
          }
        };
        recorder.start(1000);
        mediaRecorderRef.current = recorder;
      } catch {
        // Fallback if MediaRecorder unsupported
      }
    }

    // Run frame-synchronized downsampled video stream pipeline while recording
    if (videoRef.current) {
      startVideoStreamPipeline(videoRef.current, true);
    }
  }

  function stopRecordingSession() {
    setIsRecording(false);
    stopContinuousScanLoop();
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== 'inactive'
    ) {
      if (sourceMode === 'camera') {
        stopCameraOnRecorderStopRef.current = true;
      }
      mediaRecorderRef.current.stop();
    } else if (sourceMode === 'camera') {
      stopCameraStream(true);
    }
    if (
      sourceMode === 'upload' &&
      uploadedKind === 'video' &&
      uploadedVideoRef.current
    ) {
      uploadedVideoRef.current.pause();
    }
  }

  // Zero-involvement automated government portal dispatch with 5m cluster grouping on filing
  async function submitCandidateCluster(cluster: DetectedPotholeCandidate[]) {
    if (cluster.length === 0) return;
    const candidateIds = new Set(cluster.map((c) => c.candidateId));
    const leadCandidate = cluster[0];

    setDetectedList((prev) =>
      prev.map((item) =>
        candidateIds.has(item.candidateId)
          ? { ...item, decision: 'raising', error: undefined }
          : item
      )
    );

    const payload = buildComplaintPayloadFromCluster(
      cluster,
      open311Ref.current,
      COMPLAINT_CLUSTER_RADIUS_METERS
    );

    const queueLocally = async () => {
      const queued = await queueComplaintForOfflineSync(
        payload,
        leadCandidate.candidateId
      );
      const offlineRecord = pendingQueueItemToComplaintRecord(queued);
      onComplaintCreated(offlineRecord);
      await Promise.all(
        cluster.map((c) =>
          updateDetectionCandidate(c.candidateId, {
            decision: 'queued_offline',
            queuedOfflineId: queued.queueId,
            error: undefined,
          })
        )
      );
      setDetectedList((prev) =>
        prev.map((item) =>
          candidateIds.has(item.candidateId)
            ? {
                ...item,
                decision: 'queued_offline',
                queuedOfflineId: queued.queueId,
                error: undefined,
              }
            : item
        )
      );
      onDetectionsChanged?.();
    };

    if (typeof navigator !== 'undefined' && !navigator.onLine) {
      try {
        await queueLocally();
      } catch (err: any) {
        setDetectedList((prev) =>
          prev.map((item) =>
            candidateIds.has(item.candidateId)
              ? {
                  ...item,
                  decision: 'pending',
                  error: err?.message || 'Unable to store offline complaint.',
                }
              : item
          )
        );
      }
      return;
    }

    try {
      const res = await fetch('/api/complaints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.status === 502 || res.status === 503 || res.status === 504) {
        await queueLocally();
        return;
      }

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Automated portal dispatch failed.');
      }

      const raised: ComplaintRecord = data.complaint;
      await upsertCachedComplaint(raised);
      await Promise.all(
        cluster.map((c) =>
          updateDetectionCandidate(c.candidateId, {
            decision: 'raised',
            raisedComplaint: raised,
            error: undefined,
          })
        )
      );
      onComplaintCreated(raised);

      setDetectedList((prev) =>
        prev.map((item) =>
          candidateIds.has(item.candidateId) ||
          item.raisedComplaint?.id === raised.id
            ? { ...item, decision: 'raised', raisedComplaint: raised, error: undefined }
            : item
        )
      );
      onDetectionsChanged?.();
    } catch (err: any) {
      // Network failure (offline / dropped mobile signal) -> queue seamlessly in IndexedDB
      const isNetworkFailure =
        err instanceof TypeError ||
        (typeof navigator !== 'undefined' && !navigator.onLine) ||
        /failed to fetch|networkerror|load failed/i.test(err?.message || '');

      if (isNetworkFailure) {
        try {
          await queueLocally();
          return;
        } catch {
          // Fall through to surface error if IndexedDB also fails
        }
      }

      setDetectedList((prev) =>
        prev.map((item) =>
          candidateIds.has(item.candidateId)
            ? {
                ...item,
                decision: 'pending',
                error: err?.message || 'Failed to raise complaint automatically.',
              }
            : item
        )
      );
    }
  }

  async function handleAcceptCandidate(candidate: DetectedPotholeCandidate) {
    await submitCandidateCluster([candidate]);
  }

  async function handleAcceptAllPending() {
    const pendingItems = detectedList.filter((d) => d.decision === 'pending');
    if (pendingItems.length === 0) return;
    setRaisingAll(true);
    try {
      const clusters = groupCandidatesByRadius(
        pendingItems,
        COMPLAINT_CLUSTER_RADIUS_METERS
      );
      for (const cluster of clusters) {
        await submitCandidateCluster(cluster);
      }
    } finally {
      setRaisingAll(false);
    }
  }

  function handleRejectCandidate(candidateId: string) {
    setDetectedList((prev) =>
      prev.map((item) =>
        item.candidateId === candidateId ? { ...item, decision: 'rejected' } : item
      )
    );
    void updateDetectionCandidate(candidateId, { decision: 'rejected' });
  }

  function handleRejectAllPending() {
    const pendingItems = detectedList.filter((d) => d.decision === 'pending');
    if (pendingItems.length === 0) return;
    const pendingIds = new Set(pendingItems.map((d) => d.candidateId));
    setDetectedList((prev) =>
      prev.map((item) =>
        pendingIds.has(item.candidateId)
          ? { ...item, decision: 'rejected', error: undefined }
          : item
      )
    );
    void Promise.all(
      pendingItems.map((item) =>
        updateDetectionCandidate(item.candidateId, { decision: 'rejected' })
      )
    ).then(() => {
      onDetectionsChanged?.();
    });
  }

  function handleClearReviewed() {
    setDetectedList((prev) =>
      prev.filter(
        (item) => item.decision !== 'raised' && item.decision !== 'rejected'
      )
    );
    void clearFinishedDetections().then(() => {
      onDetectionsChanged?.();
    });
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (isRecording) stopRecordingSession();
    if (uploadedFileUrl && uploadedFileUrl.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedFileUrl);
    }
    const isVideoFile =
      file.type.startsWith('video/') ||
      /\.(mp4|webm|mov|m4v|ogg|ogv|avi|mkv|3gp|3g2)$/i.test(file.name);
    const url = URL.createObjectURL(file);
    setUploadedFileUrl(url);
    setUploadedKind(isVideoFile ? 'video' : 'image');
    setUploadedVideoDims(null);
    setVideoFramesScanned(0);
    setVideoStreamStats(null);
    sessionCapturedRef.current = [];
    if (isVideoFile) {
      const fileKey = `${file.name}:${file.size}:${file.lastModified}`;
      let existingTracker = videoTrackersByFileRef.current.get(fileKey);
      if (!existingTracker) {
        existingTracker = new VideoPotholeTracker();
        videoTrackersByFileRef.current.set(fileKey, existingTracker);
      }
      videoTrackerRef.current = existingTracker;
    } else {
      videoTrackerRef.current = new VideoPotholeTracker();
    }
    clearScanVisuals();
    e.target.value = '';
  }

  const activeSample =
    ROAD_SAMPLES.find((s) => s.id === selectedSampleId) || ROAD_SAMPLES[0];

  const hasActiveMedia =
    (sourceMode === 'camera' && cameraStatus === 'ready') ||
    (sourceMode === 'upload' && Boolean(uploadedFileUrl)) ||
    sourceMode === 'samples';

  const activeDetections = hasActiveMedia ? lastDetection?.detections || [] : [];
  const canRecord =
    (sourceMode === 'camera' && cameraStatus !== 'requesting') ||
    (sourceMode === 'upload' && uploadedKind === 'video' && Boolean(uploadedFileUrl));

  const pendingCount = detectedList.filter((d) => d.decision === 'pending').length;
  const queuedOfflineCount = detectedList.filter(
    (d) => d.decision === 'queued_offline'
  ).length;
  const raisedCount = detectedList.filter((d) => d.decision === 'raised').length;
  const finishedCount = detectedList.filter(
    (d) => d.decision === 'raised' || d.decision === 'rejected'
  ).length;

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 py-5 sm:py-8 space-y-6 pb-24 md:pb-10">
      {/* Header & Source Switcher */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight text-balance">
            Road Pothole Scanner
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Start recording to detect potholes, depth, and GPS location automatically—then review the list below to <strong>Approve</strong> or <strong>Reject</strong> each complaint.
          </p>
        </div>

        <div className="inline-flex items-center gap-1 p-1 bg-slate-200/75 rounded-xl self-start md:self-auto">
          <button
            type="button"
            onClick={() => {
              if (isRecording) stopRecordingSession();
              clearScanVisuals();
              setSourceMode('camera');
            }}
            className={`min-h-[40px] px-3.5 py-1.5 text-xs sm:text-sm font-medium rounded-lg inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors ${
              sourceMode === 'camera'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Camera className="w-4 h-4" />
            Live Camera
          </button>
          <button
            type="button"
            onClick={() => {
              if (isRecording) stopRecordingSession();
              clearScanVisuals();
              setSourceMode('upload');
            }}
            className={`min-h-[40px] px-3.5 py-1.5 text-xs sm:text-sm font-medium rounded-lg inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors ${
              sourceMode === 'upload'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Upload className="w-4 h-4" />
            Upload Media
          </button>
          <button
            type="button"
            onClick={() => {
              if (isRecording) stopRecordingSession();
              clearScanVisuals();
              setSourceMode('samples');
            }}
            className={`min-h-[40px] px-3.5 py-1.5 text-xs sm:text-sm font-medium rounded-lg inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors ${
              sourceMode === 'samples'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            Road Samples
          </button>
        </div>
      </div>

      {/* Two-Column Scanner Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column: Viewfinder, Overlays & Recording Controls */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div
            ref={viewfinderRef}
            className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-200 aspect-[4/3] sm:aspect-[16/11] w-full flex items-center justify-center"
          >
            {sourceMode === 'camera' && (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover ${
                    cameraStatus === 'ready' ? 'opacity-100' : 'opacity-0'
                  }`}
                />
                {cameraStatus !== 'ready' && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-slate-900/95 text-white">
                    <Camera className="w-10 h-10 text-amber-400 mb-3" />
                    <h2 className="font-display text-lg font-semibold">
                      {cameraStatus === 'idle'
                        ? 'Camera Standby'
                        : cameraStatus === 'requesting'
                          ? 'Starting Camera...'
                          : 'Camera Permission Needed'}
                    </h2>
                    <p className="text-sm text-slate-300 mt-1 max-w-sm">
                      {cameraStatus === 'idle'
                        ? 'Camera stays off until you tap Start Record. Recording automatically activates live pothole detection and GPS logging.'
                        : cameraStatus === 'requesting'
                          ? 'Allow camera access when prompted to begin recording and road hazard scanning.'
                          : cameraError ||
                            'Allow camera access when prompted to start recording and detect road potholes.'}
                    </p>
                    <div className="mt-5 flex flex-wrap items-center justify-center gap-2.5">
                      {cameraStatus !== 'requesting' && (
                        <button
                          type="button"
                          onClick={() => void startRecordingSession()}
                          className="min-h-[44px] px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white text-sm font-semibold inline-flex items-center gap-2 cursor-pointer transition-colors"
                        >
                          <Video className="w-4 h-4" />
                          {cameraStatus === 'error'
                            ? 'Retry & Start Record'
                            : 'Start Record'}
                        </button>
                      )}
                      {cameraStatus === 'idle' && (
                        <button
                          type="button"
                          onClick={() =>
                            setFacingMode((f) =>
                              f === 'environment' ? 'user' : 'environment'
                            )
                          }
                          className="min-h-[44px] px-3.5 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white text-sm font-medium inline-flex items-center gap-1.5 cursor-pointer"
                        >
                          <RefreshCw className="w-3.5 h-3.5" />
                          {facingMode === 'environment' ? 'Rear Camera' : 'Front Camera'}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => {
                          clearScanVisuals();
                          setSourceMode('samples');
                        }}
                        className="min-h-[44px] px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white text-sm font-medium cursor-pointer"
                      >
                        Test Road Samples
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}

            {sourceMode === 'upload' && (
              <>
                {uploadedFileUrl ? (
                  uploadedKind === 'video' ? (
                    <video
                      ref={uploadedVideoRef}
                      src={uploadedFileUrl}
                      controls
                      playsInline
                      preload="auto"
                      onLoadedMetadata={(e) => {
                        const v = e.currentTarget;
                        if (v.videoWidth > 0 && v.videoHeight > 0) {
                          setUploadedVideoDims({
                            width: v.videoWidth,
                            height: v.videoHeight,
                          });
                        }
                      }}
                      onPlay={handleUploadedVideoPlay}
                      onPause={handleUploadedVideoPauseOrEnd}
                      onEnded={handleUploadedVideoPauseOrEnd}
                      onError={() =>
                        setScanError(
                          'Unable to decode this video format in the browser. Please upload an MP4 (H.264) or WebM video.'
                        )
                      }
                      className="w-full h-full object-contain bg-black"
                    />
                  ) : (
                    <img
                      ref={uploadedImgRef}
                      src={uploadedFileUrl}
                      alt="Uploaded road inspection photo"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                  )
                ) : (
                  <label className="w-full h-full flex flex-col items-center justify-center p-6 text-center cursor-pointer hover:bg-slate-900 transition-colors text-white">
                    <Upload className="w-10 h-10 text-amber-400 mb-3" />
                    <span className="font-display text-lg font-semibold">
                      Upload Road Photo or Dashcam Video
                    </span>
                    <span className="text-sm text-slate-300 mt-1 max-w-sm">
                      Select an image or MP4/WebM clip to detect potholes, depth, and dimensions.
                    </span>
                    <input
                      type="file"
                      accept="image/*,video/*"
                      onChange={handleFileUpload}
                      className="sr-only"
                    />
                  </label>
                )}
              </>
            )}

            {sourceMode === 'samples' && (
              <img
                ref={sampleImgRef}
                src={activeSample.url}
                alt={activeSample.label}
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover"
              />
            )}

            {/* Pixel-level Road Surface & Bounding Box Overlay (aligned to media rect for videos) */}
            <div
              className="absolute pointer-events-none"
              style={
                videoOverlayRect
                  ? {
                      left: `${videoOverlayRect.left}px`,
                      top: `${videoOverlayRect.top}px`,
                      width: `${videoOverlayRect.width}px`,
                      height: `${videoOverlayRect.height}px`,
                    }
                  : { inset: 0 }
              }
            >
              {hasActiveMedia && overlayDataUrl && mlParams.overlayMode !== 'detections' && (
                <img
                  src={overlayDataUrl}
                  alt="Road surface inspection overlay"
                  className="absolute inset-0 w-full h-full object-cover pointer-events-none"
                  style={{ imageRendering: 'pixelated' }}
                />
              )}

              {hasActiveMedia && activeDetections.length > 0 && (
                <svg
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  className="absolute inset-0 w-full h-full pointer-events-none"
                >
                  {activeDetections.map((det, idx) => {
                    const x = Math.max(0, det.box.xMin);
                    const y = Math.max(0, det.box.yMin);
                    const w = Math.max(25, det.box.xMax - det.box.xMin);
                    const h = Math.max(25, det.box.yMax - det.box.yMin);
                    const color = getSeverityMeta(det.severity).hex;
                    return (
                      <g key={idx}>
                        <rect
                          x={x}
                          y={y}
                          width={w}
                          height={h}
                          fill="none"
                          stroke={color}
                          strokeWidth="6"
                        />
                        <rect
                          x={x}
                          y={Math.max(0, y - 44)}
                          width={Math.min(580, Math.max(280, w))}
                          height="42"
                          fill={color}
                        />
                        <text
                          x={x + 12}
                          y={Math.max(29, y - 14)}
                          fill="#ffffff"
                          fontSize="22"
                          fontFamily="IBM Plex Mono, monospace"
                          fontWeight="600"
                        >
                          {det.severity} · {det.estimatedWidthCm}cm W · {det.estimatedDepthCm}cm D
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>

            {/* Top Viewfinder HUD */}
            <div className="absolute top-3 left-3 right-3 flex items-center justify-between gap-2 pointer-events-none">
              <div className="flex items-center gap-2 pointer-events-auto">
                {isRecording && (
                  <div className="px-3 py-1.5 rounded-lg bg-red-600 text-white text-xs font-mono font-semibold flex items-center gap-2 shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                    {sourceMode === 'upload' && uploadedKind === 'video'
                      ? `SCANNING · ${videoFramesScanned}f (${videoStreamStats?.framesSkipped ?? 0} skipped · ${videoStreamConfig.maxDimension}px)`
                      : `REC ${formatDuration(recordingSeconds)} · 1/${videoStreamConfig.frameSkipInterval}f · ${videoStreamConfig.maxDimension}px`}
                  </div>
                )}
                {mlMetrics && (
                  <div className="px-2.5 py-1.5 rounded-lg bg-slate-900/80 backdrop-blur-sm text-white text-xs font-mono">
                    {mlMetrics.inferenceMs}ms · K={mlMetrics.kClusters}
                  </div>
                )}
              </div>

              {sourceMode === 'camera' && cameraStatus === 'ready' && (
                <div className="flex items-center gap-2 pointer-events-auto">
                  {torchSupported && (
                    <button
                      type="button"
                      onClick={toggleTorch}
                      aria-label="Toggle flashlight"
                      className={`min-h-[40px] min-w-[40px] px-2.5 rounded-lg backdrop-blur-sm text-xs font-medium flex items-center justify-center gap-1.5 cursor-pointer ${
                        torchOn
                          ? 'bg-amber-500 text-slate-950'
                          : 'bg-slate-900/75 text-white hover:bg-slate-900'
                      }`}
                    >
                      <Flashlight className="w-4 h-4" />
                    </button>
                  )}
                  {!isRecording && (
                    <button
                      type="button"
                      onClick={() => {
                        const next =
                          facingMode === 'environment' ? 'user' : 'environment';
                        setFacingMode(next);
                        void startCamera(next);
                      }}
                      className="min-h-[40px] px-3 rounded-lg bg-slate-900/75 hover:bg-slate-900 backdrop-blur-sm text-white text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Flip
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Bottom GPS Scrim (hidden over uploaded video controls) */}
            {currentLocation && !(sourceMode === 'upload' && uploadedKind === 'video' && uploadedFileUrl) && (
              <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent px-4 py-3 text-white pointer-events-none">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span className="font-medium truncate max-w-[70%]">
                    {geocoded?.street || geocoded?.formattedAddress || 'Resolving street...'}
                  </span>
                  <span className="font-mono text-slate-200">
                    {currentLocation.lat.toFixed(5)}, {currentLocation.lng.toFixed(5)}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Visual Overlay Switcher & Scan Settings */}
          <div className="bg-white border border-slate-200 rounded-2xl p-3.5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                <Layers className="w-4 h-4 text-amber-600" />
                <span>Inspection Overlay</span>
              </div>

              <button
                type="button"
                onClick={() => setShowHyperparams((v) => !v)}
                className="text-xs font-medium text-slate-600 hover:text-slate-900 inline-flex items-center gap-1.5 cursor-pointer"
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                {showHyperparams ? 'Hide Settings' : 'Scan Settings'}
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 p-1 bg-slate-100 rounded-xl">
              {OVERLAY_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  onClick={() =>
                    setMlParams((prev) => ({ ...prev, overlayMode: mode.id }))
                  }
                  className={`min-h-[36px] px-2.5 py-1.5 text-xs font-medium rounded-lg whitespace-nowrap cursor-pointer transition-colors ${
                    mlParams.overlayMode === mode.id
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {mode.label}
                </button>
              ))}
            </div>

            {showHyperparams && (
              <div className="space-y-3 pt-2 border-t border-slate-100">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Lighting &amp; Surface Zones (GMM/K-Means)
                    </label>
                    <select
                      value={mlParams.kClusters}
                      onChange={(e) =>
                        setMlParams((p) => ({
                          ...p,
                          kClusters: Number(e.target.value),
                        }))
                      }
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white text-xs text-slate-900"
                    >
                      <option value={3}>3 Zones (Simple Road)</option>
                      <option value={4}>4 Zones (Standard)</option>
                      <option value={5}>5 Zones (Mixed Shadows)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Detection Sensitivity
                    </label>
                    <select
                      value={mlParams.sensitivity}
                      onChange={(e) =>
                        setMlParams((p) => ({
                          ...p,
                          sensitivity: e.target.value as SensitivityPreset,
                        }))
                      }
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white text-xs text-slate-900"
                    >
                      <option value="conservative">Strict (Major Hazards)</option>
                      <option value="balanced">Balanced (Recommended)</option>
                      <option value="sensitive">High (Includes Cracks)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Minimum Hazard Size (DBSCAN)
                    </label>
                    <select
                      value={mlParams.dbscanMinPts}
                      onChange={(e) =>
                        setMlParams((p) => ({
                          ...p,
                          dbscanMinPts: Number(e.target.value),
                        }))
                      }
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white text-xs text-slate-900"
                    >
                      <option value={3}>Small Hazards</option>
                      <option value={4}>Standard Size</option>
                      <option value={6}>Large Potholes Only</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-slate-100">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Video Frame-Skip Interval
                    </label>
                    <select
                      value={videoStreamConfig.frameSkipInterval}
                      onChange={(e) =>
                        setVideoStreamConfig((prev) => ({
                          ...prev,
                          frameSkipInterval: Number(e.target.value),
                        }))
                      }
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white text-xs text-slate-900"
                    >
                      <option value={3}>Every 3 frames (High coverage)</option>
                      <option value={5}>Every 5 frames (Balanced)</option>
                      <option value={10}>Every 10 frames (Fast)</option>
                      <option value={15}>Every 15 frames (Low bandwidth)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Video Downsample Resolution
                    </label>
                    <select
                      value={videoStreamConfig.maxDimension}
                      onChange={(e) =>
                        setVideoStreamConfig((prev) => ({
                          ...prev,
                          maxDimension: Number(e.target.value),
                        }))
                      }
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white text-xs text-slate-900"
                    >
                      <option value={480}>480px (Fastest inference)</option>
                      <option value={640}>640px (Balanced — Recommended)</option>
                      <option value={800}>800px (Higher detail)</option>
                    </select>
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
                  <div className="text-[11px] text-slate-600">
                    <span className="font-semibold text-slate-800">
                      Online Unsupervised Baseline:
                    </span>{' '}
                    <span className="font-mono">
                      {learnedBaseline.framesTrained} frames learned
                    </span>
                    {baselineNotice ? (
                      <span className="ml-2 text-emerald-700 font-medium">
                        {baselineNotice}
                      </span>
                    ) : null}
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        const mediaEl = getActiveMediaElement();
                        if (!mediaEl) {
                          setBaselineNotice('Open camera or load an image first.');
                          return;
                        }
                        const next = trainRoadBaselineFromSource(mediaEl);
                        if (next) {
                          setLearnedBaseline(next);
                          setBaselineNotice(
                            `Calibrated (${next.framesTrained}f)`
                          );
                        }
                      }}
                      className="h-8 px-3 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-[11px] font-semibold cursor-pointer"
                    >
                      Train on Normal Road
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const reset = resetLearnedRoadBaseline();
                        setLearnedBaseline(reset);
                        setBaselineNotice('Reset to default');
                      }}
                      className="h-8 px-2.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-[11px] font-medium cursor-pointer"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {sourceMode === 'samples' && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {ROAD_SAMPLES.map((sample) => {
                const selected = sample.id === selectedSampleId;
                return (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => {
                      setSelectedSampleId(sample.id);
                      clearScanVisuals();
                    }}
                    className={`text-left p-2 rounded-xl border transition-all cursor-pointer ${
                      selected
                        ? 'border-amber-600 bg-amber-50/50'
                        : 'border-slate-200 bg-white hover:border-slate-300'
                    }`}
                  >
                    <img
                      src={sample.url}
                      alt={sample.label}
                      referrerPolicy="no-referrer"
                      className="w-full aspect-[4/3] object-cover rounded-lg mb-2"
                    />
                    <div className="text-xs font-semibold text-slate-900 truncate">
                      {sample.label}
                    </div>
                    <div className="text-[11px] text-slate-500 truncate">
                      {sample.subtitle}
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {sourceMode === 'upload' && uploadedFileUrl && (
            <div className="flex items-center justify-between gap-3 px-4 py-2.5 bg-white border border-slate-200 rounded-xl">
              <span className="text-xs text-slate-600">
                Loaded {uploadedKind === 'video' ? 'video clip' : 'road image'} for inspection.
              </span>
              <label className="text-xs font-semibold text-amber-700 hover:text-amber-800 cursor-pointer">
                Choose another file
                <input
                  type="file"
                  accept="image/*,video/*"
                  onChange={handleFileUpload}
                  className="sr-only"
                />
              </label>
            </div>
          )}

          {/* Recording / Video Scan Control Bar */}
          <div className="bg-white border border-slate-200 rounded-2xl p-4 space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              {sourceMode === 'camera' ? (
                <>
                  <button
                    type="button"
                    onClick={() => void startRecordingSession()}
                    disabled={!canRecord || isRecording}
                    className="flex-1 min-w-[160px] min-h-[48px] px-5 py-3 rounded-xl font-semibold text-sm inline-flex items-center justify-center gap-2 transition-all active:scale-[0.99] bg-amber-600 hover:bg-amber-700 disabled:bg-slate-200 disabled:text-slate-500 disabled:cursor-not-allowed text-white cursor-pointer"
                  >
                    <Video className="w-4 h-4" />
                    {isRecording
                      ? 'Recording...'
                      : cameraStatus === 'requesting'
                        ? 'Starting Camera...'
                        : 'Start Record'}
                  </button>
                  <button
                    type="button"
                    onClick={stopRecordingSession}
                    disabled={!isRecording}
                    className="flex-1 min-w-[160px] min-h-[48px] px-5 py-3 rounded-xl font-semibold text-sm inline-flex items-center justify-center gap-2 transition-all active:scale-[0.99] bg-red-600 hover:bg-red-700 disabled:bg-slate-100 disabled:text-slate-400 disabled:border disabled:border-slate-200 disabled:cursor-not-allowed text-white cursor-pointer"
                  >
                    <Square className="w-4 h-4 fill-current" />
                    Stop Recording{isRecording ? ` (${detectedList.length})` : ''}
                  </button>
                </>
              ) : sourceMode === 'upload' && uploadedKind === 'video' ? (
                <>
                  <button
                    type="button"
                    onClick={() =>
                      isRecording
                        ? stopRecordingSession()
                        : void startRecordingSession()
                    }
                    disabled={!canRecord}
                    className={`flex-1 min-w-[160px] min-h-[48px] px-5 py-3 rounded-xl font-semibold text-sm inline-flex items-center justify-center gap-2 transition-all active:scale-[0.99] disabled:bg-slate-200 disabled:text-slate-500 disabled:cursor-not-allowed text-white cursor-pointer ${
                      isRecording
                        ? 'bg-red-600 hover:bg-red-700'
                        : 'bg-amber-600 hover:bg-amber-700'
                    }`}
                  >
                    {isRecording ? (
                      <>
                        <Pause className="w-4 h-4" />
                        Pause Video Scan ({videoFramesScanned}f)
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-current" />
                        Play &amp; Scan Video
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    disabled={isScanning || isRecording || !hasActiveMedia}
                    onClick={() => void runDetectionPass(true, false)}
                    className="flex-1 min-w-[160px] min-h-[48px] px-5 py-3 rounded-xl font-semibold text-sm inline-flex items-center justify-center gap-2 transition-all active:scale-[0.99] bg-slate-900 hover:bg-slate-800 disabled:bg-slate-100 disabled:text-slate-400 disabled:border disabled:border-slate-200 disabled:cursor-not-allowed text-white cursor-pointer"
                  >
                    <Cpu className="w-4 h-4" />
                    {isScanning ? 'Scanning Frame...' : 'Scan Current Frame'}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  disabled={isScanning || !hasActiveMedia}
                  onClick={() => void runDetectionPass(true, false)}
                  className="flex-1 min-h-[48px] px-5 py-3 rounded-xl bg-amber-600 hover:bg-amber-700 disabled:bg-slate-200 disabled:text-slate-500 disabled:cursor-not-allowed text-white text-sm font-semibold inline-flex items-center justify-center gap-2 cursor-pointer transition-colors"
                >
                  <Cpu className="w-4 h-4" />
                  {isScanning
                    ? 'Detecting Potholes...'
                    : 'Detect Potholes & Add to Review List'}
                </button>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 pt-1 border-t border-slate-100">
              <span>
                {sourceMode === 'upload' && uploadedKind === 'video'
                  ? isRecording
                    ? `Scanning video frames live (${videoFramesScanned} frames analyzed)...`
                    : 'Press Play on the video or tap Play & Scan Video to detect potholes across all frames, or pause and tap Scan Current Frame.'
                  : isRecording
                    ? 'Scanning road live for potholes, depth, and GPS while recording...'
                    : 'Camera is off. Tap Start Record when you are ready to scan.'}
              </span>

              {recordedVideoUrl && (
                <a
                  href={recordedVideoUrl}
                  download={`roadwatch-scan-${Date.now()}.webm`}
                  className="font-semibold text-amber-700 hover:text-amber-800 inline-flex items-center gap-1.5"
                >
                  <Download className="w-3.5 h-3.5" />
                  Save Video
                </a>
              )}
            </div>

            {scanError && (
              <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 font-medium">
                {scanError}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: GPS + Unsupervised ML Pipeline Telemetry */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Location & Matched Govt Authority */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-3.5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-amber-600 shrink-0" />
                <h2 className="font-display text-base font-semibold text-slate-900">
                  Automatic GPS &amp; Authority Match
                </h2>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setEditingCoords((v) => !v)}
                  className="text-xs font-medium text-slate-600 hover:text-slate-900 cursor-pointer"
                >
                  {editingCoords ? 'Close' : 'Edit GPS'}
                </button>
                <button
                  type="button"
                  onClick={onRetryGps}
                  className="text-xs font-semibold text-amber-700 hover:text-amber-800 cursor-pointer"
                >
                  Refresh
                </button>
              </div>
            </div>

            {editingCoords && (
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-2.5">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Latitude
                    </label>
                    <input
                      type="text"
                      value={manualLat}
                      onChange={(e) => setManualLat(e.target.value)}
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white font-mono text-xs"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Longitude
                    </label>
                    <input
                      type="text"
                      value={manualLng}
                      onChange={(e) => setManualLng(e.target.value)}
                      className="w-full h-9 px-2.5 rounded-lg border border-slate-300 bg-white font-mono text-xs"
                    />
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const lat = Number(manualLat);
                    const lng = Number(manualLng);
                    if (Number.isFinite(lat) && Number.isFinite(lng)) {
                      onSetManualLocation(lat, lng);
                      setEditingCoords(false);
                    }
                  }}
                  className="w-full h-9 rounded-lg bg-slate-900 text-white text-xs font-semibold cursor-pointer"
                >
                  Apply Coordinates
                </button>
              </div>
            )}

            <div className="space-y-2 text-sm">
              <div className="font-medium text-slate-900">
                {geocoded?.formattedAddress ||
                  (gpsStatus === 'acquiring'
                    ? 'Acquiring satellite GPS lock...'
                    : gpsError || 'Location unavailable — tap Refresh or Edit GPS')}
              </div>

              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600">
                <span className="font-mono">
                  {currentLocation
                    ? `${currentLocation.lat.toFixed(6)}, ${currentLocation.lng.toFixed(6)}`
                    : 'No coordinates yet'}
                </span>
                {currentLocation?.accuracy ? (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="font-mono">
                      ±{Math.round(currentLocation.accuracy)}m accuracy
                    </span>
                  </>
                ) : null}
              </div>

              {geocoded?.portal && (
                <div className="pt-2 border-t border-slate-100 flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-slate-900 truncate">
                      {geocoded.portal.portalName}
                    </div>
                    <div className="text-[11px] text-slate-500 truncate">
                      {geocoded.portal.department}
                    </div>
                  </div>
                  <a
                    href={geocoded.portal.portalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-semibold text-amber-700 hover:text-amber-800 inline-flex items-center gap-1 shrink-0"
                  >
                    Portal
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              )}
            </div>
          </div>

          {/* Live Road Inspection Summary */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 flex-1 flex flex-col justify-between space-y-4">
            {mlMetrics ? (
              <div className="space-y-3.5">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="font-display text-base font-semibold text-slate-900">
                    Live Scan Summary
                  </h2>
                  <span className="font-mono text-xs text-slate-500">
                    {mlMetrics.roadPatches} areas · {mlMetrics.inferenceMs}ms
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2.5">
                  <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-[11px] text-slate-500">Hazards Found</div>
                    <div className="font-mono text-sm font-semibold text-slate-900">
                      {mlMetrics.dbscanClustersFound}
                    </div>
                    <div className="font-mono text-[10px] text-slate-500">
                      {activeDetections.length > 0 ? 'Active frame' : 'Road clear'}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-[11px] text-slate-500">Road Baseline</div>
                    <div className="font-mono text-sm font-semibold text-slate-900">
                      {Math.round(mlMetrics.baselineLuminanceMean * 100)}% tone
                    </div>
                    <div className="font-mono text-[10px] text-slate-500">
                      {mlMetrics.trainedFrames > 0
                        ? `Trained ${mlMetrics.trainedFrames}f`
                        : 'Auto-calibrated'}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-[11px] text-slate-500">ML Silhouette</div>
                    <div className="font-mono text-sm font-semibold text-slate-900">
                      {mlMetrics.silhouetteScore.toFixed(2)}
                    </div>
                    <div className="font-mono text-[10px] text-slate-500">
                      {mlMetrics.inferenceMs} ms · iForest
                    </div>
                  </div>
                </div>

                <div className="space-y-1.5">
                  {clusterProfiles.map((c) => (
                    <div key={c.clusterId} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-700 font-medium truncate">
                          {c.role}
                        </span>
                        <span className="font-mono text-slate-500 shrink-0">
                          {c.percentage.toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-200"
                          style={{
                            width: `${Math.max(2, c.percentage)}%`,
                            backgroundColor: c.colorHex,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="py-8 text-center space-y-2 my-auto">
                <h2 className="font-display text-base font-semibold text-slate-900">
                  Road Scanner Ready
                </h2>
                <p className="text-xs text-slate-600 max-w-xs mx-auto leading-relaxed">
                  While recording, RoadWatch automatically maps the road surface, detects potholes, estimates depth and size, and captures your GPS location.
                </p>
              </div>
            )}

            {/* Live Recording Status Summary */}
            <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
              <span className="text-slate-600">
                Review List:{' '}
                <strong className="font-mono text-slate-900">
                  {pendingCount} pending
                </strong>{' '}
                {queuedOfflineCount > 0 && (
                  <>
                    ·{' '}
                    <strong className="font-mono text-amber-700">
                      {queuedOfflineCount} offline queue
                    </strong>{' '}
                  </>
                )}
                ·{' '}
                <strong className="font-mono text-emerald-700">
                  {raisedCount} approved
                </strong>
              </span>
              {pendingCount > 0 && (
                <div className="flex items-center gap-3 shrink-0">
                  <button
                    type="button"
                    disabled={raisingAll}
                    onClick={handleRejectAllPending}
                    className="font-semibold text-red-600 hover:text-red-700 disabled:opacity-50 cursor-pointer"
                  >
                    Reject All
                  </button>
                  <button
                    type="button"
                    disabled={raisingAll}
                    onClick={handleAcceptAllPending}
                    className="font-semibold text-amber-700 hover:text-amber-800 disabled:opacity-50 cursor-pointer"
                  >
                    {raisingAll ? 'Submitting...' : 'Approve All'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Detected Potholes Detailed Review List (User Approves or Rejects before any complaint is filed) */}
      <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-bold text-slate-900">
              Detected Potholes &amp; Depth Report List ({detectedList.length})
            </h2>
            <p className="text-xs text-slate-600 mt-0.5">
              Complaints are never submitted automatically. Review each pothole&apos;s depth, size, and GPS location, then tap <strong>Approve</strong> to file or <strong>Reject</strong> to discard.
            </p>
          </div>

          {(pendingCount > 0 || finishedCount > 0) && (
            <div className="flex flex-wrap items-center gap-2 shrink-0">
              {finishedCount > 0 && (
                <button
                  type="button"
                  disabled={raisingAll}
                  onClick={handleClearReviewed}
                  className="min-h-[44px] px-3.5 py-2 rounded-xl border border-slate-200 bg-slate-50 hover:bg-slate-100 disabled:opacity-60 text-slate-700 text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-1.5 shrink-0 cursor-pointer transition-colors"
                >
                  Clear Reviewed ({finishedCount})
                </button>
              )}
              {pendingCount > 0 && (
                <>
                  <button
                    type="button"
                    disabled={raisingAll}
                    onClick={handleRejectAllPending}
                    className="min-h-[44px] px-4 py-2 rounded-xl border border-red-200 bg-red-50 hover:bg-red-100 disabled:opacity-60 text-red-700 text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-2 shrink-0 cursor-pointer transition-colors"
                  >
                    <X className="w-4 h-4" />
                    Reject All ({pendingCount})
                  </button>
                  <button
                    type="button"
                    disabled={raisingAll}
                    onClick={handleAcceptAllPending}
                    className="min-h-[44px] px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-700 disabled:opacity-60 text-white text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-2 shrink-0 cursor-pointer transition-colors"
                  >
                    <CheckCheck className="w-4 h-4" />
                    {raisingAll
                      ? 'Submitting Approved...'
                      : `Approve All (${pendingCount})`}
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {detectedList.length === 0 ? (
          <div className="p-8 sm:p-10 text-center space-y-2">
            <p className="text-sm font-medium text-slate-800">
              No potholes logged in this recording session yet
            </p>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              Tap <strong>Start Recording</strong> on your camera (or test a Road Sample) to detect potholes and inspect full depth, dimension, and GPS details here.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-slate-200">
            {detectedList.map((item) => {
              const sev = getSeverityMeta(item.primaryItem.severity);
              const d = item.primaryItem;
              return (
                <div
                  key={item.candidateId}
                  className={`p-4 sm:p-5 transition-colors ${
                    item.decision === 'rejected' ? 'opacity-50 bg-slate-50' : 'bg-white'
                  }`}
                >
                  <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                    {/* Thumbnail + Primary Details */}
                    <div className="flex flex-col sm:flex-row sm:items-start gap-4 flex-1 min-w-0">
                      <div className="relative w-full sm:w-36 aspect-[4/3] rounded-xl overflow-hidden bg-slate-900 border border-slate-200 shrink-0">
                        {item.snapshotDataUrl && (
                          <>
                            <img
                              src={item.snapshotDataUrl}
                              alt={item.detection.formalComplaintTitle}
                              referrerPolicy="no-referrer"
                              className="w-full h-full object-cover"
                            />
                            <svg
                              viewBox="0 0 1000 1000"
                              preserveAspectRatio="none"
                              className="absolute inset-0 w-full h-full pointer-events-none"
                            >
                              <rect
                                x={d.box.xMin}
                                y={d.box.yMin}
                                width={Math.max(25, d.box.xMax - d.box.xMin)}
                                height={Math.max(25, d.box.yMax - d.box.yMin)}
                                fill="none"
                                stroke={sev.hex}
                                strokeWidth="14"
                              />
                            </svg>
                          </>
                        )}
                      </div>

                      <div className="flex-1 min-w-0 space-y-2">
                        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                          <span className={`font-semibold ${sev.textClass}`}>
                            {sev.label}
                          </span>
                          <span aria-hidden="true">·</span>
                          <span className="font-mono text-slate-700">
                            Depth: {d.estimatedDepthCm} cm ({d.depthClass})
                          </span>
                          <span aria-hidden="true">·</span>
                          <span className="font-mono text-slate-700">
                            {new Date(item.detectedAt).toLocaleTimeString()}
                          </span>
                        </div>

                        <h3 className="font-display text-base font-bold text-slate-900">
                          {item.detection.formalComplaintTitle}
                        </h3>

                        {/* Structured Measurement Grid */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                          <div className="px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200">
                            <div className="text-[10px] text-slate-500">Est. Depth</div>
                            <div className="font-mono text-xs font-semibold text-slate-900">
                              {d.estimatedDepthCm} cm ({d.depthConfidenceRange})
                            </div>
                          </div>
                          <div className="px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200">
                            <div className="text-[10px] text-slate-500">Width &amp; Area</div>
                            <div className="font-mono text-xs font-semibold text-slate-900">
                              {d.estimatedWidthCm} cm · {d.estimatedAreaSqCm} cm²
                            </div>
                          </div>
                          <div className="px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200">
                            <div className="text-[10px] text-slate-500">Surface Roughness</div>
                            <div className="font-mono text-xs font-semibold text-slate-900">
                              {d.roughnessScorePct}% rough
                            </div>
                          </div>
                          <div className="px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200">
                            <div className="text-[10px] text-slate-500">Interior Shadow</div>
                            <div className="font-mono text-xs font-semibold text-slate-900">
                              {d.depthEstimationDetails
                                ? `${d.depthEstimationDetails.shadowOcclusionPct}% shadow`
                                : `${d.luminanceDropPct}% darker`}
                            </div>
                          </div>
                        </div>

                        <div className="text-xs text-slate-600 space-y-0.5 pt-0.5">
                          <div className="truncate">
                            <span className="font-medium text-slate-800">Location:</span>{' '}
                            {item.address}{' '}
                            <span className="font-mono text-slate-500">
                              ({item.lat.toFixed(5)}, {item.lng.toFixed(5)})
                            </span>
                          </div>
                          <div className="truncate">
                            <span className="font-medium text-slate-800">Govt Portal:</span>{' '}
                            {item.portalName} · {item.department}
                          </div>
                        </div>

                        {item.error && (
                          <div className="text-xs text-red-600 font-medium">
                            {item.error}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Approve / Reject Action Column */}
                    <div className="flex flex-row lg:flex-col items-center lg:items-end justify-end gap-2 shrink-0">
                      {item.decision === 'pending' && (
                        <>
                          <button
                            type="button"
                            onClick={() => handleAcceptCandidate(item)}
                            className="min-h-[44px] px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs sm:text-sm font-semibold inline-flex items-center gap-1.5 cursor-pointer transition-colors"
                          >
                            <Check className="w-4 h-4" />
                            Approve
                          </button>
                          <button
                            type="button"
                            onClick={() => handleRejectCandidate(item.candidateId)}
                            className="min-h-[44px] px-4 py-2 rounded-xl border border-red-200 bg-red-50 hover:bg-red-100 text-red-700 text-xs sm:text-sm font-semibold inline-flex items-center gap-1.5 cursor-pointer transition-colors"
                          >
                            <X className="w-4 h-4" />
                            Reject
                          </button>
                        </>
                      )}

                      {item.decision === 'raising' && (
                        <div className="px-4 py-2.5 rounded-xl bg-amber-50 border border-amber-200 text-xs font-semibold text-amber-800">
                          Submitting to government portal...
                        </div>
                      )}

                      {item.decision === 'queued_offline' && (
                        <div className="flex flex-col items-start lg:items-end gap-1">
                          <div className="px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-xs font-semibold text-amber-900">
                            Approved · Queued Offline
                          </div>
                          <span className="text-[11px] text-slate-500">
                            Will submit automatically when internet returns
                          </span>
                        </div>
                      )}

                      {item.decision === 'raised' && item.raisedComplaint && (
                        <div className="flex flex-col items-start lg:items-end gap-1.5">
                          <div className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs font-semibold text-emerald-800 inline-flex items-center gap-1.5">
                            <Check className="w-3.5 h-3.5" />
                            {item.raisedComplaint.isClusterReport &&
                            (item.raisedComplaint.clusterCount || 0) > 1
                              ? `Clustered (${item.raisedComplaint.clusterCount} in ${
                                  item.raisedComplaint.clusterRadiusMeters || 5
                                }m) · `
                              : 'Approved & Filed · '}
                            <span className="font-mono">
                              {item.raisedComplaint.externalTicketId ||
                                item.raisedComplaint.trackingCode}
                            </span>
                          </div>
                          <button
                            type="button"
                            onClick={() => onOpenComplaint(item.raisedComplaint!)}
                            className="text-xs font-semibold text-slate-700 hover:text-amber-700 cursor-pointer"
                          >
                            View Receipt &amp; Dossier
                          </button>
                        </div>
                      )}

                      {item.decision === 'rejected' && (
                        <span className="text-xs font-medium text-red-600">
                          Rejected by user
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
