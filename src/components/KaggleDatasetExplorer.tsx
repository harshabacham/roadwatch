import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  Download,
  ExternalLink,
  Eye,
  EyeOff,
  RefreshCw,
  Search,
} from 'lucide-react';
import type {
  DetectionResult,
  KaggleDatasetResponse,
  KagglePotholeRecord,
} from '../types';
import { detectPotholesWithRoboflow } from '../ml/roboflowDetector';
import { getSeverityMeta } from '../utils/geo';

interface KaggleDatasetExplorerProps {
  onOpenInScanner: (imageUrl: string, label: string) => void;
}

export function KaggleDatasetExplorer({
  onOpenInScanner,
}: KaggleDatasetExplorerProps) {
  const [data, setData] = useState<KaggleDatasetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [minBoxes, setMinBoxes] = useState(0);

  const [selected, setSelected] = useState<KagglePotholeRecord | null>(null);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [runningDetect, setRunningDetect] = useState(false);
  const [roboflowResult, setRoboflowResult] = useState<DetectionResult | null>(
    null
  );
  const [roboflowMs, setRoboflowMs] = useState<number | null>(null);
  const [detectError, setDetectError] = useState<string | null>(null);

  const previewImgRef = useRef<HTMLImageElement | null>(null);
  const limit = 18;

  const fetchDatasetPage = useCallback(
    async (nextPage: number, query: string, minCount: number) => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({
          page: String(nextPage),
          limit: String(limit),
        });
        if (query.trim()) params.set('search', query.trim());
        if (minCount > 0) params.set('minBoxes', String(minCount));

        const res = await fetch(`/api/datasets/kaggle?${params.toString()}`);
        const json = await res.json();
        if (!res.ok) {
          throw new Error(json.error || 'Failed to load Kaggle dataset.');
        }
        setData(json as KaggleDatasetResponse);
        setSelected((prev) => {
          if (
            prev &&
            json.records.some((r: KagglePotholeRecord) => r.id === prev.id)
          ) {
            return prev;
          }
          return json.records[0] || null;
        });
      } catch (err: any) {
        setError(err?.message || 'Unable to load Kaggle dataset.');
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void fetchDatasetPage(page, search, minBoxes);
    }, 150);
    return () => window.clearTimeout(timer);
  }, [page, search, minBoxes, fetchDatasetPage]);

  async function handleResync() {
    setSyncing(true);
    setError(null);
    try {
      const res = await fetch('/api/datasets/kaggle/import', {
        method: 'POST',
      });
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error || 'Failed to re-sync Kaggle dataset.');
      }
      await fetchDatasetPage(1, search, minBoxes);
      setPage(1);
    } catch (err: any) {
      setError(err?.message || 'Dataset sync failed.');
    } finally {
      setSyncing(false);
    }
  }

  async function handleRunRoboflowOnSelected() {
    if (!previewImgRef.current || previewImgRef.current.naturalWidth === 0) {
      setDetectError('Wait for the Kaggle image to finish loading first.');
      return;
    }
    setRunningDetect(true);
    setDetectError(null);
    try {
      const output = await detectPotholesWithRoboflow(previewImgRef.current, {
        kClusters: 4,
        sensitivity: 'balanced',
        dbscanMinPts: 4,
        overlayMode: 'detections',
      });
      if (!output) {
        throw new Error('Could not read image pixels for detection.');
      }
      setRoboflowResult(output.detection);
      setRoboflowMs(output.metrics.inferenceMs);
    } catch (err: any) {
      setDetectError(err?.message || 'Roboflow detection failed.');
    } finally {
      setRunningDetect(false);
    }
  }

  const totalPages = data
    ? Math.max(1, Math.ceil(data.totalFiltered / data.limit))
    : 1;

  const selectedImgUrl = selected
    ? `/api/datasets/kaggle/image/${encodeURIComponent(selected.filename)}`
    : '';

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 py-5 sm:py-8 space-y-6 pb-24 md:pb-10">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-amber-700 font-semibold uppercase tracking-wider">
            <Database className="w-3.5 h-3.5" />
            <span>Kaggle Imported Dataset</span>
          </div>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mt-1">
            andrewmvd/pothole-detection
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            665 annotated roadway images with Pascal VOC ground-truth bounding boxes. Inspect labels, test Roboflow detection side-by-side, or send any frame to the live scanner.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href="/api/datasets/kaggle/export.csv"
            download="andrewmvd-pothole-detection.csv"
            className="min-h-[40px] px-3.5 py-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-800 text-xs sm:text-sm font-semibold inline-flex items-center gap-1.5 whitespace-nowrap transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </a>
          <a
            href={
              data?.sourceUrl ||
              'https://www.kaggle.com/datasets/andrewmvd/pothole-detection'
            }
            target="_blank"
            rel="noopener noreferrer"
            className="min-h-[40px] px-3.5 py-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-xs sm:text-sm font-medium inline-flex items-center gap-1.5 whitespace-nowrap transition-colors"
          >
            Kaggle Source
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
          <button
            type="button"
            disabled={syncing}
            onClick={handleResync}
            className="min-h-[40px] px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 disabled:opacity-60 text-white text-xs sm:text-sm font-semibold inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${syncing ? 'animate-spin' : ''}`}
            />
            {syncing ? 'Syncing...' : 'Re-sync'}
          </button>
        </div>
      </div>

      {/* Dataset Summary Strip */}
      {data && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="bg-white border border-slate-200 rounded-2xl p-4">
            <div className="text-xs text-slate-500">Total Road Images</div>
            <div className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
              {data.totalImages.toLocaleString()}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              {data.format}
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl p-4">
            <div className="text-xs text-slate-500">Ground-Truth Potholes</div>
            <div className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
              {data.totalBoxes.toLocaleString()}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              Pascal VOC XML boxes
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl p-4">
            <div className="text-xs text-slate-500">Avg. Potholes / Image</div>
            <div className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
              {data.avgBoxesPerImage.toFixed(2)}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              Multi-hazard scenes included
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl p-4">
            <div className="text-xs text-slate-500">Locally Cached Frames</div>
            <div className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
              {data.cachedImageCount} / {data.totalImages}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              Remaining stream on demand
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-2xl bg-red-50 border border-red-200 text-sm text-red-700 font-medium">
          {error}
        </div>
      )}

      {/* Selected Kaggle Frame Inspector */}
      {selected && (
        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
          <div className="grid grid-cols-1 lg:grid-cols-12">
            {/* Image + SVG Boxes */}
            <div className="lg:col-span-7 bg-slate-950 relative aspect-[4/3] flex items-center justify-center overflow-hidden">
              <img
                ref={previewImgRef}
                key={selected.filename}
                src={selectedImgUrl}
                alt={`Kaggle dataset frame ${selected.filename}`}
                referrerPolicy="no-referrer"
                className="w-full h-full object-contain"
              />

              {/* Ground Truth VOC Boxes (Emerald Dashed) */}
              {showGroundTruth &&
                selected.width > 0 &&
                selected.height > 0 && (
                  <svg
                    viewBox={`0 0 ${selected.width} ${selected.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  >
                    {selected.boxes.map((b, i) => (
                      <g key={`gt-${i}`}>
                        <rect
                          x={b.xmin}
                          y={b.ymin}
                          width={Math.max(4, b.xmax - b.xmin)}
                          height={Math.max(4, b.ymax - b.ymin)}
                          fill="rgba(16, 185, 129, 0.12)"
                          stroke="#10b981"
                          strokeWidth={Math.max(2, selected.width * 0.005)}
                          strokeDasharray={`${Math.max(
                            4,
                            selected.width * 0.012
                          )} ${Math.max(3, selected.width * 0.008)}`}
                        />
                      </g>
                    ))}
                  </svg>
                )}

              {/* Roboflow Predicted Boxes (Amber/Red Solid) */}
              {roboflowResult && roboflowResult.detections.length > 0 && (
                <svg
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  className="absolute inset-0 w-full h-full pointer-events-none"
                >
                  {roboflowResult.detections.map((det, idx) => {
                    const color = getSeverityMeta(det.severity).hex;
                    return (
                      <g key={`rf-${idx}`}>
                        <rect
                          x={det.box.xMin}
                          y={det.box.yMin}
                          width={Math.max(20, det.box.xMax - det.box.xMin)}
                          height={Math.max(20, det.box.yMax - det.box.yMin)}
                          fill="none"
                          stroke={color}
                          strokeWidth="6"
                        />
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>

            {/* Metadata & Actions */}
            <div className="lg:col-span-5 p-5 flex flex-col justify-between gap-4 border-t lg:border-t-0 lg:border-l border-slate-200">
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="font-mono text-xs text-slate-500">
                      {selected.annotationFile}
                    </div>
                    <h2 className="font-display text-xl font-bold text-slate-900">
                      {selected.filename}
                    </h2>
                  </div>
                  <span className="font-mono text-xs text-slate-600 tabular-nums shrink-0">
                    {selected.width}×{selected.height}px
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2.5 text-xs">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-slate-500">VOC Ground Truth</div>
                    <div className="font-mono text-base font-bold text-emerald-700 mt-0.5 tabular-nums">
                      {selected.boxCount} pothole
                      {selected.boxCount === 1 ? '' : 's'}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-slate-500">Roboflow Prediction</div>
                    <div className="font-mono text-base font-bold text-slate-900 mt-0.5 tabular-nums">
                      {roboflowResult
                        ? `${roboflowResult.detections.length} found`
                        : 'Not run yet'}
                    </div>
                    {roboflowMs !== null && (
                      <div className="font-mono text-[10px] text-slate-500">
                        {roboflowMs} ms
                      </div>
                    )}
                  </div>
                </div>

                {/* Legend */}
                <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-3 h-3 border-2 border-dashed border-emerald-500 inline-block" />
                    Ground Truth XML
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-3 h-3 border-2 border-red-600 inline-block" />
                    Roboflow Detection
                  </span>
                </div>

                {/* Ground Truth Box Table */}
                <div className="max-h-36 overflow-y-auto border border-slate-200 rounded-xl divide-y divide-slate-100 text-xs font-mono">
                  {selected.boxes.map((b, idx) => (
                    <div
                      key={idx}
                      className="px-3 py-1.5 flex items-center justify-between text-slate-700"
                    >
                      <span>
                        #{idx + 1} {b.label}
                      </span>
                      <span className="text-slate-500 tabular-nums">
                        [{b.xmin}, {b.ymin}] → [{b.xmax}, {b.ymax}]
                      </span>
                    </div>
                  ))}
                </div>

                {detectError && (
                  <div className="text-xs text-red-600 font-medium">
                    {detectError}
                  </div>
                )}
              </div>

              <div className="space-y-2 pt-2 border-t border-slate-100">
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={runningDetect}
                    onClick={() => void handleRunRoboflowOnSelected()}
                    className="flex-1 min-h-[42px] px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-700 disabled:opacity-60 text-white text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-1.5 cursor-pointer transition-colors"
                  >
                    <Cpu className="w-4 h-4" />
                    {runningDetect ? 'Running API...' : 'Run Roboflow Detection'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowGroundTruth((v) => !v)}
                    className="min-h-[42px] px-3 py-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer"
                  >
                    {showGroundTruth ? (
                      <>
                        <EyeOff className="w-3.5 h-3.5" />
                        Hide GT
                      </>
                    ) : (
                      <>
                        <Eye className="w-3.5 h-3.5" />
                        Show GT
                      </>
                    )}
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    onOpenInScanner(
                      selectedImgUrl,
                      `Kaggle ${selected.filename}`
                    )
                  }
                  className="w-full min-h-[40px] px-4 py-2 rounded-xl border border-slate-300 bg-slate-50 hover:bg-slate-100 text-slate-900 text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-1.5 cursor-pointer transition-colors"
                >
                  <Camera className="w-4 h-4" />
                  Open Frame in Live Scanner
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search, Filter & Paginated Grid */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              placeholder="Search image ID (e.g. potholes42)..."
              className="w-full h-10 pl-9 pr-3 rounded-xl border border-slate-300 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-600"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <select
              value={minBoxes}
              onChange={(e) => {
                setMinBoxes(Number(e.target.value));
                setPage(1);
              }}
              className="h-10 px-3 rounded-xl border border-slate-300 bg-white text-xs sm:text-sm text-slate-800"
            >
              <option value={0}>All images (any pothole count)</option>
              <option value={2}>2+ potholes per image</option>
              <option value={5}>5+ potholes (severe clusters)</option>
              <option value={10}>10+ potholes (heavy damage)</option>
            </select>

            <div className="flex items-center gap-1.5 text-xs font-mono text-slate-600">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="h-9 w-9 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-40 flex items-center justify-center cursor-pointer"
                aria-label="Previous page"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-2 tabular-nums">
                {page} / {totalPages}
              </span>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="h-9 w-9 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-40 flex items-center justify-center cursor-pointer"
                aria-label="Next page"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {loading && !data ? (
          <div className="py-12 text-center text-sm text-slate-500">
            Loading imported Kaggle records...
          </div>
        ) : data && data.records.length === 0 ? (
          <div className="py-12 text-center text-sm text-slate-500">
            No Kaggle records match your filter.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {data?.records.map((rec) => {
              const isSelected = selected?.id === rec.id;
              return (
                <button
                  key={rec.id}
                  type="button"
                  onClick={() => {
                    setSelected(rec);
                    setRoboflowResult(null);
                    setRoboflowMs(null);
                    setDetectError(null);
                  }}
                  className={`text-left rounded-xl border p-2 transition-all cursor-pointer ${
                    isSelected
                      ? 'border-amber-600 bg-amber-50/60'
                      : 'border-slate-200 bg-white hover:border-slate-300'
                  }`}
                >
                  <div className="aspect-[4/3] rounded-lg overflow-hidden bg-slate-100 mb-2">
                    <img
                      src={`/api/datasets/kaggle/image/${encodeURIComponent(
                        rec.filename
                      )}`}
                      alt={rec.filename}
                      loading="lazy"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="font-mono text-xs font-semibold text-slate-900 truncate">
                    {rec.id}
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500 mt-0.5 font-mono tabular-nums">
                    <span>
                      {rec.boxCount} box{rec.boxCount === 1 ? '' : 'es'}
                    </span>
                    <span>
                      {rec.width}×{rec.height}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
