import { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cpu,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
} from 'lucide-react';
import type {
  DatasetCategory,
  DatasetLabel,
  LabeledDatasetSample,
  TrainedPotholeModel,
} from '../types';

interface MLTrainingPanelProps {
  model: TrainedPotholeModel | null;
  onModelUpdated: (model: TrainedPotholeModel) => void;
}

const FEATURE_LABELS: Record<string, string> = {
  meanZ: 'Ensemble Mahalanobis Z-Score',
  logRimRoughRatio: 'Log Rim Roughness Ratio',
  logRimCornerRatio: 'Log Rim Corner Energy Ratio',
  interiorLumDrop: 'Interior Luminance Drop',
  brightPaintExcess: 'Bright Paint / Glare Penalty',
  isotropy: 'Texture Isotropy (1 - Coherence)',
  entropyRatio: 'Shannon Texture Entropy Ratio',
  meanIsolation: 'Isolation Forest Score',
  fillDensity: 'Cluster Fill Density',
  cavityInteraction: 'Dark Rough Cavity Interaction',
};

const CATEGORY_OPTIONS: Array<{ value: DatasetCategory; label: string }> = [
  { value: 'pothole_cavity', label: 'Pothole Cavity' },
  { value: 'waterlogged_pothole', label: 'Waterlogged Pothole' },
  { value: 'alligator_cracking', label: 'Alligator Cracking' },
  { value: 'clean_road', label: 'Clean Intact Asphalt' },
  { value: 'lane_marking', label: 'Lane Marking / Crosswalk' },
  { value: 'shadow', label: 'Tree / Vehicle Shadow' },
  { value: 'tar_joint', label: 'Tar Seam / Curb Joint' },
  { value: 'wet_glare', label: 'Wet Pavement Glare' },
];

function resizeFileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read image file.'));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('Invalid image format.'));
      img.onload = () => {
        const maxDim = 480;
        const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
        const w = Math.max(64, Math.round(img.naturalWidth * scale));
        const h = Math.max(48, Math.round(img.naturalHeight * scale));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Canvas context unavailable.'));
          return;
        }
        ctx.drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL('image/jpeg', 0.86));
      };
      img.src = String(reader.result);
    };
    reader.readAsDataURL(file);
  });
}

export function MLTrainingPanel({ model, onModelUpdated }: MLTrainingPanelProps) {
  const [samples, setSamples] = useState<LabeledDatasetSample[]>([]);
  const [loadingDataset, setLoadingDataset] = useState(true);
  const [training, setTraining] = useState(false);
  const [penalty, setPenalty] = useState<number>(model?.falsePositivePenalty ?? 2.5);
  const [notice, setNotice] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Add labeled sample form
  const [newTitle, setNewTitle] = useState('');
  const [newLabel, setNewLabel] = useState<DatasetLabel>('pothole');
  const [newCategory, setNewCategory] = useState<DatasetCategory>('pothole_cavity');
  const [newImageDataUrl, setNewImageDataUrl] = useState('');
  const [submittingSample, setSubmittingSample] = useState(false);

  async function fetchDataset() {
    setLoadingDataset(true);
    try {
      const res = await fetch('/api/ml/dataset');
      if (res.ok) {
        const data = await res.json();
        setSamples(Array.isArray(data.samples) ? data.samples : []);
      }
    } catch {
      setErrorMsg('Unable to load training dataset from server.');
    } finally {
      setLoadingDataset(false);
    }
  }

  useEffect(() => {
    void fetchDataset();
  }, []);

  async function handleTrainModel() {
    setTraining(true);
    setErrorMsg(null);
    setNotice(null);
    try {
      const res = await fetch('/api/ml/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ falsePositivePenalty: penalty }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Training failed.');
      }
      onModelUpdated(data.model);
      setNotice(
        `Model v${data.model.version} trained via Python 3.11 (${Math.round(
          data.model.metrics.accuracy * 100
        )}% CV accuracy, ${(data.model.metrics.falsePositiveRate * 100).toFixed(
          1
        )}% FPR).`
      );
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to train Python ML model.');
    } finally {
      setTraining(false);
    }
  }

  async function handleImagePicked(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dataUrl = await resizeFileToDataUrl(file);
      setNewImageDataUrl(dataUrl);
      if (!newTitle.trim()) {
        setNewTitle(file.name.replace(/\.[^.]+$/, ''));
      }
    } catch (err: any) {
      setErrorMsg(err?.message || 'Could not process selected image.');
    }
  }

  async function handleAddSample(e: React.FormEvent) {
    e.preventDefault();
    if (!newImageDataUrl) {
      setErrorMsg('Select a road photo before adding to the dataset.');
      return;
    }
    setSubmittingSample(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/ml/dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: newTitle.trim() || undefined,
          label: newLabel,
          category: newCategory,
          sourceType: 'upload',
          imageDataUrl: newImageDataUrl,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to save sample.');
      setSamples((prev) => [data.sample, ...prev]);
      setNewTitle('');
      setNewImageDataUrl('');
      setNotice('Sample added. Click "Train Python Model" to update classifier weights.');
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to save labeled sample.');
    } finally {
      setSubmittingSample(false);
    }
  }

  async function handleDeleteSample(id: string) {
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/ml/dataset/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Delete failed.');
      setSamples((prev) => prev.filter((s) => s.id !== id));
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to delete sample.');
    }
  }

  const m = model?.metrics;

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6 pb-24 md:pb-12">
      {/* Header & Train Action */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            Python ML Classifier &amp; Training
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-2xl">
            Cost-sensitive L2 logistic regression trained in Python 3.11 on labeled pothole photos and hard-negative road scenes to minimize false positives.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label
              htmlFor="fp-penalty-select"
              className="text-xs font-medium text-slate-600 whitespace-nowrap"
            >
              False-Positive Penalty
            </label>
            <select
              id="fp-penalty-select"
              value={penalty}
              onChange={(e) => setPenalty(Number(e.target.value))}
              className="h-10 px-3 rounded-xl border border-slate-300 bg-white text-xs sm:text-sm font-mono text-slate-900"
            >
              <option value={1.5}>1.5× Balanced</option>
              <option value={2.5}>2.5× Recommended</option>
              <option value={3.5}>3.5× Strict</option>
              <option value={4.5}>4.5× Ultra-Strict</option>
            </select>
          </div>

          <button
            type="button"
            disabled={training}
            onClick={handleTrainModel}
            className="min-h-[42px] px-5 py-2 rounded-xl bg-amber-600 hover:bg-amber-700 disabled:opacity-60 text-white text-xs sm:text-sm font-semibold inline-flex items-center gap-2 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
          >
            {training ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Cpu className="w-4 h-4" />
            )}
            {training ? 'Training Python Model...' : 'Train Python Model'}
          </button>
        </div>
      </div>

      {notice && (
        <div className="px-4 py-3 rounded-xl bg-emerald-50 border border-emerald-200 text-xs sm:text-sm font-medium text-emerald-800 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{notice}</span>
        </div>
      )}

      {errorMsg && (
        <div className="px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-xs sm:text-sm font-medium text-red-700">
          {errorMsg}
        </div>
      )}

      {/* Cross-Validated Model Performance */}
      {model && m && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs text-slate-500">CV Accuracy</div>
              <div className="font-mono tabular-nums text-2xl font-bold text-slate-900 mt-1">
                {(m.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                {m.truePositives + m.trueNegatives}/{m.candidateRegionsEvaluated} regions
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs text-slate-500">Precision</div>
              <div className="font-mono tabular-nums text-2xl font-bold text-emerald-700 mt-1">
                {(m.precision * 100).toFixed(1)}%
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                TP={m.truePositives} · FP={m.falsePositives}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs text-slate-500">Recall</div>
              <div className="font-mono tabular-nums text-2xl font-bold text-slate-900 mt-1">
                {(m.recall * 100).toFixed(1)}%
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                FN={m.falseNegatives} missed
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs text-slate-500">False Positive Rate</div>
              <div className="font-mono tabular-nums text-2xl font-bold text-amber-700 mt-1">
                {(m.falsePositiveRate * 100).toFixed(1)}%
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                TN={m.trueNegatives} rejected
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs text-slate-500">ROC-AUC</div>
              <div className="font-mono tabular-nums text-2xl font-bold text-slate-900 mt-1">
                {m.rocAuc.toFixed(3)}
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                F0.5={(m.f05Score * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs text-slate-500">Threshold &amp; Version</div>
              <div className="font-mono tabular-nums text-2xl font-bold text-slate-900 mt-1">
                τ={model.decisionThreshold.toFixed(2)}
              </div>
              <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                v{model.version} · {m.trainingEpochs} epochs
              </div>
            </div>
          </div>

          {/* Model Details: Confusion Matrix + Feature Weights */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5 bg-white border border-slate-200 rounded-2xl p-5 space-y-4">
              <div>
                <h2 className="font-display text-base font-bold text-slate-900">
                  Cross-Validation Confusion Matrix
                </h2>
                <p className="text-xs text-slate-500 mt-0.5 font-mono">
                  {model.algorithm}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2.5 text-center">
                <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200">
                  <div className="text-[11px] font-medium text-emerald-800">
                    True Positive (Pothole Caught)
                  </div>
                  <div className="font-mono tabular-nums text-2xl font-bold text-emerald-900 mt-1">
                    {m.truePositives}
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-red-50 border border-red-200">
                  <div className="text-[11px] font-medium text-red-800">
                    False Positive (False Alarm)
                  </div>
                  <div className="font-mono tabular-nums text-2xl font-bold text-red-900 mt-1">
                    {m.falsePositives}
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200">
                  <div className="text-[11px] font-medium text-amber-800">
                    False Negative (Missed)
                  </div>
                  <div className="font-mono tabular-nums text-2xl font-bold text-amber-900 mt-1">
                    {m.falseNegatives}
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-medium text-slate-700">
                    True Negative (Suppressed)
                  </div>
                  <div className="font-mono tabular-nums text-2xl font-bold text-slate-900 mt-1">
                    {m.trueNegatives}
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-100 text-xs text-slate-600 flex flex-wrap justify-between gap-2 font-mono">
                <span>
                  Dataset: {m.positiveSamples} pos / {m.negativeSamples} neg images
                </span>
                <span>
                  Trained: {new Date(model.trainedAt).toLocaleString()}
                </span>
              </div>
            </div>

            <div className="lg:col-span-7 bg-white border border-slate-200 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-bold text-slate-900">
                  Learned 10D Feature Weights
                </h2>
                <span className="font-mono text-xs text-slate-500">
                  Bias: {model.bias.toFixed(3)}
                </span>
              </div>

              <div className="space-y-2">
                {model.featureNames.map((fname, idx) => {
                  const w = model.weights[idx] ?? 0;
                  const mag = Math.min(100, (Math.abs(w) / 1.8) * 100);
                  const isNeg = w < 0;
                  return (
                    <div
                      key={fname}
                      className="grid grid-cols-12 items-center gap-3 text-xs"
                    >
                      <div className="col-span-5 text-slate-700 font-medium truncate">
                        {FEATURE_LABELS[fname] || fname}
                      </div>
                      <div className="col-span-5 h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isNeg ? 'bg-red-500' : 'bg-emerald-600'
                          }`}
                          style={{ width: `${Math.max(4, mag)}%` }}
                        />
                      </div>
                      <div
                        className={`col-span-2 text-right font-mono tabular-nums font-semibold ${
                          isNeg ? 'text-red-700' : 'text-emerald-700'
                        }`}
                      >
                        {w >= 0 ? `+${w.toFixed(3)}` : w.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Add Labeled Training Image + Dataset Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <form
          onSubmit={handleAddSample}
          className="lg:col-span-4 bg-white border border-slate-200 rounded-2xl p-5 space-y-4 self-start"
        >
          <div className="flex items-center gap-2">
            <Plus className="w-4 h-4 text-amber-600" />
            <h2 className="font-display text-base font-bold text-slate-900">
              Add Labeled Road Sample
            </h2>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">
              Road Image
            </label>
            <label className="flex flex-col items-center justify-center p-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 hover:bg-slate-100 cursor-pointer text-center transition-colors">
              {newImageDataUrl ? (
                <img
                  src={newImageDataUrl}
                  alt="New labeled sample preview"
                  referrerPolicy="no-referrer"
                  className="w-full aspect-[4/3] object-cover rounded-lg"
                />
              ) : (
                <>
                  <Upload className="w-5 h-5 text-amber-600 mb-1.5" />
                  <span className="text-xs font-medium text-slate-700">
                    Choose pothole or clean road photo
                  </span>
                </>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleImagePicked}
                className="sr-only"
              />
            </label>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Ground-Truth Label
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => {
                  setNewLabel('pothole');
                  setNewCategory('pothole_cavity');
                }}
                className={`h-10 rounded-xl border text-xs font-semibold cursor-pointer ${
                  newLabel === 'pothole'
                    ? 'border-amber-600 bg-amber-50 text-amber-900'
                    : 'border-slate-300 bg-white text-slate-600'
                }`}
              >
                Pothole Hazard
              </button>
              <button
                type="button"
                onClick={() => {
                  setNewLabel('no_pothole');
                  setNewCategory('clean_road');
                }}
                className={`h-10 rounded-xl border text-xs font-semibold cursor-pointer ${
                  newLabel === 'no_pothole'
                    ? 'border-emerald-600 bg-emerald-50 text-emerald-900'
                    : 'border-slate-300 bg-white text-slate-600'
                }`}
              >
                No Pothole (Clean)
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Category
            </label>
            <select
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value as DatasetCategory)}
              className="w-full h-10 px-3 rounded-xl border border-slate-300 bg-white text-xs text-slate-900"
            >
              {CATEGORY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Sample Title
            </label>
            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              placeholder="e.g. Oak St Curb Shadow"
              className="w-full h-10 px-3 rounded-xl border border-slate-300 bg-white text-xs text-slate-900"
            />
          </div>

          <button
            type="submit"
            disabled={submittingSample || !newImageDataUrl}
            className="w-full min-h-[42px] rounded-xl bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs sm:text-sm font-semibold cursor-pointer transition-colors"
          >
            {submittingSample ? 'Saving...' : 'Add Sample to Dataset'}
          </button>
        </form>

        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-2xl overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
            <div>
              <h2 className="font-display text-base font-bold text-slate-900">
                Labeled Training Dataset ({samples.length})
              </h2>
              <p className="text-xs text-slate-500">
                Real road screenshots, synthetic cavities, hard-negative road artifacts, and user feedback
              </p>
            </div>
          </div>

          {loadingDataset ? (
            <div className="p-8 text-center text-xs text-slate-500">
              Loading training samples...
            </div>
          ) : (
            <div className="divide-y divide-slate-200 max-h-[540px] overflow-y-auto">
              {samples.map((s) => (
                <div
                  key={s.id}
                  className="px-5 py-3 flex items-center justify-between gap-4 hover:bg-slate-50"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span
                        className={`font-semibold ${
                          s.label === 'pothole' ? 'text-red-700' : 'text-emerald-700'
                        }`}
                      >
                        {s.label === 'pothole' ? 'Pothole' : 'No Pothole'}
                      </span>
                      <span aria-hidden="true" className="text-slate-300">
                        ·
                      </span>
                      <span className="font-mono text-slate-500">
                        {s.category}
                      </span>
                      <span aria-hidden="true" className="text-slate-300">
                        ·
                      </span>
                      <span className="text-slate-500 capitalize">
                        {s.sourceType}
                      </span>
                    </div>
                    <div className="text-sm font-medium text-slate-900 truncate mt-0.5">
                      {s.title}
                    </div>
                  </div>

                  {(s.sourceType === 'upload' || s.sourceType === 'feedback') && (
                    <button
                      type="button"
                      onClick={() => handleDeleteSample(s.id)}
                      aria-label={`Delete ${s.title}`}
                      className="p-2 rounded-lg text-slate-400 hover:text-red-600 hover:bg-red-50 cursor-pointer shrink-0"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
