import { useState } from 'react';
import { Check, ExternalLink, Globe, Server } from 'lucide-react';
import { Open311Config } from '../types';

interface PortalSettingsProps {
  config: Open311Config;
  onSaveConfig: (next: Open311Config) => void;
}

const PORTAL_DIRECTORY = [
  {
    region: 'United States',
    authority: 'Municipal 311 / Open311 / SeeClickFix / State DOT',
    url: 'https://seeclickfix.com/report',
    notes:
      'Automatically matches NYC 311, SF 311, Chicago 311, Seattle DOT, or local municipal 311 portals from GPS city lookup.',
  },
  {
    region: 'India',
    authority: 'CPGRAMS (MoRTH) / NHAI 1033 / Municipal Portals',
    url: 'https://pgportal.gov.in/',
    notes:
      'Matches BBMP Sahaya, MyBMC, MCD 311, MyGHMC, or national CPGRAMS grievance portal depending on city.',
  },
  {
    region: 'United Kingdom',
    authority: 'GOV.UK Report a Pothole & FixMyStreet',
    url: 'https://www.gov.uk/report-pothole',
    notes:
      'Routes directly to the responsible local council highways department using postcode & coordinates.',
  },
  {
    region: 'Canada',
    authority: 'Municipal 311 Transportation & Public Works',
    url: 'https://www.toronto.ca/home/311-toronto-at-your-service/',
    notes:
      'Supports 311 roadway maintenance reporting across Canadian municipalities.',
  },
  {
    region: 'Australia',
    authority: 'Snap Send Solve & State Road Authorities',
    url: 'https://www.snapsendsolve.com/',
    notes:
      'Dispatches geotagged road surface hazards to local councils across Australia.',
  },
];

export function PortalSettings({ config, onSaveConfig }: PortalSettingsProps) {
  const [draft, setDraft] = useState<Open311Config>(config);
  const [saved, setSaved] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    message: string;
  } | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSaveConfig(draft);
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  async function handleTestEndpoint() {
    if (!draft.endpointUrl.trim()) {
      setTestResult({ ok: false, message: 'Enter an endpoint URL first.' });
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch('/api/portals/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpointUrl: draft.endpointUrl.trim() }),
      });
      const data = await res.json();
      if (res.ok && data.reachable) {
        setTestResult({
          ok: true,
          message: `Endpoint reachable (HTTP ${data.status}).`,
        });
      } else {
        setTestResult({
          ok: false,
          message: data.error || 'Endpoint unreachable.',
        });
      }
    } catch (err: any) {
      setTestResult({
        ok: false,
        message: err?.message || 'Network test failed.',
      });
    } finally {
      setTesting(false);
    }
  }

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-8 pb-24 md:pb-10">
      <div>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
          Government Portal Routing &amp; API Dispatch
        </h1>
        <p className="text-sm text-slate-600 mt-1 max-w-2xl">
          RoadWatch automatically resolves your GPS coordinates to the local municipal road authority, and supports direct Open311 GeoReport v2 or webhook dispatch.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Open311 / Municipal API Integration Form */}
        <form
          onSubmit={handleSubmit}
          className="lg:col-span-7 bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 space-y-5"
        >
          <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 border border-amber-200 text-amber-700 flex items-center justify-center shrink-0">
                <Server className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-display text-base sm:text-lg font-semibold text-slate-900">
                  Direct Open311 / Municipal Webhook
                </h2>
                <p className="text-xs text-slate-600">
                  When enabled, RoadWatch POSTs complaints directly to your city Open311 API or webhook server.
                </p>
              </div>
            </div>

            <label className="inline-flex items-center gap-2 cursor-pointer shrink-0">
              <input
                type="checkbox"
                checked={draft.enabled}
                onChange={(e) =>
                  setDraft({ ...draft, enabled: e.target.checked })
                }
                className="w-5 h-5 accent-amber-600 rounded cursor-pointer"
              />
              <span className="text-sm font-medium text-slate-800">
                Enable
              </span>
            </label>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Protocol Mode
              </label>
              <select
                value={draft.mode}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    mode: e.target.value as 'open311' | 'webhook',
                  })
                }
                className="w-full h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              >
                <option value="open311">Open311 GeoReport v2 (Form POST)</option>
                <option value="webhook">Municipal JSON Webhook</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Service Code (Open311)
              </label>
              <input
                type="text"
                value={draft.serviceCode}
                onChange={(e) =>
                  setDraft({ ...draft, serviceCode: e.target.value })
                }
                placeholder="pothole"
                className="w-full h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Government Endpoint URL
            </label>
            <div className="flex flex-col sm:flex-row gap-2">
              <input
                type="url"
                value={draft.endpointUrl}
                onChange={(e) =>
                  setDraft({ ...draft, endpointUrl: e.target.value })
                }
                placeholder="https://city-311.gov/open311/v2/requests.json"
                className="flex-1 h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
              <button
                type="button"
                onClick={handleTestEndpoint}
                disabled={testing}
                className="min-h-[44px] px-4 rounded-xl border border-slate-300 bg-slate-50 hover:bg-slate-100 text-sm font-medium text-slate-800 whitespace-nowrap cursor-pointer"
              >
                {testing ? 'Testing...' : 'Test Endpoint'}
              </button>
            </div>
            {testResult && (
              <p
                className={`text-xs mt-1.5 font-medium ${
                  testResult.ok ? 'text-emerald-700' : 'text-red-600'
                }`}
              >
                {testResult.message}
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                API Key / Bearer Token (Optional)
              </label>
              <input
                type="password"
                value={draft.apiKey}
                onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
                placeholder="Optional municipality key"
                className="w-full h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Jurisdiction ID (Optional)
              </label>
              <input
                type="text"
                value={draft.jurisdictionId}
                onChange={(e) =>
                  setDraft({ ...draft, jurisdictionId: e.target.value })
                }
                placeholder="e.g. sanfrancisco.gov"
                className="w-full h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Citizen Name (for Govt Filing)
              </label>
              <input
                type="text"
                value={draft.reporterName}
                onChange={(e) =>
                  setDraft({ ...draft, reporterName: e.target.value })
                }
                placeholder="Optional full name"
                className="w-full h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Reporter Email (for Ticket Updates)
              </label>
              <input
                type="email"
                value={draft.reporterEmail}
                onChange={(e) =>
                  setDraft({ ...draft, reporterEmail: e.target.value })
                }
                placeholder="citizen@example.com"
                className="w-full h-11 px-3 rounded-xl border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-600"
              />
            </div>
          </div>

          <div className="pt-2 flex items-center justify-between gap-4">
            <p className="text-xs text-slate-500">
              Saved locally in your browser and attached to outgoing filings.
            </p>
            <button
              type="submit"
              className="min-h-[44px] px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white text-sm font-semibold inline-flex items-center gap-2 cursor-pointer transition-colors"
            >
              {saved ? (
                <>
                  <Check className="w-4 h-4" />
                  Settings Saved
                </>
              ) : (
                'Save Portal Config'
              )}
            </button>
          </div>
        </form>

        {/* Supported Jurisdictions Directory */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-2xl overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-200 flex items-center gap-2.5">
            <Globe className="w-5 h-5 text-slate-600" />
            <h2 className="font-display text-base font-semibold text-slate-900">
              Auto-Matched Civic Portals
            </h2>
          </div>

          <div className="divide-y divide-slate-200">
            {PORTAL_DIRECTORY.map((entry) => (
              <div key={entry.region} className="p-4 sm:p-5 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-amber-700">
                    {entry.region}
                  </span>
                  <a
                    href={entry.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-medium text-slate-700 hover:text-amber-700 inline-flex items-center gap-1"
                  >
                    Visit Portal
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                <div className="text-sm font-semibold text-slate-900">
                  {entry.authority}
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {entry.notes}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
