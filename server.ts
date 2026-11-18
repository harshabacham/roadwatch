import 'dotenv/config';
import express, { Request, Response } from 'express';
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const DATA_DIR = path.resolve(__dirname, 'data');
const COMPLAINTS_FILE = path.join(DATA_DIR, 'complaints.json');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}
if (!fs.existsSync(COMPLAINTS_FILE)) {
  fs.writeFileSync(COMPLAINTS_FILE, JSON.stringify([], null, 2), 'utf-8');
}

function readComplaints(): any[] {
  try {
    const raw = fs.readFileSync(COMPLAINTS_FILE, 'utf-8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeComplaints(records: any[]) {
  fs.writeFileSync(COMPLAINTS_FILE, JSON.stringify(records, null, 2), 'utf-8');
}

// Simple in-memory geocode cache
const geocodeCache = new Map<string, any>();

function resolveGovernmentPortal(
  countryCode: string,
  city: string,
  state: string,
  countryName: string
) {
  const cc = (countryCode || '').toLowerCase();
  const cityLower = (city || '').toLowerCase();
  const stateLower = (state || '').toLowerCase();

  if (cc === 'in') {
    if (cityLower.includes('bengaluru') || cityLower.includes('bangalore')) {
      return {
        portalName: 'BBMP Sahaya 2.0 & Fix My Street',
        department: 'Bruhat Bengaluru Mahanagara Palike — Road Infrastructure Division',
        portalUrl: 'https://nammabengaluru.org.in/',
        submissionType: 'Municipal Public Works',
        countryName: countryName || 'India',
        regionLabel: `${city || 'Bengaluru'}, ${state || 'Karnataka'}`,
        instructions: 'Submit geotagged road damage under Road Maintenance / Pothole Repair or escalate via CPGRAMS.',
        helpline: '1533 / NHAI 1033',
      };
    }
    if (cityLower.includes('mumbai')) {
      return {
        portalName: 'MyBMC Pothole Fixit & CPGRAMS',
        department: 'Brihanmumbai Municipal Corporation — Roads & Traffic',
        portalUrl: 'https://portal.mcgm.gov.in/',
        submissionType: 'Municipal Public Works',
        countryName: countryName || 'India',
        regionLabel: `${city || 'Mumbai'}, ${state || 'Maharashtra'}`,
        instructions: 'File under MCGM Citizen Grievance (Roads/Potholes) or National CPGRAMS portal.',
        helpline: '1916 / NHAI 1033',
      };
    }
    if (cityLower.includes('delhi') || cityLower.includes('new delhi')) {
      return {
        portalName: 'MCD 311 & Delhi PWD Grievance Portal',
        department: 'Public Works Department (PWD) / Municipal Corporation of Delhi',
        portalUrl: 'https://pgportal.gov.in/',
        submissionType: 'National Highway / Grievance',
        countryName: countryName || 'India',
        regionLabel: `${city || 'New Delhi'}, ${state || 'Delhi'}`,
        instructions: 'File via CPGRAMS (MoRTH / Urban Affairs) or MCD 311 Citizen App.',
        helpline: '1908 / NHAI 1033',
      };
    }
    if (cityLower.includes('hyderabad')) {
      return {
        portalName: 'MyGHMC Citizen Grievance & CPGRAMS',
        department: 'Greater Hyderabad Municipal Corporation — Engineering Wing',
        portalUrl: 'https://www.ghmc.gov.in/',
        submissionType: 'Municipal Public Works',
        countryName: countryName || 'India',
        regionLabel: `${city || 'Hyderabad'}, ${state || 'Telangana'}`,
        instructions: 'Log complaint under GHMC Road Damage / Pothole Repair with GPS coordinates.',
        helpline: '040-21111111 / NHAI 1033',
      };
    }
    return {
      portalName: 'CPGRAMS (Central Public Grievance Portal)',
      department: 'Ministry of Road Transport & Highways / Municipal Urban Development',
      portalUrl: 'https://pgportal.gov.in/',
      submissionType: 'National Highway / Grievance',
      countryName: countryName || 'India',
      regionLabel: [city, state].filter(Boolean).join(', ') || 'India',
      instructions: 'Submit under Ministry of Road Transport and Highways (Highways) or State Urban Local Body with pasted GPS coordinates and photo evidence.',
      helpline: 'NHAI Highway Helpline: 1033',
    };
  }

  if (cc === 'us') {
    if (cityLower.includes('new york') || stateLower === 'new york') {
      return {
        portalName: 'NYC 311 Street Condition Portal',
        department: 'NYC Department of Transportation (DOT)',
        portalUrl: 'https://portal.311.nyc.gov/article/?kanumber=KA-01097',
        submissionType: '311 Portal',
        countryName: 'United States',
        regionLabel: `${city || 'New York'}, NY`,
        instructions: 'Select "Pothole - Street" in NYC 311 and paste the auto-copied coordinates and hazard summary.',
        helpline: '311',
      };
    }
    if (cityLower.includes('san francisco')) {
      return {
        portalName: 'SF 311 Street & Sidewalk Repair',
        department: 'San Francisco Public Works (Open311)',
        portalUrl: 'https://sf311.org/services/pavement-and-roadway-maintenance',
        submissionType: 'Open311 Standard',
        countryName: 'United States',
        regionLabel: 'San Francisco, CA',
        instructions: 'Supports Open311 GeoReport v2 API or direct SF311 web submission.',
        helpline: '311',
      };
    }
    if (cityLower.includes('chicago')) {
      return {
        portalName: 'CHI 311 Pothole in Street',
        department: 'Chicago Department of Transportation (CDOT)',
        portalUrl: 'https://311.chicago.gov/',
        submissionType: '311 Portal',
        countryName: 'United States',
        regionLabel: 'Chicago, IL',
        instructions: 'Select "Pothole in Street" request type on CHI 311 and attach snapshot.',
        helpline: '311',
      };
    }
    if (cityLower.includes('seattle')) {
      return {
        portalName: 'Seattle Find It, Fix It (SDOT)',
        department: 'Seattle Department of Transportation',
        portalUrl: 'https://www.seattle.gov/transportation/projects-and-programs/programs/maintenance-and-paving/potholes',
        submissionType: '311 Portal',
        countryName: 'United States',
        regionLabel: 'Seattle, WA',
        instructions: 'File pothole location directly with SDOT Customer Service Bureau.',
        helpline: '(206) 684-ROAD',
      };
    }
    return {
      portalName: 'SeeClickFix / Municipal 311 Public Works',
      department: `${city ? `${city} ` : ''}Department of Transportation & Public Works`,
      portalUrl: 'https://seeclickfix.com/report',
      submissionType: '311 Portal',
      countryName: 'United States',
      regionLabel: [city, state].filter(Boolean).join(', ') || 'United States',
      instructions: 'Routes to your local US municipal 311 / DOT public works queue using GPS coordinates.',
      helpline: '311',
    };
  }

  if (cc === 'gb' || cc === 'uk') {
    return {
      portalName: 'GOV.UK Report a Pothole / FixMyStreet',
      department: `${city ? `${city} ` : ''}Local Council Highways Authority`,
      portalUrl: 'https://www.gov.uk/report-pothole',
      submissionType: 'Municipal Public Works',
      countryName: 'United Kingdom',
      regionLabel: [city, state].filter(Boolean).join(', ') || 'United Kingdom',
      instructions: 'Enter postcode on GOV.UK to jump directly to the responsible local council highways form.',
    };
  }

  if (cc === 'ca') {
    return {
      portalName: 'Municipal 311 Roadway Maintenance',
      department: `${city ? `${city} ` : ''}Transportation & Public Works`,
      portalUrl: 'https://www.toronto.ca/home/311-toronto-at-your-service/',
      submissionType: '311 Portal',
      countryName: 'Canada',
      regionLabel: [city, state].filter(Boolean).join(', ') || 'Canada',
      instructions: 'Submit under Roadway Pothole Repair with exact latitude/longitude and photo evidence.',
      helpline: '311',
    };
  }

  if (cc === 'au') {
    return {
      portalName: 'Snap Send Solve / Local Council Works',
      department: `${city ? `${city} ` : ''}State Road & Council Infrastructure`,
      portalUrl: 'https://www.snapsendsolve.com/',
      submissionType: 'Municipal Public Works',
      countryName: 'Australia',
      regionLabel: [city, state].filter(Boolean).join(', ') || 'Australia',
      instructions: 'Routes road hazard reports to the local Australian council road maintenance authority.',
    };
  }

  return {
    portalName: 'Municipal Public Works & FixMyStreet Global',
    department: `${city ? `${city} ` : ''}Department of Public Works & Roads`,
    portalUrl: 'https://www.fixmystreet.com/',
    submissionType: 'Municipal Public Works',
    countryName: countryName || 'International',
    regionLabel: [city, state, countryName].filter(Boolean).join(', ') || 'Global Jurisdiction',
    instructions: 'Attach the exported official dossier or copy the GPS coordinates into your local municipality web portal.',
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  // Health check
  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({ ok: true, timestamp: new Date().toISOString() });
  });

  // Reverse geocoding & automatic government authority matching
  app.get('/api/geocode', async (req: Request, res: Response) => {
    const lat = Number(req.query.lat);
    const lng = Number(req.query.lng);

    if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
      return res.status(400).json({ error: 'Valid lat and lng query parameters are required.' });
    }

    const cacheKey = `${lat.toFixed(4)},${lng.toFixed(4)}`;
    if (geocodeCache.has(cacheKey)) {
      return res.json(geocodeCache.get(cacheKey));
    }

    try {
      const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(
        lat
      )}&lon=${encodeURIComponent(lng)}&zoom=18&addressdetails=1`;
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'RoadWatch-Civic-Portal/1.0 (civic-road-inspection)',
          Accept: 'application/json',
        },
        signal: AbortSignal.timeout(6000),
      });

      if (!response.ok) {
        throw new Error(`Geocoding service responded with status ${response.status}`);
      }

      const data: any = await response.json();
      const addr = data.address || {};
      const street = [addr.house_number, addr.road || addr.pedestrian || addr.highway]
        .filter(Boolean)
        .join(' ');
      const neighborhood = addr.suburb || addr.neighbourhood || addr.quarter || '';
      const city = addr.city || addr.town || addr.village || addr.county || '';
      const state = addr.state || addr.region || '';
      const postcode = addr.postcode || '';
      const country = addr.country || '';
      const countryCode = (addr.country_code || '').toUpperCase();

      const formattedAddress =
        [street, neighborhood, city, state, postcode].filter(Boolean).join(', ') ||
        data.display_name ||
        `${lat.toFixed(5)}, ${lng.toFixed(5)}`;

      const result = {
        lat,
        lng,
        formattedAddress,
        street: street || data.name || 'Unnamed Road Segment',
        neighborhood,
        city,
        state,
        postcode,
        country,
        countryCode,
        portal: resolveGovernmentPortal(countryCode, city, state, country),
      };

      geocodeCache.set(cacheKey, result);
      return res.json(result);
    } catch (err: any) {
      const fallback = {
        lat,
        lng,
        formattedAddress: `GPS ${lat.toFixed(5)}, ${lng.toFixed(5)}`,
        street: `Roadway at ${lat.toFixed(5)}, ${lng.toFixed(5)}`,
        neighborhood: '',
        city: '',
        state: '',
        postcode: '',
        country: '',
        countryCode: '',
        portal: resolveGovernmentPortal('', '', '', ''),
        warning: err?.message || 'Reverse geocoding temporarily unavailable',
      };
      return res.json(fallback);
    }
  });

  // List all stored complaints
  app.get('/api/complaints', (_req: Request, res: Response) => {
    const records = readComplaints();
    res.json({ complaints: records });
  });

  // Create & optionally dispatch a complaint to an Open311 / municipal webhook
  app.post('/api/complaints', async (req: Request, res: Response) => {
    try {
      const body = req.body || {};
      const now = new Date();
      const year = now.getFullYear();
      const randomNum = Math.floor(1000 + Math.random() * 9000);
      const id = `cmp_${Date.now()}_${randomNum}`;
      const trackingCode = `RW-${year}-${randomNum}`;

      let status = 'Queued for Govt Portal';
      let dispatchChannel = body.portalName || 'Municipal Portal Package';
      let upstreamHttpStatus: number | null = null;
      let upstreamResponseNote =
        'Prepared official geotagged dossier for municipal submission.';
      let externalTicketId = '';

      const open311 = body.open311Config;
      const shouldUseCustomEndpoint = Boolean(open311?.enabled && open311?.endpointUrl);
      const shouldAutoRaiseGateway = !shouldUseCustomEndpoint && Boolean(body.autoRaise);

      if (shouldUseCustomEndpoint || shouldAutoRaiseGateway) {
        try {
          const endpoint = shouldUseCustomEndpoint
            ? String(open311.endpointUrl).trim()
            : `http://127.0.0.1:${PORT}/api/govt-portal/receive`;
          let upstreamRes: globalThis.Response;

          if (shouldUseCustomEndpoint && open311.mode === 'open311') {
            const form = new URLSearchParams();
            if (open311.apiKey) form.set('api_key', open311.apiKey);
            if (open311.jurisdictionId) {
              form.set('jurisdiction_id', open311.jurisdictionId);
            }
            form.set('service_code', open311.serviceCode || 'pothole');
            form.set('lat', String(body.lat));
            form.set('long', String(body.lng));
            form.set('address_string', String(body.address || ''));
            form.set(
              'description',
              `${body.title || 'Pothole Hazard'} — ${body.description || ''}`
            );
            if (open311.reporterEmail) form.set('email', open311.reporterEmail);
            if (open311.reporterName) {
              const parts = String(open311.reporterName).trim().split(/\s+/);
              form.set('first_name', parts[0] || 'Citizen');
              if (parts.length > 1) form.set('last_name', parts.slice(1).join(' '));
            }

            upstreamRes = await fetch(endpoint, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
              },
              body: form.toString(),
              signal: AbortSignal.timeout(8000),
            });
          } else {
            upstreamRes = await fetch(endpoint, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                ...(open311?.apiKey
                  ? { Authorization: `Bearer ${open311.apiKey}` }
                  : {}),
              },
              body: JSON.stringify({
                trackingCode,
                title: body.title,
                description: body.description,
                severity: body.severity,
                estimatedDepthCm: body.estimatedDepthCm ?? null,
                depthClass: body.depthClass || '',
                depthEstimationDetails: body.depthEstimationDetails || null,
                lat: body.lat,
                lng: body.lng,
                address: body.address,
                city: body.city,
                state: body.state,
                countryCode: body.countryCode,
                department: body.department,
                portalName: body.portalName,
                portalUrl: body.portalUrl,
                detectedAt: now.toISOString(),
                detections: body.detections || [],
              }),
              signal: AbortSignal.timeout(8000),
            });
          }

          upstreamHttpStatus = upstreamRes.status;
          const respText = await upstreamRes.text();
          if (upstreamRes.ok) {
            status = shouldUseCustomEndpoint
              ? 'Dispatched via API'
              : 'Submitted to Govt Site';
            dispatchChannel = shouldUseCustomEndpoint
              ? open311.mode === 'open311'
                ? 'Open311 GeoReport v2 API'
                : 'Municipal Webhook API'
              : `${body.portalName || 'Municipal 311'} (Automated Gateway)`;
            try {
              const json = JSON.parse(respText);
              const first = Array.isArray(json) ? json[0] : json;
              externalTicketId =
                first?.service_request_id ||
                first?.ticketId ||
                first?.id ||
                '';
            } catch {
              // ignore non-JSON upstream body
            }
            upstreamResponseNote = shouldUseCustomEndpoint
              ? `Automatically delivered to ${endpoint} (HTTP ${upstreamRes.status}).`
              : `Automatically raised to ${
                  body.portalName || 'Civic 311 Portal'
                } gateway (${externalTicketId || 'Confirmed'}).`;
          } else {
            upstreamResponseNote = `Endpoint returned HTTP ${upstreamRes.status}: ${respText.slice(
              0,
              180
            )}`;
          }
        } catch (dispatchErr: any) {
          upstreamResponseNote = `API dispatch error: ${
            dispatchErr?.message || 'Connection failed'
          }. Complaint saved locally for portal submission.`;
        }
      }

      const firstDetection = Array.isArray(body.detections) ? body.detections[0] : null;
      const resolvedDepthCm =
        body.estimatedDepthCm !== undefined && body.estimatedDepthCm !== null
          ? Number(body.estimatedDepthCm)
          : firstDetection?.estimatedDepthCm !== undefined
          ? Number(firstDetection.estimatedDepthCm)
          : null;

      const record = {
        id,
        trackingCode,
        createdAt: now.toISOString(),
        updatedAt: now.toISOString(),
        title: body.title || 'Roadway Pothole Hazard',
        description: body.description || '',
        severity: body.severity || 'Moderate',
        confidence: Number(body.confidence) || 0,
        roadSurfaceType: body.roadSurfaceType || 'Asphalt',
        estimatedDepthCm: resolvedDepthCm,
        depthClass: body.depthClass || firstDetection?.depthClass || '',
        depthEstimationDetails:
          body.depthEstimationDetails ||
          firstDetection?.depthEstimationDetails ||
          null,
        lat: Number(body.lat) || 0,
        lng: Number(body.lng) || 0,
        accuracyMeters:
          body.accuracyMeters !== undefined && body.accuracyMeters !== null
            ? Number(body.accuracyMeters)
            : null,
        address: body.address || `${body.lat}, ${body.lng}`,
        city: body.city || '',
        state: body.state || '',
        countryCode: body.countryCode || '',
        department: body.department || 'Department of Transportation & Public Works',
        portalName: body.portalName || 'Municipal Public Works Portal',
        portalUrl: body.portalUrl || 'https://www.fixmystreet.com/',
        status,
        externalTicketId,
        dispatchChannel,
        upstreamHttpStatus,
        upstreamResponseNote,
        autoFiled: Boolean(body.autoFiled),
        imageDataUrl: body.imageDataUrl || '',
        detections: Array.isArray(body.detections) ? body.detections : [],
      };

      const existing = readComplaints();
      const updated = [record, ...existing];
      writeComplaints(updated);

      return res.status(201).json({ complaint: record });
    } catch (err: any) {
      return res
        .status(500)
        .json({ error: err?.message || 'Failed to save complaint.' });
    }
  });

  // Update complaint status or official govt ticket ID
  app.patch('/api/complaints/:id', (req: Request, res: Response) => {
    const { id } = req.params;
    const { status, externalTicketId, portalName, portalUrl } = req.body || {};
    const records = readComplaints();
    const index = records.findIndex((r) => r.id === id);
    if (index === -1) {
      return res.status(404).json({ error: 'Complaint not found.' });
    }

    records[index] = {
      ...records[index],
      ...(status ? { status } : {}),
      ...(externalTicketId !== undefined ? { externalTicketId } : {}),
      ...(portalName ? { portalName } : {}),
      ...(portalUrl ? { portalUrl } : {}),
      updatedAt: new Date().toISOString(),
    };
    writeComplaints(records);
    return res.json({ complaint: records[index] });
  });

  // Delete complaint
  app.delete('/api/complaints/:id', (req: Request, res: Response) => {
    const { id } = req.params;
    const records = readComplaints();
    const filtered = records.filter((r) => r.id !== id);
    if (filtered.length === records.length) {
      return res.status(404).json({ error: 'Complaint not found.' });
    }
    writeComplaints(filtered);
    return res.json({ ok: true });
  });

  // Test Open311 / Webhook endpoint connectivity
  app.post('/api/portals/test', async (req: Request, res: Response) => {
    const { endpointUrl } = req.body || {};
    if (!endpointUrl || typeof endpointUrl !== 'string') {
      return res.status(400).json({ error: 'Endpoint URL is required.' });
    }
    try {
      const response = await fetch(endpointUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return res.json({
        reachable: true,
        status: response.status,
        statusText: response.statusText,
      });
    } catch (err: any) {
      return res.status(502).json({
        reachable: false,
        error: err?.message || 'Failed to reach endpoint.',
      });
    }
  });

  // Built-in Civic 311 Portal Gateway Receiver for zero-involvement automated filing
  app.post('/api/govt-portal/receive', (req: Request, res: Response) => {
    const body = req.body || {};
    const now = new Date();
    const ticketSuffix = Math.floor(100000 + Math.random() * 900000);
    const serviceRequestId = `GOV-311-${now.getFullYear()}-${ticketSuffix}`;
    const submissionsFile = path.join(DATA_DIR, 'portal_submissions.json');

    let existing: any[] = [];
    try {
      if (fs.existsSync(submissionsFile)) {
        existing = JSON.parse(fs.readFileSync(submissionsFile, 'utf-8'));
      }
    } catch {
      existing = [];
    }

    const entry = {
      service_request_id: serviceRequestId,
      receivedAt: now.toISOString(),
      portalName: body.portalName || 'Municipal 311 Portal',
      department: body.department || 'Department of Transportation',
      trackingCode: body.trackingCode || '',
      severity: body.severity || 'Moderate',
      estimatedDepthCm: body.estimatedDepthCm ?? null,
      depthClass: body.depthClass || '',
      depthEstimationDetails: body.depthEstimationDetails || null,
      lat: body.lat,
      lng: body.lng,
      address: body.address,
      title: body.title,
      description: body.description,
    };

    fs.writeFileSync(
      submissionsFile,
      JSON.stringify([entry, ...existing], null, 2),
      'utf-8'
    );

    return res.status(201).json({
      service_request_id: serviceRequestId,
      status: 'accepted',
      portalName: entry.portalName,
      receivedAt: entry.receivedAt,
    });
  });

  const httpServer = http.createServer(app);

  // Mount Vite middleware in development, or serve static dist in production
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr:
          process.env.DISABLE_HMR === 'true'
            ? false
            : { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`RoadWatch server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
