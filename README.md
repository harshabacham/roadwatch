# RoadWatch

A mobile-first **Progressive Web App (PWA)** for real-time road pothole detection, cavity depth estimation, GPS geotagging, human-in-the-loop verification (**Approve / Reject**), and automated municipal complaint filing.

---

## Key Features

- **Real-Time Camera Inspection with Explicit Standby Controls**
  - Opens in a standby state and starts recording only when the user taps **Start Recording**, with **Stop Recording** available at any time.
  - Supports mobile rear-camera capture (`facingMode: 'environment'`), inline playback, flashlight/torch toggling when supported, and frame upload inspection.
- **Unsupervised Machine Learning & Depth Estimation**
  - **Patch Feature Extraction & K-Means++ Segmentation (`src/ml/unsupervisedDetector.ts`):** Splits each downsampled road frame (`240×180`, `40×30` patch grid) into surface clusters and learns the normal asphalt baseline dynamically per frame without labeled training datasets.
  - **Robust Mahalanobis Anomaly Scoring & DBSCAN Clustering:** Identifies depressed, rough, high-contrast road cavities via Median/MAD standardized anomaly Z-scores and groups adjacent patches with DBSCAN while pruning isolated noise.
  - **Unsupervised Depth & Severity Estimator (`src/ml/depthEstimator.ts`):** Combines 16-bin Shannon image entropy, perspective-normalized relative scale, and interior shadow occlusion to estimate cavity depth (`2.5–26.0 cm`) and classify hazards (`Shallow`, `Medium`, `Deep`, `Severe`).
- **Human-in-the-Loop Review Queue**
  - Captures pothole evidence screenshots and metrics for every detection without auto-filing until the user explicitly clicks **Approve** or **Reject**.
- **GPS Geotagging & Government Portal Routing**
  - Captures live coordinates (`navigator.geolocation`), performs reverse geocoding via OpenStreetMap Nominatim (`/api/geocode`), maps jurisdictions to local portals (e.g., CPGRAMS, BBMP, MyBMC, NYC/SF/CHI/Seattle 311, GOV.UK FixMyStreet), and displays incidents on an interactive Leaflet hazard map.
  - Supports direct dispatch to custom **Open311 GeoReport v2** endpoints, custom municipal webhooks, or the built-in Civic 311 gateway (`/api/govt-portal/receive`).
- **Offline-First PWA & Background Sync**
  - Built with `vite-plugin-pwa` (`injectManifest`) and Workbox (`src/sw.ts`).
  - Persists detections, queued complaints, and cached records in **IndexedDB** (`src/offline/indexedDb.ts`) with automatic **Background Sync** (`src/offline/syncManager.ts`) when connectivity returns.
  - Includes an in-app install prompt (`PWAInstallButton`) and online/offline sync indicator (`OfflineSyncBar`).

---

## Tech Stack

- **Frontend:** React 19, TypeScript, Vite 8, Tailwind CSS v4, Lucide Icons, Motion, Leaflet
- **PWA & Offline Storage:** `vite-plugin-pwa`, `workbox-precaching`, IndexedDB, Service Worker Background Sync
- **Backend:** Node.js, Express (`server.ts`), JSON file persistence (`data/complaints.json`, `data/portal_submissions.json`)

---

## Unsupervised Machine Learning Topics Used

RoadWatch operates without pre-labeled pothole training datasets by learning the normal road surface dynamically per frame using the following unsupervised ML techniques:

1. **Patch-Based Feature Engineering**
   - Downsamples each frame to `240×180` (`40×30 = 1,200` `6×6` px patches) and computes mean luminance, luminance standard deviation, Sobel gradient roughness, peak gradient, chroma shift, warmth, and local background depression/contrast.
2. **K-Means++ Clustering (`K = 3, 4, 5`)**
   - Uses K-Means++ centroid initialization and iterative assignment to partition road patches into asphalt baseline, surface transitions, rough/dark cavities, road markings, and specular glare.
   - Evaluates clustering quality in real time via **Within-Cluster Sum of Squares (WCSS)** and **Silhouette Score**.
3. **Robust Baseline Modeling (Median & MAD)**
   - Estimates the normal asphalt distribution using **Median** and **Median Absolute Deviation (MAD)** so existing potholes do not skew the reference statistics.
4. **Mahalanobis Distance Anomaly Scoring**
   - Scores every road patch relative to the learned normal asphalt baseline using standardized **Mahalanobis Z-scores** to detect depressed, textured anomalies.
5. **DBSCAN Density-Based Spatial Clustering**
   - Groups adjacent anomalous patches (`eps = 1.65`, configurable `minPts = 3, 4, 6`) into coherent pothole bounding boxes while pruning isolated single-patch noise.
6. **Unsupervised Cavity Depth Estimation**
   - Combines **16-Bin Shannon Image Entropy** ($H = -\sum p_i \log_2 p_i$) against learned normal road entropy, **Perspective-Normalized Relative Scale Analysis**, and **Interior Shadow Occlusion Ratio** to estimate depth (`cm`) and severity class.

---

## Project Structure

```text
├── public/                     # PWA icons (SVG/PNG, maskable & apple-touch-icon)
├── data/                       # Local JSON storage for complaints & 311 gateway receipts
├── src/
│   ├── components/
│   │   ├── LiveScanner.tsx     # Camera inspection, Start/Stop controls, Approve/Reject queue
│   │   ├── HazardMap.tsx       # Interactive Leaflet map of reported road hazards
│   │   ├── ComplaintLog.tsx    # Filed complaints history & status management
│   │   ├── ComplaintDossierModal.tsx # Printable/exportable hazard dossier
│   │   ├── PortalSettings.tsx  # Open311 / Municipal Webhook configuration
│   │   ├── OfflineSyncBar.tsx  # Offline status & queued sync banner
│   │   ├── PWAInstallButton.tsx# Cross-platform PWA install button + iOS guide
│   │   └── TopNav.tsx          # App navigation bar
│   ├── hooks/
│   │   ├── useOnlineStatus.ts  # Browser connectivity status hook
│   │   └── usePWAInstall.ts    # beforeinstallprompt & standalone detection hook
│   ├── ml/
│   │   ├── unsupervisedDetector.ts # K-Means++, Mahalanobis Z-score, DBSCAN pipeline
│   │   └── depthEstimator.ts   # Shannon entropy, shadow occlusion & perspective depth model
│   ├── offline/
│   │   ├── dbConstants.ts      # IndexedDB database/store names and sync tags
│   │   ├── indexedDb.ts        # Local persistence for detections, queue & cache
│   │   └── syncManager.ts      # Service worker registration & Background Sync flush
│   ├── utils/
│   │   └── geo.ts              # Distance & coordinate helpers
│   ├── App.tsx                 # Main application shell & state orchestration
│   ├── sw.ts                   # Custom Service Worker (precaching + Background Sync)
│   └── types.ts                # Shared TypeScript interfaces
├── server.ts                   # Express API server & Vite middleware host
├── vite.config.ts              # Vite + Tailwind + VitePWA configuration
├── index.html                  # HTML entry point & PWA meta tags
└── package.json
```

---

## Getting Started

### Prerequisites

- **Node.js** 18+ (or compatible runtime)
- **npm** (or **bun**)

### Installation

```bash
npm install
```

### Runtime Configuration

Optional server environment variables:
- `PORT` — Server port (defaults to `3000`).
- `DISABLE_HMR` — Set to `true` to disable Vite HMR file watching in hosted preview environments.
- `APP_URL` — Public service URL where the application is hosted.

### Running in Development

Starts the Express API server with Vite middleware on port `3000`:

```bash
npm run dev
```

### Building & Running in Production

```bash
npm run build
NODE_ENV=production npm start
```

### Type Checking

```bash
npm run lint
```

---

## Backend API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/geocode?lat=&lng=` | Reverse geocode coordinates and resolve matching municipal portal |
| `GET` | `/api/complaints` | List saved pothole complaints |
| `POST` | `/api/complaints` | Save a complaint and optionally dispatch to Open311, webhook, or built-in gateway |
| `PATCH` | `/api/complaints/:id` | Update complaint status, external ticket ID, or portal metadata |
| `DELETE` | `/api/complaints/:id` | Delete a stored complaint |
| `POST` | `/api/portals/test` | Test reachability of a configured Open311 or webhook URL |
| `POST` | `/api/govt-portal/receive` | Built-in Civic 311 gateway receiver returning a `GOV-311-*` ticket ID |

---

## Browser Permissions

- **Camera (`camera`):** Required for live road scanning.
- **Location (`geolocation`):** Required to geotag detections and match local public works portals.
