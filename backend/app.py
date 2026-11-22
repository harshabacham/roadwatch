#!/usr/bin/env python3
"""
RoadWatch Full Python 3.11 Backend Server.

Handles 100% of backend API logic for RoadWatch using Python 3.11 stdlib:
  - /api/health
  - /api/detect/roboflow       (in-process Python PotholeDetectionModel)
  - /api/geocode               (Nominatim reverse geocoding + govt portal resolver)
  - /api/complaints            (CRUD + 5-meter Haversine spatial clustering + Open311/311 dispatch)
  - /api/portals/test          (Open311 / webhook reachability test)
  - /api/govt-portal/receive   (Built-in Civic 311 gateway receiver)
  - /api/datasets/kaggle*      (Kaggle dataset catalog, image streaming, CSV export, import)
  - /api/ml/model              (Trained classifier artifact lookup)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "ml"))

from pothole_model import PotholeDetectionModel  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
COMPLAINTS_FILE = DATA_DIR / "complaints.json"
SUBMISSIONS_FILE = DATA_DIR / "portal_submissions.json"
TRAINED_MODEL_FILE = DATA_DIR / "trained_pothole_model.json"
KAGGLE_JSON_FILE = DATA_DIR / "kaggle_pothole_dataset.json"
KAGGLE_CSV_FILE = DATA_DIR / "kaggle_pothole_dataset.csv"
KAGGLE_IMAGES_DIR = DATA_DIR / "kaggle_images"
KAGGLE_IMPORTER_SCRIPT = ROOT_DIR / "ml" / "import_kaggle_dataset.py"

BACKEND_HOST = os.environ.get("PYTHON_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("PYTHON_BACKEND_PORT", "8000"))

COMPLAINT_CLUSTER_RADIUS_METERS = 5.0
SEVERITY_RANK: dict[str, int] = {
    "None": 0,
    "Low": 1,
    "Moderate": 2,
    "Severe": 3,
    "Critical": 4,
}


def _load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv(ROOT_DIR / ".env")

DATA_DIR.mkdir(parents=True, exist_ok=True)
if not COMPLAINTS_FILE.exists():
    COMPLAINTS_FILE.write_text("[]\n", encoding="utf-8")

# Thread-safe locks & per-thread model instances (keeps HTTPS keep-alive safe across threads)
_complaints_lock = threading.Lock()
_submissions_lock = threading.Lock()
_geocode_lock = threading.Lock()
_geocode_cache: dict[str, dict[str, Any]] = {}
_kaggle_locks_guard = threading.Lock()
_kaggle_image_locks: dict[str, threading.Lock] = {}
_thread_local = threading.local()


def _get_thread_model() -> PotholeDetectionModel:
    model = getattr(_thread_local, "model", None)
    if model is None:
        model = PotholeDetectionModel()
        _thread_local.model = model
    return model


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_complaints() -> list[dict[str, Any]]:
    try:
        raw = json.loads(COMPLAINTS_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _write_complaints(records: list[dict[str, Any]]) -> None:
    COMPLAINTS_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_m = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return earth_radius_m * c


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
        return num if math.isfinite(num) else default
    except (TypeError, ValueError):
        return default


def resolve_government_portal(
    country_code: str,
    city: str,
    state: str,
    country_name: str,
) -> dict[str, Any]:
    cc = (country_code or "").lower()
    city_lower = (city or "").lower()
    state_lower = (state or "").lower()

    if cc == "in":
        if "bengaluru" in city_lower or "bangalore" in city_lower:
            return {
                "portalName": "BBMP Sahaya 2.0 & Fix My Street",
                "department": "Bruhat Bengaluru Mahanagara Palike — Road Infrastructure Division",
                "portalUrl": "https://nammabengaluru.org.in/",
                "submissionType": "Municipal Public Works",
                "countryName": country_name or "India",
                "regionLabel": f"{city or 'Bengaluru'}, {state or 'Karnataka'}",
                "instructions": "Submit geotagged road damage under Road Maintenance / Pothole Repair or escalate via CPGRAMS.",
                "helpline": "1533 / NHAI 1033",
            }
        if "mumbai" in city_lower:
            return {
                "portalName": "MyBMC Pothole Fixit & CPGRAMS",
                "department": "Brihanmumbai Municipal Corporation — Roads & Traffic",
                "portalUrl": "https://portal.mcgm.gov.in/",
                "submissionType": "Municipal Public Works",
                "countryName": country_name or "India",
                "regionLabel": f"{city or 'Mumbai'}, {state or 'Maharashtra'}",
                "instructions": "File under MCGM Citizen Grievance (Roads/Potholes) or National CPGRAMS portal.",
                "helpline": "1916 / NHAI 1033",
            }
        if "delhi" in city_lower or "new delhi" in city_lower:
            return {
                "portalName": "MCD 311 & Delhi PWD Grievance Portal",
                "department": "Public Works Department (PWD) / Municipal Corporation of Delhi",
                "portalUrl": "https://pgportal.gov.in/",
                "submissionType": "National Highway / Grievance",
                "countryName": country_name or "India",
                "regionLabel": f"{city or 'New Delhi'}, {state or 'Delhi'}",
                "instructions": "File via CPGRAMS (MoRTH / Urban Affairs) or MCD 311 Citizen App.",
                "helpline": "1908 / NHAI 1033",
            }
        if "hyderabad" in city_lower:
            return {
                "portalName": "MyGHMC Citizen Grievance & CPGRAMS",
                "department": "Greater Hyderabad Municipal Corporation — Engineering Wing",
                "portalUrl": "https://www.ghmc.gov.in/",
                "submissionType": "Municipal Public Works",
                "countryName": country_name or "India",
                "regionLabel": f"{city or 'Hyderabad'}, {state or 'Telangana'}",
                "instructions": "Log complaint under GHMC Road Damage / Pothole Repair with GPS coordinates.",
                "helpline": "040-21111111 / NHAI 1033",
            }
        region = ", ".join(p for p in [city, state] if p) or "India"
        return {
            "portalName": "CPGRAMS (Central Public Grievance Portal)",
            "department": "Ministry of Road Transport & Highways / Municipal Urban Development",
            "portalUrl": "https://pgportal.gov.in/",
            "submissionType": "National Highway / Grievance",
            "countryName": country_name or "India",
            "regionLabel": region,
            "instructions": "Submit under Ministry of Road Transport and Highways (Highways) or State Urban Local Body with pasted GPS coordinates and photo evidence.",
            "helpline": "NHAI Highway Helpline: 1033",
        }

    if cc == "us":
        if "new york" in city_lower or state_lower == "new york":
            return {
                "portalName": "NYC 311 Street Condition Portal",
                "department": "NYC Department of Transportation (DOT)",
                "portalUrl": "https://portal.311.nyc.gov/article/?kanumber=KA-01097",
                "submissionType": "311 Portal",
                "countryName": "United States",
                "regionLabel": f"{city or 'New York'}, NY",
                "instructions": 'Select "Pothole - Street" in NYC 311 and paste the auto-copied coordinates and hazard summary.',
                "helpline": "311",
            }
        if "san francisco" in city_lower:
            return {
                "portalName": "SF 311 Street & Sidewalk Repair",
                "department": "San Francisco Public Works (Open311)",
                "portalUrl": "https://sf311.org/services/pavement-and-roadway-maintenance",
                "submissionType": "Open311 Standard",
                "countryName": "United States",
                "regionLabel": "San Francisco, CA",
                "instructions": "Supports Open311 GeoReport v2 API or direct SF311 web submission.",
                "helpline": "311",
            }
        if "chicago" in city_lower:
            return {
                "portalName": "CHI 311 Pothole in Street",
                "department": "Chicago Department of Transportation (CDOT)",
                "portalUrl": "https://311.chicago.gov/",
                "submissionType": "311 Portal",
                "countryName": "United States",
                "regionLabel": "Chicago, IL",
                "instructions": 'Select "Pothole in Street" request type on CHI 311 and attach snapshot.',
                "helpline": "311",
            }
        if "seattle" in city_lower:
            return {
                "portalName": "Seattle Find It, Fix It (SDOT)",
                "department": "Seattle Department of Transportation",
                "portalUrl": "https://www.seattle.gov/transportation/projects-and-programs/programs/maintenance-and-paving/potholes",
                "submissionType": "311 Portal",
                "countryName": "United States",
                "regionLabel": "Seattle, WA",
                "instructions": "File pothole location directly with SDOT Customer Service Bureau.",
                "helpline": "(206) 684-ROAD",
            }
        region = ", ".join(p for p in [city, state] if p) or "United States"
        return {
            "portalName": "SeeClickFix / Municipal 311 Public Works",
            "department": f"{city + ' ' if city else ''}Department of Transportation & Public Works",
            "portalUrl": "https://seeclickfix.com/report",
            "submissionType": "311 Portal",
            "countryName": "United States",
            "regionLabel": region,
            "instructions": "Routes to your local US municipal 311 / DOT public works queue using GPS coordinates.",
            "helpline": "311",
        }

    if cc in ("gb", "uk"):
        region = ", ".join(p for p in [city, state] if p) or "United Kingdom"
        return {
            "portalName": "GOV.UK Report a Pothole / FixMyStreet",
            "department": f"{city + ' ' if city else ''}Local Council Highways Authority",
            "portalUrl": "https://www.gov.uk/report-pothole",
            "submissionType": "Municipal Public Works",
            "countryName": "United Kingdom",
            "regionLabel": region,
            "instructions": "Enter postcode on GOV.UK to jump directly to the responsible local council highways form.",
        }

    if cc == "ca":
        region = ", ".join(p for p in [city, state] if p) or "Canada"
        return {
            "portalName": "Municipal 311 Roadway Maintenance",
            "department": f"{city + ' ' if city else ''}Transportation & Public Works",
            "portalUrl": "https://www.toronto.ca/home/311-toronto-at-your-service/",
            "submissionType": "311 Portal",
            "countryName": "Canada",
            "regionLabel": region,
            "instructions": "Submit under Roadway Pothole Repair with exact latitude/longitude and photo evidence.",
            "helpline": "311",
        }

    if cc == "au":
        region = ", ".join(p for p in [city, state] if p) or "Australia"
        return {
            "portalName": "Snap Send Solve / Local Council Works",
            "department": f"{city + ' ' if city else ''}State Road & Council Infrastructure",
            "portalUrl": "https://www.snapsendsolve.com/",
            "submissionType": "Municipal Public Works",
            "countryName": "Australia",
            "regionLabel": region,
            "instructions": "Routes road hazard reports to the local Australian council road maintenance authority.",
        }

    region = ", ".join(p for p in [city, state, country_name] if p) or "Global Jurisdiction"
    return {
        "portalName": "Municipal Public Works & FixMyStreet Global",
        "department": f"{city + ' ' if city else ''}Department of Public Works & Roads",
        "portalUrl": "https://www.fixmystreet.com/",
        "submissionType": "Municipal Public Works",
        "countryName": country_name or "International",
        "regionLabel": region,
        "instructions": "Attach the exported official dossier or copy the GPS coordinates into your local municipality web portal.",
    }


def _record_govt_portal_submission(body: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    ticket_suffix = random.randint(100000, 999999)
    service_request_id = f"GOV-311-{now.year}-{ticket_suffix}"
    entry = {
        "service_request_id": service_request_id,
        "receivedAt": now.isoformat().replace("+00:00", "Z"),
        "portalName": body.get("portalName") or "Municipal 311 Portal",
        "department": body.get("department") or "Department of Transportation",
        "trackingCode": body.get("trackingCode") or "",
        "severity": body.get("severity") or "Moderate",
        "estimatedDepthCm": body.get("estimatedDepthCm"),
        "depthClass": body.get("depthClass") or "",
        "depthEstimationDetails": body.get("depthEstimationDetails"),
        "lat": body.get("lat"),
        "lng": body.get("lng"),
        "address": body.get("address"),
        "title": body.get("title"),
        "description": body.get("description"),
    }
    with _submissions_lock:
        existing: list[Any] = []
        try:
            if SUBMISSIONS_FILE.is_file():
                parsed = json.loads(SUBMISSIONS_FILE.read_text(encoding="utf-8"))
                if isinstance(parsed, list):
                    existing = parsed
        except Exception:
            existing = []
        SUBMISSIONS_FILE.write_text(
            json.dumps([entry, *existing], indent=2),
            encoding="utf-8",
        )
    return {
        "service_request_id": service_request_id,
        "status": "accepted",
        "portalName": entry["portalName"],
        "receivedAt": entry["receivedAt"],
    }


class RoadWatchAPIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep stdout clean; surface only errors if needed
        return

    def _send_json(self, status_code: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(
        self,
        file_path: Path,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        if length > 20 * 1024 * 1024:
            raise ValueError("Request payload exceeds 20 MB limit.")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        pathname = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if pathname == "/api/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "backend": "python",
                        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                        "timestamp": _iso_now(),
                    },
                )
                return

            if pathname == "/api/complaints":
                with _complaints_lock:
                    records = _read_complaints()
                self._send_json(200, {"complaints": records})
                return

            if pathname == "/api/geocode":
                self._handle_geocode(query)
                return

            if pathname == "/api/ml/model":
                if TRAINED_MODEL_FILE.is_file():
                    try:
                        model_data = json.loads(
                            TRAINED_MODEL_FILE.read_text(encoding="utf-8")
                        )
                        self._send_json(200, {"model": model_data})
                        return
                    except Exception:
                        pass
                self._send_json(200, {"model": None})
                return

            if pathname == "/api/datasets/kaggle":
                self._handle_kaggle_list(query)
                return

            if pathname == "/api/datasets/kaggle/export.csv":
                if not KAGGLE_CSV_FILE.is_file():
                    self._send_json(
                        404, {"error": "Kaggle CSV export not generated yet."}
                    )
                    return
                self._send_file(
                    KAGGLE_CSV_FILE,
                    "text/csv; charset=utf-8",
                    {
                        "Content-Disposition": 'attachment; filename="andrewmvd-pothole-detection.csv"'
                    },
                )
                return

            m_img = re.fullmatch(r"/api/datasets/kaggle/image/([^/]+)", pathname)
            if m_img:
                filename = urllib.parse.unquote(m_img.group(1))
                self._handle_kaggle_image(filename)
                return

            self._send_json(404, {"error": f"Unknown API endpoint: {pathname}"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        pathname = parsed.path.rstrip("/") or "/"

        try:
            if pathname == "/api/detect/roboflow":
                self._handle_detect_roboflow()
                return

            if pathname == "/api/complaints":
                self._handle_create_complaint()
                return

            if pathname == "/api/portals/test":
                self._handle_test_portal()
                return

            if pathname == "/api/govt-portal/receive":
                body = self._read_json_body()
                receipt = _record_govt_portal_submission(body)
                self._send_json(201, receipt)
                return

            if pathname == "/api/datasets/kaggle/import":
                self._handle_kaggle_import()
                return

            self._send_json(404, {"error": f"Unknown API endpoint: {pathname}"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_PATCH(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        pathname = parsed.path.rstrip("/") or "/"

        m_cmp = re.fullmatch(r"/api/complaints/([^/]+)", pathname)
        if not m_cmp:
            self._send_json(404, {"error": f"Unknown API endpoint: {pathname}"})
            return

        try:
            comp_id = urllib.parse.unquote(m_cmp.group(1))
            body = self._read_json_body()
            with _complaints_lock:
                records = _read_complaints()
                idx = next(
                    (i for i, r in enumerate(records) if r.get("id") == comp_id),
                    -1,
                )
                if idx == -1:
                    self._send_json(404, {"error": "Complaint not found."})
                    return

                updated = dict(records[idx])
                if body.get("status"):
                    updated["status"] = body["status"]
                if "externalTicketId" in body:
                    updated["externalTicketId"] = body["externalTicketId"]
                if body.get("portalName"):
                    updated["portalName"] = body["portalName"]
                if body.get("portalUrl"):
                    updated["portalUrl"] = body["portalUrl"]
                updated["updatedAt"] = _iso_now()
                records[idx] = updated
                _write_complaints(records)

            self._send_json(200, {"complaint": updated})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        pathname = parsed.path.rstrip("/") or "/"

        m_cmp = re.fullmatch(r"/api/complaints/([^/]+)", pathname)
        if not m_cmp:
            self._send_json(404, {"error": f"Unknown API endpoint: {pathname}"})
            return

        try:
            comp_id = urllib.parse.unquote(m_cmp.group(1))
            with _complaints_lock:
                records = _read_complaints()
                filtered = [r for r in records if r.get("id") != comp_id]
                if len(filtered) == len(records):
                    self._send_json(404, {"error": "Complaint not found."})
                    return
                _write_complaints(filtered)

            self._send_json(200, {"ok": True})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    # --- Route Handlers ---

    def _handle_detect_roboflow(self) -> None:
        api_key = (os.environ.get("ROBOFLOW_API_KEY") or "").strip()
        if not api_key:
            self._send_json(
                503, {"error": "ROBOFLOW_API_KEY is not configured on the server."}
            )
            return

        body = self._read_json_body()
        raw_base64 = ""
        if isinstance(body.get("imageBase64"), str) and body["imageBase64"].strip():
            raw_base64 = body["imageBase64"].strip()
        elif isinstance(body.get("imageDataUrl"), str) and body["imageDataUrl"].strip():
            data_url = body["imageDataUrl"].strip()
            comma_idx = data_url.find(",")
            raw_base64 = (
                data_url[comma_idx + 1 :].strip() if comma_idx >= 0 else data_url
            )

        if not raw_base64:
            self._send_json(
                400, {"error": "imageDataUrl or imageBase64 is required."}
            )
            return

        conf = _safe_float(body.get("confidence"), 0.28)
        overlap = _safe_float(body.get("overlap"), 0.45)

        try:
            model = _get_thread_model()
            result = model.predict_base64(
                raw_base64, confidence=conf, overlap=overlap
            )
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(
                502, {"error": str(exc) or "Python pothole model inference failed."}
            )

    def _handle_geocode(self, query: dict[str, list[str]]) -> None:
        lat_raw = (query.get("lat") or [""])[0]
        lng_raw = (query.get("lng") or [""])[0]
        try:
            lat = float(lat_raw)
            lng = float(lng_raw)
            if not (math.isfinite(lat) and math.isfinite(lng)):
                raise ValueError("Non-finite coordinates")
        except ValueError:
            self._send_json(
                400, {"error": "Valid lat and lng query parameters are required."}
            )
            return

        cache_key = f"{lat:.4f},{lng:.4f}"
        with _geocode_lock:
            cached = _geocode_cache.get(cache_key)
        if cached is not None:
            self._send_json(200, cached)
            return

        try:
            url = (
                "https://nominatim.openstreetmap.org/reverse?format=jsonv2"
                f"&lat={urllib.parse.quote(str(lat))}"
                f"&lon={urllib.parse.quote(str(lng))}"
                "&zoom=18&addressdetails=1"
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "RoadWatch-Civic-Portal/1.0 (civic-road-inspection)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(
                        f"Geocoding service responded with status {resp.status}"
                    )
                data = json.loads(resp.read().decode("utf-8"))

            addr = data.get("address") or {}
            street = " ".join(
                p
                for p in [
                    addr.get("house_number"),
                    addr.get("road") or addr.get("pedestrian") or addr.get("highway"),
                ]
                if p
            )
            neighborhood = (
                addr.get("suburb")
                or addr.get("neighbourhood")
                or addr.get("quarter")
                or ""
            )
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
                or ""
            )
            state = addr.get("state") or addr.get("region") or ""
            postcode = addr.get("postcode") or ""
            country = addr.get("country") or ""
            country_code = str(addr.get("country_code") or "").upper()

            formatted_address = (
                ", ".join(
                    p for p in [street, neighborhood, city, state, postcode] if p
                )
                or data.get("display_name")
                or f"{lat:.5f}, {lng:.5f}"
            )

            result = {
                "lat": lat,
                "lng": lng,
                "formattedAddress": formatted_address,
                "street": street or data.get("name") or "Unnamed Road Segment",
                "neighborhood": neighborhood,
                "city": city,
                "state": state,
                "postcode": postcode,
                "country": country,
                "countryCode": country_code,
                "portal": resolve_government_portal(country_code, city, state, country),
            }
            with _geocode_lock:
                _geocode_cache[cache_key] = result
            self._send_json(200, result)
        except Exception as exc:
            fallback = {
                "lat": lat,
                "lng": lng,
                "formattedAddress": f"GPS {lat:.5f}, {lng:.5f}",
                "street": f"Roadway at {lat:.5f}, {lng:.5f}",
                "neighborhood": "",
                "city": "",
                "state": "",
                "postcode": "",
                "country": "",
                "countryCode": "",
                "portal": resolve_government_portal("", "", "", ""),
                "warning": str(exc) or "Reverse geocoding temporarily unavailable",
            }
            self._send_json(200, fallback)

    def _handle_create_complaint(self) -> None:
        body = self._read_json_body()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat().replace("+00:00", "Z")

        incoming_lat = _safe_float(body.get("lat"), 0.0)
        incoming_lng = _safe_float(body.get("lng"), 0.0)
        incoming_detections = (
            body.get("detections") if isinstance(body.get("detections"), list) else []
        )
        incoming_count = max(
            1,
            int(_safe_float(body.get("clusterCount"), 0))
            or len(incoming_detections)
            or 1,
        )
        first_det = incoming_detections[0] if incoming_detections else None
        if body.get("estimatedDepthCm") is not None:
            resolved_depth_cm: float | None = _safe_float(body.get("estimatedDepthCm"))
        elif isinstance(first_det, dict) and first_det.get("estimatedDepthCm") is not None:
            resolved_depth_cm = _safe_float(first_det.get("estimatedDepthCm"))
        else:
            resolved_depth_cm = None

        # 1. Automatic 5-meter spatial clustering into existing open complaints
        with _complaints_lock:
            existing_records = _read_complaints()
            match_idx = -1
            closest_dist = float("inf")

            if incoming_lat != 0.0 or incoming_lng != 0.0:
                for i, rec in enumerate(existing_records):
                    if not isinstance(rec, dict) or rec.get("status") == "Resolved":
                        continue
                    rec_lat = _safe_float(rec.get("lat"), float("nan"))
                    rec_lng = _safe_float(rec.get("lng"), float("nan"))
                    if not (math.isfinite(rec_lat) and math.isfinite(rec_lng)):
                        continue
                    dist = _haversine_meters(rec_lat, rec_lng, incoming_lat, incoming_lng)
                    if dist <= COMPLAINT_CLUSTER_RADIUS_METERS and dist < closest_dist:
                        closest_dist = dist
                        match_idx = i

            if match_idx != -1:
                target = existing_records[match_idx]
                prev_dets = (
                    target.get("detections")
                    if isinstance(target.get("detections"), list)
                    else []
                )
                prev_count = max(
                    1,
                    int(_safe_float(target.get("clusterCount"), 0))
                    or len(prev_dets)
                    or 1,
                )
                next_cluster_count = prev_count + incoming_count
                merged_dets = list(prev_dets)
                for idx_d, det in enumerate(incoming_detections):
                    det_copy = dict(det) if isinstance(det, dict) else {}
                    sev_lbl = det_copy.get("severity") or body.get("severity") or "Moderate"
                    det_copy["label"] = (
                        f"Cluster Pothole #{len(prev_dets) + idx_d + 1} ({sev_lbl})"
                    )
                    merged_dets.append(det_copy)

                prev_sev_rank = SEVERITY_RANK.get(str(target.get("severity") or ""), 0)
                new_sev_rank = SEVERITY_RANK.get(str(body.get("severity") or ""), 0)
                merged_severity = (
                    body.get("severity")
                    if new_sev_rank > prev_sev_rank
                    else target.get("severity") or "Moderate"
                )

                prev_depth = (
                    _safe_float(target.get("estimatedDepthCm"))
                    if target.get("estimatedDepthCm") is not None
                    else None
                )
                use_incoming_depth = resolved_depth_cm is not None and (
                    prev_depth is None or resolved_depth_cm > prev_depth
                )
                merged_depth_cm = resolved_depth_cm if use_incoming_depth else prev_depth
                merged_depth_class = (
                    body.get("depthClass")
                    or (first_det.get("depthClass") if isinstance(first_det, dict) else "")
                    or target.get("depthClass")
                    or ""
                    if use_incoming_depth
                    else target.get("depthClass") or body.get("depthClass") or ""
                )
                merged_depth_details = (
                    body.get("depthEstimationDetails")
                    or (
                        first_det.get("depthEstimationDetails")
                        if isinstance(first_det, dict)
                        else None
                    )
                    or target.get("depthEstimationDetails")
                    if use_incoming_depth
                    else target.get("depthEstimationDetails")
                    or body.get("depthEstimationDetails")
                )

                max_width_cm = max(
                    (_safe_float(d.get("estimatedWidthCm"), 0.0) for d in merged_dets if isinstance(d, dict)),
                    default=0.0,
                )
                total_area_sq_cm = sum(
                    _safe_float(d.get("estimatedAreaSqCm"), 0.0)
                    for d in merged_dets
                    if isinstance(d, dict)
                )

                weighted_lat = (
                    _safe_float(target.get("lat")) * prev_count
                    + incoming_lat * incoming_count
                ) / next_cluster_count
                weighted_lng = (
                    _safe_float(target.get("lng")) * prev_count
                    + incoming_lng * incoming_count
                ) / next_cluster_count

                width_txt = int(round(max_width_cm)) if max_width_cm > 0 else "—"
                depth_txt = merged_depth_cm if merged_depth_cm is not None else "—"
                depth_cls_txt = f" ({merged_depth_class})" if merged_depth_class else ""
                area_txt = (
                    f", combined damaged area ~{int(round(total_area_sq_cm))} cm²"
                    if total_area_sq_cm > 0
                    else ""
                )
                radius_int = int(COMPLAINT_CLUSTER_RADIUS_METERS)
                plural_s = "s" if incoming_count > 1 else ""

                merged_record = {
                    **target,
                    "updatedAt": now_iso,
                    "title": f"Cluster Report ({next_cluster_count} Potholes within {radius_int}m) — {merged_severity} Road Defect",
                    "description": (
                        f"Consolidated cluster complaint grouping {next_cluster_count} detected potholes "
                        f"within a {radius_int}-meter radius to prevent duplicate reports. "
                        f"Worst hazard measures approx. {width_txt} cm wide × {depth_txt} cm deep"
                        f"{depth_cls_txt}{area_txt}."
                    ),
                    "severity": merged_severity,
                    "confidence": max(
                        _safe_float(target.get("confidence"), 0.0),
                        _safe_float(body.get("confidence"), 0.0),
                    ),
                    "estimatedDepthCm": merged_depth_cm,
                    "depthClass": merged_depth_class,
                    "depthEstimationDetails": merged_depth_details,
                    "lat": weighted_lat,
                    "lng": weighted_lng,
                    "imageDataUrl": (
                        body.get("imageDataUrl")
                        if use_incoming_depth and body.get("imageDataUrl")
                        else target.get("imageDataUrl") or body.get("imageDataUrl") or ""
                    ),
                    "detections": merged_dets,
                    "isClusterReport": True,
                    "clusterCount": next_cluster_count,
                    "clusterRadiusMeters": radius_int,
                    "upstreamResponseNote": (
                        f"{target.get('upstreamResponseNote') or 'Cluster report active.'} "
                        f"Merged +{incoming_count} nearby detection{plural_s} "
                        f"({closest_dist:.1f}m away) into this cluster report."
                    ),
                }
                existing_records[match_idx] = merged_record
                _write_complaints(existing_records)
                self._send_json(
                    200,
                    {
                        "complaint": merged_record,
                        "clustered": True,
                        "mergedIntoExisting": True,
                    },
                )
                return

        # 2. New complaint + optional Open311 / Civic 311 gateway auto-dispatch
        random_num = random.randint(1000, 9999)
        comp_id = f"cmp_{int(now.timestamp() * 1000)}_{random_num}"
        tracking_code = f"RW-{now.year}-{random_num}"

        status = "Queued for Govt Portal"
        dispatch_channel = body.get("portalName") or "Municipal Portal Package"
        upstream_http_status: int | None = None
        upstream_response_note = (
            "Prepared official geotagged dossier for municipal submission."
        )
        external_ticket_id = ""

        open311 = body.get("open311Config") if isinstance(body.get("open311Config"), dict) else None
        should_use_custom = bool(
            open311 and open311.get("enabled") and open311.get("endpointUrl")
        )
        should_auto_gateway = (not should_use_custom) and bool(body.get("autoRaise"))

        if should_use_custom:
            endpoint = str(open311["endpointUrl"]).strip()  # type: ignore[index]
            try:
                if open311.get("mode") == "open311":  # type: ignore[union-attr]
                    form_fields: dict[str, str] = {
                        "service_code": str(open311.get("serviceCode") or "pothole"),  # type: ignore[union-attr]
                        "lat": str(incoming_lat),
                        "long": str(incoming_lng),
                        "address_string": str(body.get("address") or ""),
                        "description": f"{body.get('title') or 'Pothole Hazard'} — {body.get('description') or ''}",
                    }
                    if open311.get("apiKey"):  # type: ignore[union-attr]
                        form_fields["api_key"] = str(open311["apiKey"])  # type: ignore[index]
                    if open311.get("jurisdictionId"):  # type: ignore[union-attr]
                        form_fields["jurisdiction_id"] = str(open311["jurisdictionId"])  # type: ignore[index]
                    if open311.get("reporterEmail"):  # type: ignore[union-attr]
                        form_fields["email"] = str(open311["reporterEmail"])  # type: ignore[index]
                    if open311.get("reporterName"):  # type: ignore[union-attr]
                        parts = str(open311["reporterName"]).strip().split()  # type: ignore[index]
                        form_fields["first_name"] = parts[0] if parts else "Citizen"
                        if len(parts) > 1:
                            form_fields["last_name"] = " ".join(parts[1:])

                    req_data = urllib.parse.urlencode(form_fields).encode("utf-8")
                    req_headers = {"Content-Type": "application/x-www-form-urlencoded"}
                else:
                    webhook_payload = {
                        "trackingCode": tracking_code,
                        "title": body.get("title"),
                        "description": body.get("description"),
                        "severity": body.get("severity"),
                        "estimatedDepthCm": resolved_depth_cm,
                        "depthClass": body.get("depthClass") or "",
                        "depthEstimationDetails": body.get("depthEstimationDetails"),
                        "lat": incoming_lat,
                        "lng": incoming_lng,
                        "address": body.get("address"),
                        "city": body.get("city"),
                        "state": body.get("state"),
                        "countryCode": body.get("countryCode"),
                        "department": body.get("department"),
                        "portalName": body.get("portalName"),
                        "portalUrl": body.get("portalUrl"),
                        "detectedAt": now_iso,
                        "detections": incoming_detections,
                        "isClusterReport": bool(body.get("isClusterReport") or incoming_count > 1),
                        "clusterCount": incoming_count,
                        "clusterRadiusMeters": int(COMPLAINT_CLUSTER_RADIUS_METERS),
                    }
                    req_data = json.dumps(webhook_payload).encode("utf-8")
                    req_headers = {"Content-Type": "application/json"}
                    if open311.get("apiKey"):  # type: ignore[union-attr]
                        req_headers["Authorization"] = f"Bearer {open311['apiKey']}"  # type: ignore[index]

                req = urllib.request.Request(
                    endpoint, data=req_data, headers=req_headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    upstream_http_status = resp.status
                    resp_text = resp.read().decode("utf-8", errors="replace")
                    status = "Dispatched via API"
                    dispatch_channel = (
                        "Open311 GeoReport v2 API"
                        if open311.get("mode") == "open311"  # type: ignore[union-attr]
                        else "Municipal Webhook API"
                    )
                    try:
                        parsed_resp = json.loads(resp_text)
                        first = parsed_resp[0] if isinstance(parsed_resp, list) and parsed_resp else parsed_resp
                        if isinstance(first, dict):
                            external_ticket_id = str(
                                first.get("service_request_id")
                                or first.get("ticketId")
                                or first.get("id")
                                or ""
                            )
                    except Exception:
                        pass
                    upstream_response_note = (
                        f"Automatically delivered to {endpoint} (HTTP {resp.status})."
                    )
            except urllib.error.HTTPError as http_err:
                upstream_http_status = http_err.code
                err_body = http_err.read().decode("utf-8", errors="replace")[:180]
                upstream_response_note = (
                    f"Endpoint returned HTTP {http_err.code}: {err_body}"
                )
            except Exception as dispatch_err:
                upstream_response_note = (
                    f"API dispatch error: {dispatch_err}. "
                    "Complaint saved locally for portal submission."
                )
        elif should_auto_gateway:
            # Direct in-process dispatch to the Python Civic 311 gateway receiver
            receipt = _record_govt_portal_submission(
                {
                    **body,
                    "trackingCode": tracking_code,
                    "estimatedDepthCm": resolved_depth_cm,
                    "lat": incoming_lat,
                    "lng": incoming_lng,
                }
            )
            upstream_http_status = 201
            status = "Submitted to Govt Site"
            dispatch_channel = f"{body.get('portalName') or 'Municipal 311'} (Automated Gateway)"
            external_ticket_id = str(receipt.get("service_request_id") or "")
            upstream_response_note = (
                f"Automatically raised to {body.get('portalName') or 'Civic 311 Portal'} "
                f"gateway ({external_ticket_id or 'Confirmed'})."
            )

        is_cluster = bool(body.get("isClusterReport") or incoming_count > 1)
        record = {
            "id": comp_id,
            "trackingCode": tracking_code,
            "createdAt": now_iso,
            "updatedAt": now_iso,
            "title": body.get("title") or "Roadway Pothole Hazard",
            "description": body.get("description") or "",
            "severity": body.get("severity") or "Moderate",
            "confidence": _safe_float(body.get("confidence"), 0.0),
            "roadSurfaceType": body.get("roadSurfaceType") or "Asphalt",
            "estimatedDepthCm": resolved_depth_cm,
            "depthClass": body.get("depthClass")
            or (first_det.get("depthClass") if isinstance(first_det, dict) else "")
            or "",
            "depthEstimationDetails": body.get("depthEstimationDetails")
            or (
                first_det.get("depthEstimationDetails")
                if isinstance(first_det, dict)
                else None
            ),
            "lat": incoming_lat,
            "lng": incoming_lng,
            "accuracyMeters": (
                _safe_float(body.get("accuracyMeters"))
                if body.get("accuracyMeters") is not None
                else None
            ),
            "address": body.get("address") or f"{incoming_lat}, {incoming_lng}",
            "city": body.get("city") or "",
            "state": body.get("state") or "",
            "countryCode": body.get("countryCode") or "",
            "department": body.get("department")
            or "Department of Transportation & Public Works",
            "portalName": body.get("portalName") or "Municipal Public Works Portal",
            "portalUrl": body.get("portalUrl") or "https://www.fixmystreet.com/",
            "status": status,
            "externalTicketId": external_ticket_id,
            "dispatchChannel": dispatch_channel,
            "upstreamHttpStatus": upstream_http_status,
            "upstreamResponseNote": upstream_response_note,
            "autoFiled": bool(body.get("autoFiled")),
            "imageDataUrl": body.get("imageDataUrl") or "",
            "detections": incoming_detections,
            "isClusterReport": is_cluster,
            "clusterCount": incoming_count,
            "clusterRadiusMeters": int(COMPLAINT_CLUSTER_RADIUS_METERS),
        }

        with _complaints_lock:
            latest = _read_complaints()
            _write_complaints([record, *latest])

        self._send_json(
            201,
            {
                "complaint": record,
                "clustered": is_cluster,
                "mergedIntoExisting": False,
            },
        )

    def _handle_test_portal(self) -> None:
        body = self._read_json_body()
        endpoint_url = body.get("endpointUrl")
        if not endpoint_url or not isinstance(endpoint_url, str):
            self._send_json(400, {"error": "Endpoint URL is required."})
            return

        try:
            req = urllib.request.Request(endpoint_url.strip(), method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._send_json(
                    200,
                    {
                        "reachable": True,
                        "status": resp.status,
                        "statusText": getattr(resp, "reason", "OK"),
                    },
                )
        except urllib.error.HTTPError as http_err:
            self._send_json(
                200,
                {
                    "reachable": True,
                    "status": http_err.code,
                    "statusText": http_err.reason,
                },
            )
        except Exception as exc:
            self._send_json(
                502,
                {
                    "reachable": False,
                    "error": str(exc) or "Failed to reach endpoint.",
                },
            )

    def _handle_kaggle_list(self, query: dict[str, list[str]]) -> None:
        if not KAGGLE_JSON_FILE.is_file():
            self._send_json(
                404,
                {
                    "error": "Kaggle dataset not imported yet. Trigger POST /api/datasets/kaggle/import."
                },
            )
            return

        catalog = json.loads(KAGGLE_JSON_FILE.read_text(encoding="utf-8"))
        all_records = catalog.get("records") if isinstance(catalog.get("records"), list) else []
        search = (query.get("search") or [""])[0].strip().lower()
        min_boxes = int(_safe_float((query.get("minBoxes") or ["0"])[0], 0))
        page = max(1, int(_safe_float((query.get("page") or ["1"])[0], 1)))
        limit = max(1, min(100, int(_safe_float((query.get("limit") or ["24"])[0], 24))))

        cached_set = (
            set(os.listdir(KAGGLE_IMAGES_DIR)) if KAGGLE_IMAGES_DIR.is_dir() else set()
        )

        filtered = all_records
        if search:
            filtered = [
                r
                for r in filtered
                if search in str(r.get("id") or "").lower()
                or search in str(r.get("filename") or "").lower()
            ]
        if min_boxes > 0:
            filtered = [
                r for r in filtered if int(_safe_float(r.get("boxCount"), 0)) >= min_boxes
            ]

        start_idx = (page - 1) * limit
        page_records = [
            {**r, "cachedLocally": r.get("filename") in cached_set}
            for r in filtered[start_idx : start_idx + limit]
        ]

        self._send_json(
            200,
            {
                "dataset": catalog.get("dataset") or "andrewmvd/pothole-detection",
                "sourceUrl": catalog.get("sourceUrl")
                or "https://www.kaggle.com/datasets/andrewmvd/pothole-detection",
                "format": catalog.get("format") or "Pascal VOC XML + PNG Images",
                "importedAt": catalog.get("importedAt") or "",
                "totalImages": catalog.get("totalImages") or len(all_records),
                "totalBoxes": catalog.get("totalBoxes") or 0,
                "avgBoxesPerImage": catalog.get("avgBoxesPerImage") or 0,
                "cachedImageCount": len(cached_set),
                "page": page,
                "limit": limit,
                "totalFiltered": len(filtered),
                "records": page_records,
            },
        )

    def _handle_kaggle_image(self, filename: str) -> None:
        safe_name = os.path.basename(filename)
        if not re.fullmatch(r"[a-zA-Z0-9._-]+\.(png|jpg|jpeg)", safe_name, re.I):
            self._send_json(400, {"error": "Invalid image filename."})
            return

        local_path = KAGGLE_IMAGES_DIR / safe_name
        content_type = "image/png" if safe_name.lower().endswith(".png") else "image/jpeg"

        if local_path.is_file() and local_path.stat().st_size > 0:
            self._send_file(
                local_path, content_type, {"Cache-Control": "public, max-age=86400"}
            )
            return

        with _kaggle_locks_guard:
            img_lock = _kaggle_image_locks.setdefault(safe_name, threading.Lock())

        try:
            with img_lock:
                if not (local_path.is_file() and local_path.stat().st_size > 0):
                    subprocess.run(
                        [
                            sys.executable,
                            str(KAGGLE_IMPORTER_SCRIPT),
                            "--fetch-image",
                            safe_name,
                        ],
                        check=True,
                        timeout=45,
                        capture_output=True,
                    )
            if local_path.is_file():
                self._send_file(
                    local_path, content_type, {"Cache-Control": "public, max-age=86400"}
                )
                return
            self._send_json(404, {"error": "Image not found in Kaggle dataset."})
        except Exception as exc:
            self._send_json(
                502, {"error": str(exc) or "Failed to stream image from Kaggle dataset."}
            )

    def _handle_kaggle_import(self) -> None:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(KAGGLE_IMPORTER_SCRIPT),
                    "--precache",
                    "12",
                ],
                check=True,
                timeout=120,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            self._send_json(200, {"ok": True, "summary": summary})
        except Exception as exc:
            self._send_json(
                500, {"error": str(exc) or "Failed to import Kaggle dataset."}
            )


def main() -> None:
    server = ThreadingHTTPServer((BACKEND_HOST, BACKEND_PORT), RoadWatchAPIHandler)
    server.daemon_threads = True
    print(
        f"RoadWatch Python backend running at http://{BACKEND_HOST}:{BACKEND_PORT} "
        f"(Python {sys.version.split()[0]})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
