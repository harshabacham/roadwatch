#!/usr/bin/env python3
"""
RoadWatch Python Pothole Detection Model (Python 3.11).

Runs pothole inference using Roboflow Serverless Inference (via `inference_sdk`
with header-based API key auth when installed, or a persistent keep-alive Python
HTTPS client using the identical header-based auth protocol) combined with
Python-side NMS filtering and perspective-scaled physical geometry/severity
post-processing.

Modes:
  1. Persistent NDJSON stdio worker (used by Express backend for sub-ms IPC):
       python3 -u ml/pothole_model.py --worker
  2. Standalone CLI test on a local image:
       python3 ml/pothole_model.py path/to/road.jpg
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.parse
from typing import Any

# Optional inference_sdk support (header-based auth, inference v1.5.0+)
try:
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient  # type: ignore

    HAS_INFERENCE_SDK = True
except Exception:
    HAS_INFERENCE_SDK = False


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1 = a["x"] - a["width"] / 2.0, a["y"] - a["height"] / 2.0
    ax2, ay2 = a["x"] + a["width"] / 2.0, a["y"] + a["height"] / 2.0
    bx1, by1 = b["x"] - b["width"] / 2.0, b["y"] - b["height"] / 2.0
    bx2, by2 = b["x"] + b["width"] / 2.0, b["y"] + b["height"] / 2.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms_predictions(
    predictions: list[dict[str, Any]],
    confidence_threshold: float,
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    filtered = [
        p
        for p in predictions
        if float(p.get("confidence", 0.0)) >= confidence_threshold
    ]
    filtered.sort(key=lambda p: float(p.get("confidence", 0.0)), reverse=True)

    kept: list[dict[str, Any]] = []
    for pred in filtered:
        box = {
            "x": float(pred.get("x", 0.0)),
            "y": float(pred.get("y", 0.0)),
            "width": float(pred.get("width", 0.0)),
            "height": float(pred.get("height", 0.0)),
        }
        if any(_box_iou(box, k) >= overlap_threshold for k in kept):
            continue
        kept.append(pred)
    return kept


def _enrich_prediction(
    pred: dict[str, Any], img_w: float, img_h: float
) -> dict[str, Any]:
    w = max(1.0, img_w)
    h = max(1.0, img_h)
    pw = float(pred.get("width", 0.0))
    ph = float(pred.get("height", 0.0))
    px = float(pred.get("x", 0.0))
    py = float(pred.get("y", 0.0))
    conf = float(pred.get("confidence", 0.0))

    norm_w = max(0.01, pw / w)
    norm_h = max(0.01, ph / h)
    area_frac = norm_w * norm_h
    center_norm_y = _clamp(py / h, 0.0, 1.0)
    perspective = 0.7 + (1.0 - center_norm_y) * 1.1

    est_width_cm = round(_clamp(norm_w * 145.0 * perspective, 18.0, 180.0))
    est_length_cm = round(_clamp(norm_h * 125.0 * perspective, 15.0, 160.0))
    est_area_sq_cm = round(est_width_cm * est_length_cm * 0.75)
    est_depth_cm = round(_clamp(4.0 + area_frac * 45.0, 3.0, 20.0), 1)

    severity_idx = area_frac * 18.0 + est_depth_cm * 0.1 + conf * 0.8
    if est_depth_cm >= 16.0 or est_width_cm >= 95 or severity_idx >= 3.6:
        severity = "Critical"
    elif est_depth_cm >= 11.0 or est_width_cm >= 55 or severity_idx >= 2.5:
        severity = "Severe"
    elif est_depth_cm >= 6.0 or est_width_cm >= 30 or severity_idx >= 1.7:
        severity = "Moderate"
    else:
        severity = "Low"

    enriched = dict(pred)
    enriched["pythonPostProcess"] = {
        "estimatedWidthCm": est_width_cm,
        "estimatedLengthCm": est_length_cm,
        "estimatedAreaSqCm": est_area_sq_cm,
        "estimatedDepthCm": est_depth_cm,
        "severity": severity,
    }
    return enriched


class PotholeDetectionModel:
    """Persistent Python pothole inference model backed by Roboflow Serverless API."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.api_url = (
            api_url
            or os.environ.get("ROBOFLOW_API_URL")
            or "https://serverless.roboflow.com"
        ).rstrip("/")
        self.api_key = (
            api_key or os.environ.get("ROBOFLOW_API_KEY") or ""
        ).strip()
        self.model_id = (
            model_id
            or os.environ.get("ROBOFLOW_MODEL_ID")
            or "pothole-detection-bqu6s-dwjbo/1"
        ).lstrip("/")

        parsed = urllib.parse.urlparse(self.api_url)
        self._scheme = parsed.scheme or "https"
        self._host = parsed.hostname or "serverless.roboflow.com"
        self._port = parsed.port or (443 if self._scheme == "https" else 80)
        self._base_path = parsed.path.rstrip("/")
        self._ssl_ctx = ssl.create_default_context()
        self._conn: http.client.HTTPConnection | None = None

        self._sdk_client: Any = None
        if HAS_INFERENCE_SDK and self.api_key:
            self._sdk_client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key,
            ).configure(
                InferenceConfiguration(api_key_transport="header")
            )

    def _get_connection(self) -> http.client.HTTPConnection:
        if self._conn is not None:
            return self._conn
        if self._scheme == "https":
            self._conn = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=15,
                context=self._ssl_ctx,
            )
        else:
            self._conn = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=15,
            )
        return self._conn

    def _reset_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _infer_via_http(
        self,
        raw_base64: str,
        confidence: float,
        overlap: float,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {
                "confidence": f"{confidence:.2f}",
                "overlap": f"{overlap:.2f}",
            }
        )
        req_path = f"{self._base_path}/{self.model_id}?{query}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Connection": "keep-alive",
        }
        body_bytes = raw_base64.encode("ascii")

        # Retry once if keep-alive socket was closed by upstream
        last_err: Exception | None = None
        for attempt in range(2):
            conn = self._get_connection()
            try:
                conn.request("POST", req_path, body=body_bytes, headers=headers)
                resp = conn.getresponse()
                resp_body = resp.read().decode("utf-8", errors="replace")
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(
                        f"Roboflow inference failed (HTTP {resp.status}): {resp_body[:300]}"
                    )
                return json.loads(resp_body)
            except (
                http.client.RemoteDisconnected,
                http.client.CannotSendRequest,
                http.client.BadStatusLine,
                BrokenPipeError,
                ConnectionResetError,
            ) as exc:
                last_err = exc
                self._reset_connection()
                if attempt == 1:
                    raise
            except Exception:
                self._reset_connection()
                raise

        raise RuntimeError(str(last_err or "Upstream connection failed"))

    def _infer_via_sdk(
        self,
        raw_base64: str,
        confidence: float,
        overlap: float,
    ) -> dict[str, Any]:
        # inference_sdk accepts a file path or base64 image; temp file is universally supported
        img_bytes = base64.b64decode(raw_base64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            cfg = InferenceConfiguration(
                api_key_transport="header",
                confidence_threshold=confidence,
                iou_threshold=overlap,
            )
            with self._sdk_client.use_configuration(cfg):
                return self._sdk_client.infer(tmp_path, model_id=self.model_id)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def predict_base64(
        self,
        raw_base64: str,
        confidence: float = 0.28,
        overlap: float = 0.45,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not configured on the server.")
        if not raw_base64:
            raise ValueError("imageBase64 is required.")

        conf = _clamp(float(confidence), 0.05, 0.99)
        ov = _clamp(float(overlap), 0.05, 0.99)

        t0 = time.perf_counter()
        if self._sdk_client is not None:
            data = self._infer_via_sdk(raw_base64, conf, ov)
            runtime_mode = "python-inference-sdk"
        else:
            data = self._infer_via_http(raw_base64, conf, ov)
            runtime_mode = "python-stdlib-keepalive"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        image_info = data.get("image") or {"width": 0, "height": 0}
        img_w = float(image_info.get("width") or 0)
        img_h = float(image_info.get("height") or 0)

        raw_preds = data.get("predictions")
        preds_list = raw_preds if isinstance(raw_preds, list) else []
        cleaned = _nms_predictions(preds_list, conf, ov)
        enriched = [_enrich_prediction(p, img_w, img_h) for p in cleaned]

        upstream_time = data.get("time")
        reported_ms = (
            float(upstream_time) * 1000.0
            if isinstance(upstream_time, (int, float))
            else elapsed_ms
        )

        resolved_model = (
            (data.get("resolved_model") or {}).get("model_id")
            if isinstance(data.get("resolved_model"), dict)
            else None
        )

        return {
            "inferenceId": str(data.get("inference_id") or ""),
            "inferenceMs": round(reported_ms, 1),
            "modelId": resolved_model or self.model_id,
            "engineRuntime": runtime_mode,
            "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "image": image_info,
            "predictions": enriched,
        }


def run_worker_loop() -> None:
    model = PotholeDetectionModel()
    # Signal readiness to parent Node process
    sys.stdout.write(json.dumps({"ready": True, "python": sys.version.split()[0]}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            if payload.get("cmd") == "ping":
                resp = {"id": req_id, "ok": True, "result": {"pong": True}}
            else:
                raw_b64 = str(payload.get("imageBase64") or "").strip()
                conf = float(payload.get("confidence", 0.28))
                ov = float(payload.get("overlap", 0.45))
                result = model.predict_base64(raw_b64, confidence=conf, overlap=ov)
                resp = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:
            resp = {"id": req_id, "ok": False, "error": str(exc)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        run_worker_loop()
        return

    if len(sys.argv) >= 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        model = PotholeDetectionModel()
        out = model.predict_base64(b64)
        print(json.dumps(out, indent=2))
        return

    print(
        json.dumps(
            {
                "usage": "python3 ml/pothole_model.py [--worker | <image_path>]",
                "hasInferenceSdk": HAS_INFERENCE_SDK,
                "python": sys.version.split()[0],
            }
        )
    )


if __name__ == "__main__":
    main()
