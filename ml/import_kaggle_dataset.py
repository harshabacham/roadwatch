#!/usr/bin/env python3
"""
Imports the Kaggle 'andrewmvd/pothole-detection' Pascal VOC dataset (665 images + 665 XML annotations)
using HTTP Range requests against Kaggle's public dataset bundle—zero external pip dependencies required.

Usage:
  python3 ml/import_kaggle_dataset.py                 # Import full 665-record metadata + pre-cache 12 images
  python3 ml/import_kaggle_dataset.py --fetch-image potholes0.png  # Extract & cache a single image on demand
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

KAGGLE_SLUG = "andrewmvd/pothole-detection"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "kaggle_images"
JSON_OUT = DATA_DIR / "kaggle_pothole_dataset.json"
CSV_OUT = DATA_DIR / "kaggle_pothole_dataset.csv"


class RemoteZipFile:
    """Seekable HTTP Range wrapper so zipfile.ZipFile reads only needed bytes."""

    def __init__(self, url: str):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"},
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            self.final_url = r.geturl()
            content_range = r.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise RuntimeError("Remote server does not support HTTP Range requests.")
            self.size = int(content_range.split("/")[-1])
        self.pos = 0
        self._cache: tuple[int, int, bytes] | None = None

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == 0 or self.pos >= self.size:
            return b""
        end = self.size - 1 if n < 0 else min(self.size - 1, self.pos + n - 1)

        if self._cache is not None:
            c_start, c_end, c_bytes = self._cache
            if self.pos >= c_start and end <= c_end:
                slice_data = c_bytes[self.pos - c_start : end - c_start + 1]
                self.pos += len(slice_data)
                return slice_data

        # Fetch with minimum 64KB prefetch when scanning central directory / small XMLs
        fetch_end = min(self.size - 1, max(end, self.pos + 262144 - 1))
        req = urllib.request.Request(
            self.final_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={self.pos}-{fetch_end}",
            },
        )
        with urllib.request.urlopen(req, timeout=45) as r:
            chunk = r.read()
        self._cache = (self.pos, self.pos + len(chunk) - 1, chunk)
        actual_len = end - self.pos + 1
        out = chunk[:actual_len]
        self.pos += len(out)
        return out


def natural_sort_key(name: str) -> tuple[str, int]:
    m = re.search(r"(\d+)", name)
    return (re.sub(r"\d+", "", name), int(m.group(1)) if m else 0)


def parse_voc_xml(xml_bytes: bytes, fallback_filename: str) -> dict:
    root = ET.fromstring(xml_bytes)
    filename = (root.findtext("filename") or fallback_filename).strip()
    size_el = root.find("size")
    width = int(float(size_el.findtext("width", "0"))) if size_el is not None else 0
    height = int(float(size_el.findtext("height", "0"))) if size_el is not None else 0

    boxes = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "pothole").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = int(round(float(bnd.findtext("xmin", "0"))))
        ymin = int(round(float(bnd.findtext("ymin", "0"))))
        xmax = int(round(float(bnd.findtext("xmax", "0"))))
        ymax = int(round(float(bnd.findtext("ymax", "0"))))
        boxes.append(
            {
                "label": label,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "boxCount": len(boxes),
        "boxes": boxes,
    }


def fetch_single_image(filename: str) -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(filename)
    out_path = IMAGES_DIR / safe_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    rz = RemoteZipFile(KAGGLE_DOWNLOAD_URL)
    with zipfile.ZipFile(rz) as zf:
        entry_name = f"images/{safe_name}"
        if entry_name not in zf.namelist():
            raise FileNotFoundError(f"Image {entry_name} not found in Kaggle dataset.")
        img_bytes = zf.read(entry_name)
        out_path.write_bytes(img_bytes)
    return out_path


def import_full_dataset(precache_count: int = 12) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rz = RemoteZipFile(KAGGLE_DOWNLOAD_URL)
    records = []

    with zipfile.ZipFile(rz) as zf:
        infos = zf.infolist()
        xml_infos = sorted(
            [info for info in infos if info.filename.startswith("annotations/") and info.filename.endswith(".xml")],
            key=lambda i: i.header_offset,
        )
        img_info_map = {
            os.path.basename(info.filename): info
            for info in infos
            if info.filename.startswith("images/")
            and info.filename.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        # Read all XML files in header_offset order so RemoteZipFile's 256KB prefetch
        # reads almost all 665 XMLs in just 2-3 HTTP range calls!
        for info in xml_infos:
            xml_bytes = zf.read(info)
            stem = Path(info.filename).stem
            parsed = parse_voc_xml(xml_bytes, f"{stem}.png")
            img_info = img_info_map.get(parsed["filename"])
            if img_info is None:
                # Fallback by stem
                for candidate_name, c_info in img_info_map.items():
                    if Path(candidate_name).stem == stem:
                        parsed["filename"] = candidate_name
                        img_info = c_info
                        break

            img_size_bytes = img_info.file_size if img_info else 0
            records.append(
                {
                    "id": stem,
                    "filename": parsed["filename"],
                    "annotationFile": info.filename,
                    "width": parsed["width"],
                    "height": parsed["height"],
                    "fileSizeBytes": img_size_bytes,
                    "boxCount": parsed["boxCount"],
                    "boxes": parsed["boxes"],
                }
            )

        records.sort(key=lambda r: natural_sort_key(r["id"]))

        # Pre-cache a curated sample of images locally so initial UI load is instant
        if precache_count > 0:
            step = max(1, len(records) // precache_count)
            selected = [records[i] for i in range(0, len(records), step)][:precache_count]
            for rec in selected:
                out_path = IMAGES_DIR / rec["filename"]
                if not out_path.exists():
                    entry_name = f"images/{rec['filename']}"
                    if entry_name in zf.namelist():
                        out_path.write_bytes(zf.read(entry_name))

    total_boxes = sum(r["boxCount"] for r in records)
    cached_files = {p.name for p in IMAGES_DIR.iterdir() if p.is_file()}
    for r in records:
        r["cachedLocally"] = r["filename"] in cached_files

    catalog = {
        "dataset": KAGGLE_SLUG,
        "sourceUrl": f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}",
        "format": "Pascal VOC XML + PNG Images",
        "importedAt": datetime.now(timezone.utc).isoformat(),
        "totalImages": len(records),
        "totalBoxes": total_boxes,
        "avgBoxesPerImage": round(total_boxes / max(1, len(records)), 2),
        "cachedImageCount": len(cached_files),
        "records": records,
    }

    JSON_OUT.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    # Also write tabular CSV (compatible with pandas / spreadsheet inspection)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "filename", "width", "height", "box_count", "boxes_json"])
        for r in records:
            writer.writerow(
                [
                    r["id"],
                    r["filename"],
                    r["width"],
                    r["height"],
                    r["boxCount"],
                    json.dumps(r["boxes"]),
                ]
            )

    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Kaggle pothole-detection dataset")
    parser.add_argument("--fetch-image", type=str, default="", help="Download & cache a single image filename")
    parser.add_argument("--precache", type=int, default=12, help="Number of images to pre-cache locally")
    args = parser.parse_args()

    if args.fetch_image:
        out_path = fetch_single_image(args.fetch_image)
        print(json.dumps({"ok": True, "path": str(out_path), "filename": out_path.name}))
        return 0

    catalog = import_full_dataset(precache_count=args.precache)
    summary = {k: v for k, v in catalog.items() if k != "records"}
    summary["first5"] = catalog["records"][:5]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
