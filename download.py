#!/usr/bin/env python3
"""
PAN-NZ catalogue downloader (no CLI args), with named APIs by host.

Supports:
- ArcGIS REST Feature Services -> GeoJSON (paged)
- Koordinates/LINZ/LRIS/Waikato WFS -> GeoJSON, with per-host API keys

Expected layout:
repo_root/
  download_catalogue_datasets.py
  secrets.py
  _data/
    national_protected_areas.csv
    regional_protected_areas.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# USER SETTINGS
# =========================
REPO_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_DIR / "downloads"

ONLY = "all"  # "all" | "rest" | "wfs"

ARCGIS_PAGE_SIZE = 2000
DEFAULT_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.15

MAX_WORKERS = 8
MAX_WORKERS_ARCGIS = 3
MAX_WORKERS_WFS = 8

RETRIES = 3
RETRY_BACKOFF_SEC = 1.5

SKIP_IF_EXISTS = True

USER_AGENT = "PAN-NZ-downloader/2.0"


# =========================
# API NAMING / HOST MAP
# =========================
# Map host -> friendly API name
API_BY_HOST: Dict[str, str] = {
    "data.linz.govt.nz": "LINZ",
    "lris.scinfo.org.nz": "LRIS",
    "data.waikatodistrict.govt.nz": "WAIKATO",
    # add more Koordinates portals here as you encounter them
    # "data.aucklandcouncil.govt.nz": "AUCKLAND",
}

# Names to use in secrets.py (keys)
# If a host isn't in API_BY_HOST, we'll label it "OTHER:<host>" and look for secrets.OTHER_<sanitised_host>
# Example: secrets.OTHER_data_example_com = "..."
# (see read_api_key_for_host())
# =========================


# =========================
# Helpers
# =========================
def safe_filename(s: str, max_len: int = 180) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-\. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("_.")
    return (s or "dataset")[:max_len]


def parse_host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def api_name_for_url(url: str) -> str:
    host = parse_host(url)
    return API_BY_HOST.get(host, f"OTHER:{host}" if host else "OTHER:unknown")


def sanitise_host_for_attr(host: str) -> str:
    # secrets attribute-friendly
    return re.sub(r"[^\w]+", "_", host.strip().lower()).strip("_")


def read_api_key_for_host(host: str) -> Optional[str]:
    """
    Resolve an API key for a given host.

    Priority:
    1) secrets.<API_NAME>  (e.g., secrets.LINZ, secrets.LRIS, secrets.WAIKATO)
    2) secrets.OTHER_<sanitised_host> for unknown portals
    3) env var KOORDINATES_API_KEY_<API_NAME>
    4) env var KOORDINATES_API_KEY (fallback)

    Returns None if no key is available (some WFS layers might be public).
    """
    api_name = API_BY_HOST.get(host)
    try:
        import secrets as user_secrets  # type: ignore

        if api_name:
            v = getattr(user_secrets, api_name, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # fallback for unknown hosts
        other_attr = f"OTHER_{sanitise_host_for_attr(host)}"
        v2 = getattr(user_secrets, other_attr, None)
        if isinstance(v2, str) and v2.strip():
            return v2.strip()

    except Exception:
        pass

    if api_name:
        v3 = os.environ.get(f"KOORDINATES_API_KEY_{api_name}", "").strip()
        if v3:
            return v3

    v4 = os.environ.get("KOORDINATES_API_KEY", "").strip()
    if v4:
        return v4

    return None


def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def with_retries(fn, *args, **kwargs):
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise last_err


@dataclass
class CatalogueRow:
    catalogue: str
    custodian_code: str
    custodian_name: str
    dataset_name: str
    api_type: str
    api_url: str
    download_url: str

    @property
    def label(self) -> str:
        bits = [self.catalogue, self.custodian_code or self.custodian_name, self.dataset_name]
        return safe_filename("_".join([b for b in bits if b]))


def load_catalogue_rows(csv_path: Path, catalogue_name: str) -> List[CatalogueRow]:
    out: List[CatalogueRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append(
                CatalogueRow(
                    catalogue=catalogue_name,
                    custodian_code=(r.get("Custodian code") or "").strip(),
                    custodian_name=(r.get("Custodian name") or "").strip(),
                    dataset_name=(r.get("Dataset name") or "").strip(),
                    api_type=(r.get("API Type") or "").strip(),
                    api_url=(r.get("API URL") or "").strip(),
                    download_url=(r.get("Download URL") or "").strip(),
                )
            )
    return out


def choose_service_url(row: CatalogueRow) -> str:
    return row.api_url or row.download_url


def is_arcgis_rest(url: str, api_type: str) -> bool:
    u = (url or "").lower()
    t = (api_type or "").lower()
    return ("arcgis/rest" in u) or ("feature service" in t and "arcgis" in t)


def is_wfs(url: str, api_type: str) -> bool:
    u = (url or "").lower()
    t = (api_type or "").lower()
    return ("wfs" in u) or ("ogc wfs" in t)


# =========================
# ArcGIS REST downloader
# =========================
def arcgis_query_geojson(
    session: requests.Session,
    layer_url: str,
    out_geojson_path: Path,
    where: str = "1=1",
    out_fields: str = "*",
    page_size: int = 2000,
    max_pages: int = 100000,
) -> None:
    layer_url = layer_url.rstrip("/")
    query_url = f"{layer_url}/query"

    features: List[dict] = []
    offset = 0
    page = 0

    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "returnGeometry": "true",
        }

        resp = session.get(query_url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"ArcGIS query failed ({resp.status_code}): {resp.text[:500]}")

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"ArcGIS error: {json.dumps(data['error'], indent=2)[:2000]}")

        page_features = data.get("features", [])
        if not page_features:
            break

        features.extend(page_features)
        offset += len(page_features)
        page += 1

        if page >= max_pages:
            raise RuntimeError("Hit max_pages safety limit while paging ArcGIS service.")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    out_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    out_geojson_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


# =========================
# WFS downloader (Koordinates portals + others)
# =========================
def extract_layer_id_like(url: str) -> Optional[str]:
    """
    Koordinates portals often expose layer ids like 'layer-115910' in URLs.
    """
    m = re.search(r"(layer-\d+)", url or "")
    return m.group(1) if m else None


def build_wfs_candidate_urls(host: str, layer_id: str, api_key: Optional[str]) -> List[str]:
    """
    Build multiple GetFeature URL candidates.
    We try:
      - /services;key=KEY/wfs?...
      - /services;key=KEY/wfs/{layer_id}?...
      - /wfs?key=KEY&...      (some portals use key= as query param)
      - /wfs?...

    We keep typeNames=layer-xxxxx and outputFormat=json (GeoJSON) in query.
    """
    layer_id = layer_id.strip()

    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": layer_id,
        "outputFormat": "json",
    }

    # base query without key
    qs_no_key = urllib.parse.urlencode(params)

    urls: List[str] = []

    if api_key:
        key_enc = urllib.parse.quote(api_key)

        # Pattern A/B: services;key=... (common on LINZ/LRIS and many Koordinates portals)
        urls.append(f"https://{host}/services;key={key_enc}/wfs?{qs_no_key}")
        urls.append(f"https://{host}/services;key={key_enc}/wfs/{layer_id}?{qs_no_key}")
        urls.append(f"https://{host}/services;key={key_enc}/wfs/?{qs_no_key}")

        # Pattern C: key=... as query parameter (Koordinates Query API style)
        params_with_key = dict(params)
        params_with_key["key"] = api_key
        qs_key = urllib.parse.urlencode(params_with_key)
        urls.append(f"https://{host}/wfs?{qs_key}")
        urls.append(f"https://{host}/wfs/?{qs_key}")

    # Pattern D/E: no key at all (public)
    urls.append(f"https://{host}/wfs?{qs_no_key}")
    urls.append(f"https://{host}/wfs/?{qs_no_key}")

    return urls


def download_url_to_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, headers=headers or {}, stream=True, timeout=DEFAULT_TIMEOUT) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Download failed ({r.status_code}): {url}\n{r.text[:500]}")
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


# =========================
# Per-row worker
# =========================
def process_row(row: CatalogueRow) -> Tuple[str, str]:
    """
    Returns (label, status), where status is:
      - "ok"
      - "skip"
    Raises on failure.
    """
    url = choose_service_url(row)
    if not url or url.strip() in ("-", "NA", "N/A"):
        return (row.label, "skip")

    session = requests_session()

    # ArcGIS REST
    if ONLY in ("all", "rest") and is_arcgis_rest(url, row.api_type):
        out_path = OUT_DIR / "arcgis_rest" / f"{row.label}.geojson"
        if SKIP_IF_EXISTS and out_path.exists():
            return (row.label, "skip")

        with_retries(
            arcgis_query_geojson,
            session=session,
            layer_url=url,
            out_geojson_path=out_path,
            page_size=ARCGIS_PAGE_SIZE,
        )
        return (row.label, "ok")

    # WFS
    if ONLY in ("all", "wfs") and is_wfs(url, row.api_type):
        host = parse_host(url)
        layer_id = extract_layer_id_like(url)
        api_name = api_name_for_url(url)

        if not host or not layer_id:
            # Not a Koordinates-style WFS row we can infer; skip quietly
            return (row.label, "skip")

        api_key = read_api_key_for_host(host)

        out_path = OUT_DIR / "wfs" / api_name.replace(":", "_") / f"{row.label}.geojson"
        if SKIP_IF_EXISTS and out_path.exists():
            return (row.label, "skip")

        last_err: Optional[Exception] = None
        for cand in build_wfs_candidate_urls(host, layer_id, api_key):
            try:
                with_retries(download_url_to_file, session, cand, out_path)
                return (row.label, "ok")
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"WFS failed for {row.label} ({api_name}, {host}, {layer_id}). Last: {last_err}")

    return (row.label, "skip")


# =========================
# Script entrypoint
# =========================
def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    national_csv = REPO_DIR / "_data" / "national_protected_areas.csv"
    regional_csv = REPO_DIR / "_data" / "regional_protected_areas.csv"
    if not national_csv.exists() or not regional_csv.exists():
        raise FileNotFoundError(
            "Expected:\n"
            f"  {national_csv}\n"
            f"  {regional_csv}\n"
            "Place this script in the repo root."
        )

    rows: List[CatalogueRow] = []
    rows += load_catalogue_rows(national_csv, "national")
    rows += load_catalogue_rows(regional_csv, "regional")

    # Split pools so ArcGIS paging doesn’t dominate
    arcgis_rows = [r for r in rows if is_arcgis_rest(choose_service_url(r), r.api_type)]
    wfs_rows = [r for r in rows if is_wfs(choose_service_url(r), r.api_type)]
    other_rows = [r for r in rows if r not in arcgis_rows and r not in wfs_rows]

    successes = 0
    skips = 0
    failures: List[Tuple[str, str]] = []

    def run_pool(pool_rows: List[CatalogueRow], workers: int, label: str) -> None:
        nonlocal successes, skips, failures
        if not pool_rows:
            return

        print(f"\n== {label}: {len(pool_rows)} datasets, workers={workers} ==")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_row, r): r for r in pool_rows}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    name, status = fut.result()
                    if status == "ok":
                        successes += 1
                        print(f"[OK]   {name}")
                    else:
                        skips += 1
                except Exception as e:
                    failures.append((r.label, str(e)))
                    print(f"[FAIL] {r.label}: {e}")

    run_pool(arcgis_rows, min(MAX_WORKERS, MAX_WORKERS_ARCGIS), "ArcGIS REST")
    run_pool(wfs_rows, min(MAX_WORKERS, MAX_WORKERS_WFS), "WFS (Koordinates/LINZ/LRIS/Waikato etc.)")
    run_pool(other_rows, min(MAX_WORKERS, 3), "Other / Unclassified")

    print(f"\nDone. Successes: {successes}, Skips: {skips}, Failures: {len(failures)}")
    if failures:
        fail_log = OUT_DIR / "failures.txt"
        with fail_log.open("w", encoding="utf-8") as f:
            for name, err in failures:
                f.write(f"{name}\n{err}\n\n")
        print(f"Wrote failure details: {fail_log}")


if __name__ == "__main__":
    run()