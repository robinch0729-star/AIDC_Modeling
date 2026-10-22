"""Download the complete official Azure 2024 traces, preserving exact bytes.

No resampling, row filtering, scaling, or power-model changes are performed.
The published GitHub release size AND SHA-256 must match before finalization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com/repos/Azure/AzurePublicDataset/releases/tags/dataset-llm-2024"
NAMES = {"code": "AzureLLMInferenceTrace_code_1week.csv",
         "conversation": "AzureLLMInferenceTrace_conv_1week.csv"}
HEADERS = {"User-Agent": "AIDC-research-long-trace-audit/1.0", "Accept-Encoding": "identity"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def get(url, extra=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**HEADERS, **(extra or {})}), timeout=45)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def download(service, asset, directory):
    destination = directory / asset["name"]
    partial = directory / (asset["name"] + ".part")
    resume_meta = directory / (asset["name"] + ".download.json")
    size = int(asset["size"])
    digest = asset.get("digest", "")
    if not digest or not digest.startswith("sha256:"):
        raise RuntimeError("A published SHA-256 is required; no silent size-only verification")
    expected = digest.split(":", 1)[1]
    identity = {"id": asset["id"], "size": size, "sha256": expected}
    if destination.exists():
        if destination.stat().st_size != size or sha256(destination) != expected:
            raise RuntimeError(f"Existing final file fails verification: {destination}")
        print(f"{service}: existing complete file verified", flush=True)
    else:
        if partial.exists():
            if not resume_meta.exists() or json.loads(resume_meta.read_text(encoding="utf-8")) != identity:
                raise RuntimeError("Partial file identity does not match current release")
        else:
            save_json(resume_meta, identity)
        for attempt in range(1, 5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise RuntimeError("Partial file exceeds published size")
            if offset == size:
                break
            request_headers = {"Range": f"bytes={offset}-"} if offset else {}
            url = asset["browser_download_url"] + "?audit=" + uuid.uuid4().hex
            last_print = time.monotonic()
            try:
                with get(url, request_headers) as response:
                    if offset:
                        expected_range = f"bytes {offset}-{size-1}/{size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise RuntimeError("Resume response has unexpected byte range")
                    elif response.status != 200:
                        raise RuntimeError(f"Expected a complete response, got {response.status}")
                    if int(response.headers.get("Content-Length", -1)) != size - offset:
                        raise RuntimeError("Content-Length differs from published asset size")
                    with partial.open("ab" if offset else "wb") as output:
                        while True:
                            data = response.read(1024**2)
                            if not data:
                                break
                            output.write(data)
                            offset += len(data)
                            if offset > size:
                                raise RuntimeError("Response exceeded expected length")
                            if time.monotonic() - last_print > 10:
                                print(f"{service}: {offset/size:.1%} ({offset/1e6:.0f}/{size/1e6:.0f} MB)", flush=True)
                                last_print = time.monotonic()
                if offset == size:
                    break
            except (OSError, TimeoutError) as error:
                print(f"{service}: attempt {attempt}: {type(error).__name__}; partial retained", flush=True)
                if attempt == 4:
                    raise
        if partial.stat().st_size != size or sha256(partial) != expected:
            raise RuntimeError(f"Download integrity failure for {service}; partial retained")
        partial.rename(destination)
        print(f"{service}: full file and published SHA-256 verified", flush=True)
    return {"service": service, "path": str(destination), "source_url": asset["browser_download_url"],
            "asset_id": asset["id"], "asset_updated_at": asset["updated_at"], "bytes": size,
            "sha256": expected, "published_sha256_match": True, "complete_download": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/azure_inference_2024_full")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with get(API) as response:
        release = json.load(response)
    assets = {a["name"]: a for a in release["assets"]}
    save_json(directory / "release_metadata.json", release)
    for name, url in {
        "source_readme.md": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureLLMInferenceDataset2024.md",
        "LICENSE.txt": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/LICENSE",
    }.items():
        with get(url) as response:
            (directory / name).write_bytes(response.read())
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(download, service, assets[name], directory) for service, name in NAMES.items()]
        sources = [job.result() for job in jobs]
    manifest = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "release_api": API,
                "dataset": "Azure LLM inference trace 2024", "license": "CC BY 4.0",
                "reference": "Stojkovic et al., DynamoLLM, IEEE HPCA 2025, DOI 10.1109/HPCA61900.2025.00102",
                "source_readme_sha256": sha256(directory / "source_readme.md"),
                "license_sha256": sha256(directory / "LICENSE.txt"), "sources": sources,
                "raw_bytes_modified": False, "synthetic_fallback": False}
    save_json(directory / "download_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
