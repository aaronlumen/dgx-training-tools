#!/usr/bin/env python3
"""
cap_downloader.py — Bulk PDF downloader for Harvard Caselaw Access Project.

Downloads all volume PDFs from static.case.law bulk download pages into
organised local directories. Fully resumable — skips files already on disk.

Destinations:
  ~/llm-training/pdfs/legal/wash/     ← wash, wash-2d, wash-app, wash-terr
  ~/llm-training/pdfs/legal/us/       ← us, us-app-dc, us-ct-cl
  ~/llm-training/pdfs/legal/montana/  ← mont
  ~/llm-training/pdfs/legal/idaho/    ← idaho

Usage:
  python3 cap_downloader.py                   # download all
  python3 cap_downloader.py --only wash       # just Washington sources
  python3 cap_downloader.py --only us idaho   # US federal + Idaho
  python3 cap_downloader.py --workers 4       # parallel downloads (default: 3)
  python3 cap_downloader.py --dry-run         # show what would be downloaded

Author: Aaron Surina
"""

import argparse
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("[ERR] requests not installed: pip install requests")

# ── destination mapping ───────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(os.environ.get("TOOR_HOME", "/home/toor")) / "llm-training" / "pdfs" / "legal"

SOURCES = {
    "wash": {
        "dir": BASE_DIR / "wash",
        "urls": [
            "https://static.case.law/wash/",
            "https://static.case.law/wash-2d/",
            "https://static.case.law/wash-app/",
            "https://static.case.law/wash-terr/",
        ],
    },
    "us": {
        "dir": BASE_DIR / "us",
        "urls": [
            "https://static.case.law/us/",
            "https://static.case.law/us-app-dc/",
            "https://static.case.law/us-ct-cl/",
        ],
    },
    "montana": {
        "dir": BASE_DIR / "montana",
        "urls": [
            "https://static.case.law/mont/",
        ],
    },
    "idaho": {
        "dir": BASE_DIR / "idaho",
        "urls": [
            "https://static.case.law/idaho/",
        ],
    },
}

# ── HTTP session with retry ───────────────────────────────────────────────────

def _make_session():
    s = requests.Session()
    s.headers["User-Agent"] = (
        "dgx-training-tools/1.0 (aaron.surina@gmail.com; "
        "bulk research download, Harvard CAP)"
    )
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

SESSION = _make_session()

# ── index scraper ─────────────────────────────────────────────────────────────

def get_pdf_urls(index_url):
    """Fetch an index page and return all absolute .pdf URLs found."""
    try:
        resp = SESSION.get(index_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERR] Could not fetch index {index_url}: {e}")
        return []

    # match href="<something>.pdf" or href="<something>.pdf"
    hrefs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', resp.text, re.IGNORECASE)

    pdf_urls = []
    for href in hrefs:
        if href.startswith("http"):
            pdf_urls.append(href)
        elif href.startswith("/"):
            base = re.match(r"(https?://[^/]+)", index_url).group(1)
            pdf_urls.append(base + href)
        else:
            pdf_urls.append(index_url.rstrip("/") + "/" + href.lstrip("/"))

    return sorted(set(pdf_urls))

# ── single file download ──────────────────────────────────────────────────────

def download_pdf(url, dest_dir, dry_run=False):
    """
    Download a single PDF to dest_dir/<source_name>/<filename>.
    Returns (url, status) where status is 'ok', 'skip', or 'err:<msg>'.
    """
    # derive a subdirectory name from the URL path segment
    # e.g. https://static.case.law/wash-2d/47.pdf
    #   → dest_dir/wash-2d/47.pdf
    parts   = url.replace("https://static.case.law/", "").split("/")
    subdir  = parts[0] if len(parts) > 1 else "unknown"
    fname   = parts[-1]
    out_dir = dest_dir / subdir
    out_path = out_dir / fname

    if out_path.exists() and out_path.stat().st_size > 1024:
        return url, "skip"

    if dry_run:
        return url, f"dry-run → {out_path}"

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = SESSION.get(url, timeout=120, stream=True)
        resp.raise_for_status()

        tmp_path = out_path.with_suffix(".part")
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        tmp_path.rename(out_path)
        return url, "ok"

    except requests.RequestException as e:
        if out_path.with_suffix(".part").exists():
            out_path.with_suffix(".part").unlink(missing_ok=True)
        return url, f"err:{e}"

# ── progress printer ──────────────────────────────────────────────────────────

def _bar(done, total, width=40):
    filled = int(width * done / max(total, 1))
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"

# ── main downloader ───────────────────────────────────────────────────────────

def run_source(name, cfg, workers, dry_run):
    dest_dir = cfg["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  {name.upper()}  →  {dest_dir}")
    print(f"{'═' * 60}")

    all_urls = []
    for idx_url in cfg["urls"]:
        print(f"  Scanning: {idx_url}")
        pdfs = get_pdf_urls(idx_url)
        print(f"    {len(pdfs)} PDFs found")
        all_urls.extend(pdfs)

    if not all_urls:
        print("  [WARN] No PDFs found — skipping.")
        return

    total  = len(all_urls)
    done   = 0
    errors = []
    skips  = 0

    print(f"\n  Total: {total} PDFs  |  workers: {workers}  |  dest: {dest_dir}\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_pdf, url, dest_dir, dry_run): url
                   for url in all_urls}
        for future in as_completed(futures):
            url, status = future.result()
            done += 1
            fname = url.split("/")[-1]
            if status == "skip":
                skips += 1
            elif status.startswith("err"):
                errors.append((url, status))
                print(f"\r  ✗ {fname}: {status[4:]}")
            else:
                pass  # ok or dry-run

            print(f"  {_bar(done, total)}  skip={skips} err={len(errors)}", end="\r")

    print(f"\n\n  ✓ {done - len(errors) - skips} downloaded  "
          f"|  {skips} skipped (already on disk)  "
          f"|  {len(errors)} errors")

    if errors:
        print(f"\n  Failed downloads:")
        for url, msg in errors[:20]:
            print(f"    {url}  ({msg[4:]})")
        if len(errors) > 20:
            print(f"    … and {len(errors) - 20} more")

        err_log = dest_dir / "download_errors.txt"
        err_log.write_text("\n".join(f"{u}\t{m}" for u, m in errors) + "\n")
        print(f"\n  Error log: {err_log}")

# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Harvard CAP bulk PDFs by jurisdiction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available groups:  wash  us  montana  idaho

Examples:
  python3 cap_downloader.py                     # all jurisdictions
  python3 cap_downloader.py --only wash         # Washington only
  python3 cap_downloader.py --only us idaho     # federal + Idaho
  python3 cap_downloader.py --workers 6         # 6 parallel downloads
  python3 cap_downloader.py --dry-run           # show counts, no download
        """,
    )
    parser.add_argument("--only",    nargs="+", choices=list(SOURCES.keys()),
                        help="Limit to specific jurisdiction group(s)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel download threads per group (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan indexes and show what would be downloaded, no files written")
    args = parser.parse_args()

    targets = args.only if args.only else list(SOURCES.keys())

    print(f"\nHarvard CAP Bulk PDF Downloader")
    print(f"Base dir : {BASE_DIR}")
    print(f"Groups   : {', '.join(targets)}")
    print(f"Workers  : {args.workers}")
    if args.dry_run:
        print("DRY RUN  : no files will be written")

    start = time.time()
    for name in targets:
        run_source(name, SOURCES[name], args.workers, args.dry_run)

    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"  All done in {elapsed/60:.1f} min")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
