#!/usr/bin/env python3
"""
legal_downloader.py — Download legal training data from free public sources.

Sources:
  courtlistener  — CourtListener bulk opinion downloads (Free Law Project)
  stackexchange  — law.stackexchange.com data dump (archive.org)
  harvardcap     — Harvard Caselaw Access Project API
  crs            — Congressional Research Service reports (everycrsreport.com)
  olc            — DOJ Office of Legal Counsel memoranda

Usage:
  python3 legal_downloader.py --source courtlistener --output data/courtlistener.jsonl
  python3 legal_downloader.py --source stackexchange  --output data/law_se.jsonl
  python3 legal_downloader.py --source harvardcap     --output data/harvard_cap.jsonl --api-key YOUR_KEY
  python3 legal_downloader.py --source crs            --output data/crs_reports.jsonl
  python3 legal_downloader.py --source olc            --output data/olc_memos.jsonl
  python3 legal_downloader.py --source all            --output data/

Author: Aaron Surina
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
import zipfile
from io import BytesIO

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"

try:
    import requests
except ImportError:
    sys.exit("[ERR] requests not installed: pip install requests")


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(records, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):,} records → {path}")
    return path


def _clean(text):
    if not text:
        return ""
    text = re.sub(r"\s{3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _session(token=None):
    s = requests.Session()
    s.headers.update({"User-Agent": "dgx-training-tools/1.0 (aaron.surina@gmail.com)"})
    if token:
        s.headers["Authorization"] = f"Token {token}"
    return s


# ── CourtListener ─────────────────────────────────────────────────────────────

def download_courtlistener(output_path, max_records=50000, jurisdiction=None):
    """
    Stream opinions from the CourtListener REST API.
    Docs: https://www.courtlistener.com/api/rest/v4/
    No API key required for read access (rate-limited to 5,000/day unauthenticated).
    Set CL_TOKEN env var for 100k/day authenticated access.
    """
    print(f"\n[CourtListener] Fetching up to {max_records:,} opinions …")
    token  = os.environ.get("CL_TOKEN")
    sess   = _session(token)
    params = {
        "format":   "json",
        "page_size": 100,
        "order_by":  "id",
        "type":      "o",   # opinions
    }
    if jurisdiction:
        params["court__jurisdiction"] = jurisdiction  # e.g. "F" federal, "S" state

    url     = "https://www.courtlistener.com/api/rest/v4/opinions/"
    records = []
    page    = 0

    while url and len(records) < max_records:
        try:
            resp = sess.get(url, params=params if page == 0 else None, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] Request failed: {e} — stopping early")
            break

        data    = resp.json()
        results = data.get("results", [])
        page   += 1

        for op in results:
            text = _clean(
                op.get("plain_text") or
                op.get("html_with_citations") or
                op.get("html") or ""
            )
            # strip HTML tags if present
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s{3,}", "  ", text).strip()
            if len(text) < 200:
                continue

            meta = {
                "source":      "courtlistener",
                "case_name":   op.get("cluster", {}).get("case_name", ""),
                "date_filed":  op.get("cluster", {}).get("date_filed", ""),
                "court":       op.get("cluster", {}).get("court", ""),
                "citation":    op.get("cluster", {}).get("citations", [{}])[0] if op.get("cluster", {}).get("citations") else "",
            }
            records.append({"text": text, **meta})
            if len(records) >= max_records:
                break

        if len(records) % 1000 < 100:
            print(f"  … {len(records):,} / {max_records:,}", end="\r")

        url    = data.get("next")
        params = None
        time.sleep(0.5)  # be polite

    return _write_jsonl(records, output_path)


# ── Stack Exchange Law dump ───────────────────────────────────────────────────

def download_stackexchange(output_path, max_records=50000):
    """
    Download law.stackexchange.com data dump from archive.org and parse
    Posts.xml into Q&A pairs.
    """
    print("\n[StackExchange Law] Downloading data dump …")
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        sys.exit("[ERR] xml.etree.ElementTree not available")

    dump_url = (
        "https://archive.org/download/stackexchange/"
        "law.stackexchange.com.7z"
    )
    # Try the plain zip mirror first (smaller, no 7z dependency)
    zip_url  = (
        "https://archive.org/download/stackexchange/"
        "law.stackexchange.com_Posts.xml.zip"
    )

    sess = _session()

    # attempt zip first
    xml_content = None
    for url in (zip_url,):
        try:
            print(f"  Downloading: {url}")
            resp = sess.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            buf   = BytesIO()
            done  = 0
            for chunk in resp.iter_content(65536):
                buf.write(chunk)
                done += len(chunk)
                if total:
                    print(f"  {done/1e6:.1f} / {total/1e6:.1f} MB", end="\r")
            buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                posts_name = next((n for n in names if "Posts" in n), names[0])
                xml_content = zf.read(posts_name)
            print(f"\n  Extracted {len(xml_content)/1e6:.1f} MB XML")
            break
        except Exception as e:
            print(f"  [WARN] {e}")

    if xml_content is None:
        print("  [ERR] Could not download dump. Try manually:")
        print("    wget 'https://archive.org/download/stackexchange/law.stackexchange.com_Posts.xml.zip'")
        print("    Then: python3 legal_downloader.py --source stackexchange --input Posts.xml --output data/law_se.jsonl")
        return None

    # parse XML
    print("  Parsing Posts.xml …")
    root      = ET.fromstring(xml_content)
    questions = {}
    answers   = []

    for row in root.iter("row"):
        a = row.attrib
        post_type = a.get("PostTypeId")
        if post_type == "1":  # question
            questions[a["Id"]] = {
                "title": _clean(a.get("Title", "")),
                "body":  re.sub(r"<[^>]+>", " ", a.get("Body", "")),
                "tags":  a.get("Tags", ""),
                "score": int(a.get("Score", 0)),
            }
        elif post_type == "2":  # answer
            answers.append({
                "parent": a.get("ParentId"),
                "body":   re.sub(r"<[^>]+>", " ", a.get("Body", "")),
                "score":  int(a.get("Score", 0)),
                "accepted": a.get("Id") == a.get("AcceptedAnswerId", ""),
            })

    # pair Q+A, keep highest-scored answer per question
    best_answer = {}
    for ans in answers:
        pid = ans["parent"]
        if pid not in best_answer or ans["score"] > best_answer[pid]["score"]:
            best_answer[pid] = ans

    records = []
    for qid, q in questions.items():
        if q["score"] < 0:
            continue
        ans = best_answer.get(qid)
        if not ans:
            continue
        a_text = _clean(ans["body"])
        q_text = _clean(q["body"])
        if len(q_text) < 50 or len(a_text) < 50:
            continue

        text = f"Question: {q['title']}\n\n{q_text}\n\nAnswer:\n{a_text}"
        records.append({
            "text":   text,
            "source": "law_stackexchange",
            "tags":   q["tags"],
        })
        if len(records) >= max_records:
            break

    return _write_jsonl(records, output_path)


# ── Harvard Caselaw Access Project ────────────────────────────────────────────

def download_harvardcap(output_path, max_records=50000, jurisdiction="us", api_key=None):
    """
    Download cases from the Harvard Caselaw Access Project.
    Free API key at: https://case.law/user/register/
    Without a key: full text is redacted; only metadata available.
    """
    print(f"\n[Harvard CAP] Fetching up to {max_records:,} cases (jurisdiction={jurisdiction}) …")
    if not api_key:
        api_key = os.environ.get("HARVARD_CAP_KEY")
    if not api_key:
        print("  [WARN] No API key — full text will be unavailable.")
        print("  Register free at: https://case.law/user/register/")

    sess    = _session()
    if api_key:
        sess.headers["Authorization"] = f"Token {api_key}"

    url     = "https://api.case.law/v1/cases/"
    params  = {
        "jurisdiction": jurisdiction,
        "full_case":    "true",
        "page_size":    100,
    }
    records = []

    while url and len(records) < max_records:
        try:
            resp = sess.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] {e} — stopping early")
            break

        data = resp.json()
        for case in data.get("results", []):
            body = case.get("casebody", {})
            opinions = body.get("data", {}).get("opinions", []) if isinstance(body.get("data"), dict) else []
            text_parts = [op.get("text", "") for op in opinions]
            text = _clean("\n\n".join(text_parts))
            if len(text) < 200:
                continue
            records.append({
                "text":         f"{case.get('name', '')}\n\n{text}",
                "source":       "harvard_cap",
                "citation":     case.get("citations", [{}])[0].get("cite", "") if case.get("citations") else "",
                "court":        case.get("court", {}).get("name", ""),
                "decision_date": case.get("decision_date", ""),
            })
            if len(records) >= max_records:
                break

        if len(records) % 1000 < 100:
            print(f"  … {len(records):,} / {max_records:,}", end="\r")

        url    = data.get("next")
        params = None
        time.sleep(0.3)

    return _write_jsonl(records, output_path)


# ── CRS Reports ───────────────────────────────────────────────────────────────

def download_crs(output_path, max_records=10000):
    """
    Congressional Research Service reports via everycrsreport.com.
    Full-text memos on legal, policy, and regulatory topics. Public domain.
    """
    print(f"\n[CRS Reports] Fetching index …")
    sess = _session()

    try:
        resp = sess.get("https://www.everycrsreport.com/reports.json", timeout=30)
        resp.raise_for_status()
        index = resp.json()
    except Exception as e:
        print(f"  [ERR] Could not fetch index: {e}")
        return None

    print(f"  Index: {len(index):,} reports available")
    records = []

    for entry in index[:max_records * 2]:  # fetch extra to account for failures
        if len(records) >= max_records:
            break
        versions = entry.get("versions", [])
        if not versions:
            continue
        latest = versions[-1]
        txt_url = None
        for fmt in latest.get("formats", []):
            if fmt.get("format") == "txt":
                txt_url = "https://www.everycrsreport.com" + fmt["filename"]
                break
        if not txt_url:
            continue
        try:
            r = sess.get(txt_url, timeout=20)
            r.raise_for_status()
            text = _clean(r.text)
            if len(text) < 200:
                continue
            records.append({
                "text":   text,
                "source": "crs_reports",
                "title":  entry.get("title", ""),
                "number": entry.get("number", ""),
                "date":   latest.get("date", ""),
            })
            if len(records) % 100 == 0:
                print(f"  … {len(records):,} / {max_records:,}", end="\r")
            time.sleep(0.1)
        except Exception:
            continue

    return _write_jsonl(records, output_path)


# ── DOJ OLC Memoranda ─────────────────────────────────────────────────────────

def download_olc(output_path, max_records=2000):
    """
    DOJ Office of Legal Counsel opinions from justice.gov.
    These are the gold standard of legal memoranda — formal, reasoned, cited.
    """
    print(f"\n[DOJ OLC] Fetching opinion index …")
    sess = _session()

    try:
        resp = sess.get(
            "https://www.justice.gov/olc/opinions",
            timeout=30
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERR] {e}")
        return None

    try:
        from html.parser import HTMLParser
    except ImportError:
        pass

    # extract PDF/text links from page
    links = re.findall(
        r'href="(/olc/[^"]+\.(?:pdf|txt|htm|html))"',
        resp.text
    )
    base    = "https://www.justice.gov"
    records = []

    # also check paginated pages
    page_links = re.findall(r'href="(/olc/opinions\?[^"]*page=\d+[^"]*)"', resp.text)
    all_pages  = [""] + page_links[:20]

    all_doc_links = set(links)
    for pg in all_pages[1:]:
        try:
            r = sess.get(base + pg, timeout=20)
            more = re.findall(r'href="(/olc/[^"]+\.(?:pdf|txt|htm|html))"', r.text)
            all_doc_links.update(more)
            time.sleep(0.3)
        except Exception:
            continue

    print(f"  Found {len(all_doc_links)} opinion links")

    for link in list(all_doc_links)[:max_records]:
        if len(records) >= max_records:
            break
        url = base + link
        try:
            r = sess.get(url, timeout=30)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "pdf" in content_type or link.endswith(".pdf"):
                # try pdfminer or skip if not available
                try:
                    import fitz
                    doc  = fitz.open(stream=r.content, filetype="pdf")
                    text = "\n\n".join(page.get_text() for page in doc)
                    doc.close()
                except ImportError:
                    continue
            else:
                text = re.sub(r"<[^>]+>", " ", r.text)
            text = _clean(text)
            if len(text) < 200:
                continue
            records.append({
                "text":   text,
                "source": "doj_olc",
                "url":    url,
            })
            if len(records) % 50 == 0:
                print(f"  … {len(records):,}", end="\r")
            time.sleep(0.5)
        except Exception:
            continue

    return _write_jsonl(records, output_path)


# ── dataset_info.json registration ───────────────────────────────────────────

def _register(key, filename, data_dir):
    info_path = pathlib.Path(data_dir) / "dataset_info.json"
    data      = json.loads(info_path.read_text()) if info_path.exists() else {}
    data[key] = {
        "file_name":  filename,
        "formatting": "text",
        "columns":    {"prompt": "text"},
    }
    info_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"  Registered '{key}' in dataset_info.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

SOURCES = {
    "courtlistener": download_courtlistener,
    "stackexchange":  download_stackexchange,
    "harvardcap":     download_harvardcap,
    "crs":            download_crs,
    "olc":            download_olc,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download legal training data from free public sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
  courtlistener  4M+ US court opinions  (CourtListener / Free Law Project)
  stackexchange  Law Q&A pairs          (law.stackexchange.com data dump)
  harvardcap     6.7M US cases          (Harvard Caselaw Access Project)
  crs            ~15k policy memos      (Congressional Research Service)
  olc            DOJ legal memoranda    (Office of Legal Counsel)
  all            Run all of the above

Environment variables:
  CL_TOKEN            CourtListener API token (100k/day vs 5k/day)
  HARVARD_CAP_KEY     Harvard CAP API key (required for full text)
        """,
    )
    parser.add_argument("--source",   required=True,
                        choices=list(SOURCES.keys()) + ["all"],
                        help="Data source to download")
    parser.add_argument("--output",   default=None,
                        help="Output .jsonl file or directory (for --source all)")
    parser.add_argument("--max",      type=int, default=50000,
                        help="Max records to download per source (default: 50000)")
    parser.add_argument("--api-key",  default=None,
                        help="API key for sources that require one (Harvard CAP)")
    parser.add_argument("--jurisdiction", default=None,
                        help="Jurisdiction filter for CourtListener/Harvard CAP (e.g. 'us', 'cal')")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help=f"LLaMA-Factory data dir for dataset_info.json (default: {DATA_DIR})")
    parser.add_argument("--register", action="store_true",
                        help="Register downloaded dataset(s) in dataset_info.json")

    args = parser.parse_args()

    sources = list(SOURCES.keys()) if args.source == "all" else [args.source]

    for src in sources:
        if args.source == "all":
            out_dir = pathlib.Path(args.output) if args.output else DATA_DIR
            out     = out_dir / f"{src}.jsonl"
        else:
            out = pathlib.Path(args.output) if args.output else DATA_DIR / f"{src}.jsonl"

        kwargs = {"output_path": out, "max_records": args.max}
        if src == "harvardcap":
            kwargs["api_key"]      = args.api_key
            kwargs["jurisdiction"] = args.jurisdiction or "us"
        elif src == "courtlistener":
            kwargs["jurisdiction"] = args.jurisdiction

        fn   = SOURCES[src]
        path = fn(**kwargs)

        if path and args.register:
            _register(src, str(path.name), args.data_dir)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
