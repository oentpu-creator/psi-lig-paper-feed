"""
fetch_papers.py
===============
Daily crawler: search Semantic Scholar + arXiv for PSi / LIG papers,
append new ones to data/papers.json, and write a stats summary.

Designed to run inside GitHub Actions (no local dependencies).
Only stdlib + requests required.

Usage:
    python crawler/fetch_papers.py
    python crawler/fetch_papers.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "papers.json"
SEEN_FILE = ROOT / "data" / "seen_ids.json"

# ── Search queries ────────────────────────────────────────────────────────────

QUERIES = [
    # PSi
    "porous silicon optical sensor",
    "porous silicon photoluminescence",
    "porous silicon biosensor",
    "porous silicon nanostructure",
    "porous silicon reflectance",
    "porous silicon drug delivery",
    # LIG
    "laser induced graphene",
    "laser-induced graphene flexible",
    "laser induced graphene sensor",
    "laser induced graphene supercapacitor",
    "LIG polyimide strain sensor",
    "laser scribed graphene electrode",
    "laser induced graphene wearable",
]

MAX_PER_QUERY = int(os.getenv("MAX_PER_QUERY", "20"))
SS_API_KEY    = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")   # optional, raises rate limit


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def _ss_search(query: str, limit: int = 20) -> list[dict]:
    try:
        import requests
    except ImportError:
        return []

    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,authors,year,abstract,externalIds,openAccessPdf,journal,publicationVenue"
    headers = {"x-api-key": SS_API_KEY} if SS_API_KEY else {}

    for attempt in range(3):
        try:
            resp = requests.get(url, params={"query": query, "limit": limit, "fields": fields},
                                headers=headers, timeout=30)
            if resp.status_code == 429:
                if not SS_API_KEY:
                    print("    [SS] rate-limited (no API key) — skipping query")
                    return []
                wait = 60 * (attempt + 1)
                print(f"    [SS] rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return data
        except Exception as exc:
            print(f"    [SS] error: {exc}")
            time.sleep(5)
    return []


def _parse_ss(item: dict) -> dict | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None
    authors = [a.get("name", "") for a in item.get("authors", [])]
    ext     = item.get("externalIds") or {}
    doi     = ext.get("DOI", "")
    pdf_url = (item.get("openAccessPdf") or {}).get("url", "")
    journal_obj = item.get("journal") or {}
    venue       = item.get("publicationVenue") or {}
    journal = journal_obj.get("name") or venue.get("name") or ""

    ss_id    = item.get("paperId", "")
    paper_id = f"ss_{ss_id[:16]}" if ss_id else f"ss_doi_{doi[:20]}"

    return {
        "paper_id":   paper_id,
        "source":     "semantic_scholar",
        "title":      title,
        "authors":    authors[:6],
        "year":       item.get("year") or 0,
        "abstract":   (item.get("abstract") or "")[:800],
        "doi":        doi,
        "pdf_url":    pdf_url,
        "journal":    journal,
        "url":        f"https://www.semanticscholar.org/paper/{ss_id}" if ss_id else "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ── arXiv ─────────────────────────────────────────────────────────────────────

def _arxiv_search(query: str, max_results: int = 20) -> list[dict]:
    try:
        import requests
        import xml.etree.ElementTree as ET
    except ImportError:
        return []

    url    = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "max_results": max_results,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(3)   # ArXiv rate limit: 3s between requests
    except Exception as exc:
        print(f"    [arXiv] error: {exc}")
        return []

    NS   = {"atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.text)
    results = []

    for entry in root.findall("atom:entry", NS):
        def _t(el): return (el.text or "").strip() if el is not None else ""

        title    = _t(entry.find("atom:title", NS)).replace("\n", " ")
        abstract = _t(entry.find("atom:summary", NS)).replace("\n", " ")[:800]
        year_str = _t(entry.find("atom:published", NS))[:4]
        arxiv_url = _t(entry.find("atom:id", NS))
        arxiv_id  = arxiv_url.split("/abs/")[-1]
        paper_id  = f"arxiv_{arxiv_id.replace('/', '_')}"

        authors = [_t(a.find("atom:name", NS)) for a in entry.findall("atom:author", NS)]

        doi_el  = entry.find("arxiv:doi", NS)
        doi     = _t(doi_el) if doi_el is not None else ""

        # Only keep arXiv papers that have a DOI (published in a journal)
        if not doi:
            continue

        journal_el = entry.find("arxiv:journal_ref", NS)
        journal    = _t(journal_el) if journal_el is not None else "arXiv"

        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break

        results.append({
            "paper_id":   paper_id,
            "source":     "arxiv",
            "title":      title,
            "authors":    authors[:6],
            "year":       int(year_str) if year_str.isdigit() else 0,
            "abstract":   abstract,
            "doi":        doi,
            "pdf_url":    pdf_url,
            "journal":    journal,
            "url":        arxiv_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    return results


# ── Dedup ─────────────────────────────────────────────────────────────────────

def _norm_title(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()


def _dedup(records: list[dict]) -> list[dict]:
    seen_doi: set[str]   = set()
    seen_title: set[str] = set()
    out: list[dict]      = []
    for r in records:
        if r["doi"] and r["doi"] in seen_doi:
            continue
        nt = _norm_title(r["title"])
        if nt and nt in seen_title:
            continue
        if r["doi"]:
            seen_doi.add(r["doi"])
        if nt:
            seen_title.add(nt)
        out.append(r)
    return out


# ── Material tag ──────────────────────────────────────────────────────────────

def _tag_material(rec: dict) -> str:
    text = (rec["title"] + " " + rec["abstract"]).lower()
    has_psi = any(w in text for w in ["porous silicon", " psi", "por-si"])
    has_lig = any(w in text for w in ["laser induced graphene", " lig ",
                                       "laser-induced graphene", "laser scribing"])
    if has_psi and has_lig:
        return "hybrid"
    if has_psi:
        return "PSi"
    if has_lig:
        return "LIG"
    return "other"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing papers
    existing: list[dict] = []
    if DATA_FILE.exists():
        existing = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    seen_ids:  set[str] = {p["paper_id"] for p in existing}
    seen_dois: set[str] = {p["doi"]      for p in existing if p.get("doi")}

    print(f"Existing papers: {len(existing)}")
    print(f"Fetching from {len(QUERIES)} queries × 2 sources...")

    all_fetched: list[dict] = []

    for qi, query in enumerate(QUERIES, 1):
        print(f"\n[{qi}/{len(QUERIES)}] {query}")

        ss_items = _ss_search(query, MAX_PER_QUERY)
        parsed   = [_parse_ss(i) for i in ss_items]
        ss_recs  = [p for p in parsed if p]
        print(f"  SS: {len(ss_recs)} records")
        all_fetched.extend(ss_recs)
        time.sleep(2)

        arx_recs = _arxiv_search(query, MAX_PER_QUERY)
        print(f"  arXiv: {len(arx_recs)} records (with DOI)")
        all_fetched.extend(arx_recs)

    # Dedup all fetched
    all_fetched = _dedup(all_fetched)
    print(f"\nTotal unique fetched: {len(all_fetched)}")

    # Keep only genuinely new ones
    new_records = []
    for r in all_fetched:
        if r["paper_id"] in seen_ids:
            continue
        if r["doi"] and r["doi"] in seen_dois:
            continue
        r["material"] = _tag_material(r)
        new_records.append(r)

    print(f"New papers (not in DB): {len(new_records)}")

    if dry_run:
        for r in new_records[:5]:
            print(f"  [{r['year']}] {r['title'][:70]} ({r['material']})")
        print("Dry-run — nothing written.")
        return

    if new_records:
        merged = existing + new_records
        DATA_FILE.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {len(merged)} papers to {DATA_FILE}")
    else:
        print("No new papers found today.")

    # Write stats file for GitHub Actions summary
    stats = {
        "last_run":    datetime.now(timezone.utc).isoformat(),
        "total":       len(existing) + len(new_records),
        "new_today":   len(new_records),
        "by_material": {},
    }
    all_papers = existing + new_records
    for p in all_papers:
        mat = p.get("material", "other")
        stats["by_material"][mat] = stats["by_material"].get(mat, 0) + 1

    (DATA_FILE.parent / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
