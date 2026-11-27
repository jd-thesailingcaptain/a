#!/usr/bin/env python3
"""run_classic_tools.py - Batch-run Didier Stevens' pdfid.py across the entire PDF corpus.

Runs on: Windows native (C:\research, with (venv) active and pdfid.py in current directory).
Reads:   pdfs/manifest.csv
Writes:  results/classic_results.csv

Usage:
  python run_classic_tools.py --manifest pdfs/manifest.csv --out results/classic_results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ACTIVE_KEYWORDS = ["/JS", "/JavaScript", "/AA", "/OpenAction", "/Launch", "/EmbeddedFile", "/RichMedia"]
ALL_KEYWORDS = [
    "obj", "endobj", "stream", "endstream", "xref", "trailer", "startxref",
    "/Page", "/Encrypt", "/ObjStm", "/JS", "/JavaScript", "/AA", "/OpenAction",
    "/AcroForm", "/JBIG2Decode", "/RichMedia", "/Launch", "/EmbeddedFile", "/XFA", "/URI"
]


def run_pdfid(pdfid_script: Path, pdf_path: Path) -> dict[str, int]:
    proc = subprocess.run(
        [sys.executable, str(pdfid_script), str(pdf_path)],
        capture_output=True, text=True, check=True
    )
    counts = {k: 0 for k in ALL_KEYWORDS}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] in counts:
            # Handle obfuscated count format like "1(1)" if present
            m = re.match(r"(\d+)", parts[1])
            if m:
                counts[parts[0]] = int(m.group(1))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pdfs/manifest.csv")
    ap.add_argument("--pdfid", default="pdfid.py")
    ap.add_argument("--out", default="results/classic_results.csv")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent
    pdfid_path = Path(args.pdfid)
    if not pdfid_path.exists():
        sys.exit(f"Missing {pdfid_path}. Run from C:\\research where pdfid.py is located.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    fieldnames = [
        "filename", "tier", "label", "technique", "variant", "surface",
        "classic_flagged", "active_keyword_sum", "flagged_keywords"
    ] + ALL_KEYWORDS

    out_rows = []
    for idx, row in enumerate(rows, 1):
        sub = "tier1_core" if str(row["tier"]) == "1" else "tier2_stress"
        pdf_path = base_dir / sub / row["filename"]
        counts = run_pdfid(pdfid_path, pdf_path)
        active_hits = [k for k in ACTIVE_KEYWORDS if counts[k] > 0]
        active_sum = sum(counts[k] for k in ACTIVE_KEYWORDS)
        out_row = {
            "filename": row["filename"],
            "tier": row["tier"],
            "label": row["label"],
            "technique": row["technique"],
            "variant": row["variant"],
            "surface": row["surface"],
            "classic_flagged": len(active_hits) > 0,
            "active_keyword_sum": active_sum,
            "flagged_keywords": ";".join(active_hits),
            **counts,
        }
        out_rows.append(out_row)
        if idx % 20 == 0 or idx == len(rows):
            print(f"Processed {idx}/{len(rows)} files with pdfid...")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    flagged_attacks = sum(1 for r in out_rows if r["label"] != "clean" and r["classic_flagged"])
    total_attacks = sum(1 for r in out_rows if r["label"] != "clean")
    print(f"\nSaved {len(out_rows)} rows to {out_path.resolve()}")
    print(f"Active-content flags on attack PDFs: {flagged_attacks}/{total_attacks} (confirms H2 if 0)")


if __name__ == "__main__":
    main()
