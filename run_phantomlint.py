#!/usr/bin/env python3
"""run_phantomlint.py - Batch-run PhantomLint across the entire PDF corpus.

Runs in: WSL Ubuntu (~/phantom-lint, with (venv) active).
Reads:   /mnt/c/research/pdfs/manifest.csv
Writes:  /mnt/c/research/results/phantomlint_results.csv

Usage:
  python run_phantomlint.py --manifest /mnt/c/research/pdfs/manifest.csv \
                            --out /mnt/c/research/results/phantomlint_results.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/mnt/c/research/pdfs/manifest.csv")
    ap.add_argument("--out", default="/mnt/c/research/results/phantomlint_results.csv")
    ap.add_argument("--analyze", default="nlp", choices=["nlp", "passthrough", "llm-guard"])
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    done = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add(r["filename"])
        print(f"Resuming: {len(done)}/{len(rows)} files already scanned.")

    fieldnames = [
        "filename", "tier", "label", "technique", "variant", "surface", "analyze_mode",
        "pl_hidden_flagged", "pl_suspicious_flagged", "pl_canary_caught",
        "runtime_s", "exit_code", "hidden_excerpt"
    ]

    mode = "a" if (args.resume and out_path.exists()) else "w"
    tmp_root = Path(tempfile.mkdtemp(prefix="phantomlint_batch_"))

    try:
        with open(out_path, mode, newline="", encoding="utf-8") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()

            for idx, row in enumerate(rows, 1):
                fname = row["filename"]
                if fname in done:
                    continue
                sub = "tier1_core" if str(row["tier"]) == "1" else "tier2_stress"
                pdf_path = base_dir / sub / fname
                run_dir = tmp_root / pdf_path.stem
                if run_dir.exists():
                    shutil.rmtree(run_dir)

                t0 = time.time()
                proc = subprocess.run(
                    [
                        "phantomlint", str(pdf_path),
                        "--output", str(run_dir),
                        "--analyze", args.analyze,
                        "--threshold", str(args.threshold),
                    ],
                    capture_output=True, text=True
                )
                dt = round(time.time() - t0, 2)

                hidden_file = run_dir / "hidden_suspicious_phrases.txt"
                susp_file = run_dir / "suspicious_phrases.txt"
                hidden_txt = hidden_file.read_text(encoding="utf-8", errors="replace").strip() if hidden_file.exists() else ""
                susp_txt = susp_file.read_text(encoding="utf-8", errors="replace").strip() if susp_file.exists() else ""

                canary = row.get("canary", "")
                canary_caught = bool(canary) and (squash(canary) in squash(hidden_txt))

                out_row = {
                    "filename": fname,
                    "tier": row["tier"],
                    "label": row["label"],
                    "technique": row["technique"],
                    "variant": row["variant"],
                    "surface": row["surface"],
                    "analyze_mode": args.analyze,
                    "pl_hidden_flagged": len(hidden_txt) > 0,
                    "pl_suspicious_flagged": len(susp_txt) > 0,
                    "pl_canary_caught": canary_caught,
                    "runtime_s": dt,
                    "exit_code": proc.returncode,
                    "hidden_excerpt": hidden_txt.replace("\n", " ")[:220],
                }
                writer.writerow(out_row)
                out_fh.flush()
                shutil.rmtree(run_dir, ignore_errors=True)
                print(
                    f"[{idx:3d}/{len(rows)}] {fname} -> hidden={out_row['pl_hidden_flagged']} "
                    f"susp={out_row['pl_suspicious_flagged']} ({dt}s)"
                )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nDone. Saved PhantomLint results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
