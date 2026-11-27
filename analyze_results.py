#!/usr/bin/env python3
"""analyze_results.py - Merge all experiment CSVs against manifest.csv and generate tables + charts.

Runs on: Windows native (C:\research, with (venv) active).
Reads:   pdfs/manifest.csv + any available files in results/
Writes:  results/merged_master.csv
         results/summary_overall_metrics.csv
         results/summary_by_technique.csv
         results/summary_by_surface.csv
         results/summary_tier1_asr.csv (if tier1_results.csv exists)
         results/chart1_tier1_asr.png (if tier1_results.csv exists)
         results/chart2_detector_comparison.png

Usage:
  python analyze_results.py --manifest pdfs/manifest.csv --results results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def compute_binary_metrics(df: pd.DataFrame, pred_col: str, name: str) -> dict:
    # Evaluate on hidden attacks (positive class) vs clean controls (negative class)
    pos = df[df["label"] == "attack_hidden"]
    neg = df[df["label"] == "clean"]
    tp = int(as_bool(pos[pred_col]).sum())
    fn = int(len(pos) - tp)
    fp = int(as_bool(neg[pred_col]).sum())
    tn = int(len(neg) - fp)
    tpr = (tp / len(pos) * 100.0) if len(pos) else 0.0
    fpr = (fp / len(neg) * 100.0) if len(neg) else 0.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else 0.0
    return {
        "detector": name,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Recall_TPR_%": round(tpr, 1),
        "FPR_%": round(fpr, 1),
        "Precision_%": round(prec, 1),
        "F1_%": round(f1, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pdfs/manifest.csv")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    res_dir = Path(args.results)
    res_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)

    detectors = []
    master = manifest.copy()

    classic_path = res_dir / "classic_results.csv"
    if classic_path.exists():
        classic = pd.read_csv(classic_path)[["filename", "classic_flagged", "active_keyword_sum", "flagged_keywords"]]
        master = master.merge(classic, on="filename", how="left")
        detectors.append(("classic_flagged", "Classic Tools (pdfid)"))

    pl_path = res_dir / "phantomlint_results.csv"
    if pl_path.exists():
        pl = pd.read_csv(pl_path)[["filename", "pl_hidden_flagged", "pl_suspicious_flagged", "pl_canary_caught"]]
        master = master.merge(pl, on="filename", how="left")
        detectors.append(("pl_hidden_flagged", "PhantomLint (default)"))

    heur_path = res_dir / "heuristic_results.csv"
    if heur_path.exists():
        heur = pd.read_csv(heur_path).drop(columns=["tier", "label", "technique", "variant", "surface"], errors="ignore")
        master = master.merge(heur, on="filename", how="left")
        detectors.append(("heuristic_flagged", "Heuristic Scanner (Ours)"))

    master.to_csv(res_dir / "merged_master.csv", index=False)

    # 1. Overall Detector Performance Metrics
    if detectors:
        overall_df = pd.DataFrame([compute_binary_metrics(master, col, label) for col, label in detectors])
        overall_df.to_csv(res_dir / "summary_overall_metrics.csv", index=False)
        print("\n=== OVERALL DETECTOR METRICS (Hidden Attacks vs. Clean Controls) ===")
        print(overall_df.to_string(index=False))

        # 2. Detection Rate by Technique
        tech_rows = []
        for (label, tech), grp in master.groupby(["label", "technique"], sort=False):
            row = {"label": label, "technique": tech, "n": len(grp)}
            for col, dname in detectors:
                hits = int(as_bool(grp[col]).sum())
                row[f"{dname} (hits)"] = f"{hits}/{len(grp)}"
                row[f"{dname} (%)"] = round(hits / len(grp) * 100.0, 1)
            tech_rows.append(row)
        tech_df = pd.DataFrame(tech_rows)
        tech_df.to_csv(res_dir / "summary_by_technique.csv", index=False)
        print("\n=== DETECTION BY TECHNIQUE ===")
        print(tech_df.to_string(index=False))

        # 3. Detection Rate by PDF Surface (Hidden Attacks Only)
        hidden_only = master[master["label"] == "attack_hidden"]
        surf_rows = []
        for surf, grp in hidden_only.groupby("surface", sort=False):
            row = {"surface": surf, "n": len(grp)}
            for col, dname in detectors:
                hits = int(as_bool(grp[col]).sum())
                row[f"{dname} (hits)"] = f"{hits}/{len(grp)}"
                row[f"{dname} (%)"] = round(hits / len(grp) * 100.0, 1)
            surf_rows.append(row)
        surf_df = pd.DataFrame(surf_rows)
        surf_df.to_csv(res_dir / "summary_by_surface.csv", index=False)
        print("\n=== DETECTION BY PDF SURFACE (RQ3 Scanned-Surface Comparison) ===")
        print(surf_df.to_string(index=False))

        # Chart 2: Detector Comparison by Surface
        fig, ax = plt.subplots(figsize=(9, 4.8))
        surfaces = surf_df["surface"].tolist()
        x = np.arange(len(surfaces))
        width = 0.8 / len(detectors)
        palette = ["#8c969e", "#2b5c8f", "#1b8a5a"]

        for i, (_, dname) in enumerate(detectors):
            vals = surf_df[f"{dname} (%)"].tolist()
            bars = ax.bar(x + (i - (len(detectors) - 1) / 2) * width, vals, width,
                          label=dname, color=palette[i % len(palette)])
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.0f}%", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Detection Rate / Recall (%)")
        ax.set_title("Hidden Prompt Injection Detection Rate by PDF Surface")
        ax.set_xticks(x)
        ax.set_xticklabels(surfaces, rotation=15, ha="right")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(res_dir / "chart2_detector_comparison.png", dpi=200)
        plt.close(fig)

    # 4. Tier 1 LLM Attack Success Rate (ASR)
    t1_path = res_dir / "tier1_results.csv"
    if t1_path.exists():
        t1 = pd.read_csv(t1_path)
        t1["obeyed"] = t1["outcome"] == "obeyed"
        t1["refused"] = t1["outcome"] == "refused_or_unparseable"
        t1["canary_hit"] = as_bool(t1["canary_hit"])
        t1["payload_extracted"] = as_bool(t1["payload_extracted"])

        asr_df = (
            t1.groupby(["label", "technique", "surface", "extractor"], sort=False)
            .agg(
                runs=("run", "count"),
                extracted=("payload_extracted", "mean"),
                obeyed_n=("obeyed", "sum"),
                asr_pct=("obeyed", lambda s: round(s.mean() * 100.0, 1)),
                canary_pct=("canary_hit", lambda s: round(s.mean() * 100.0, 1)),
                refused_pct=("refused", lambda s: round(s.mean() * 100.0, 1)),
            )
            .reset_index()
        )
        asr_df.to_csv(res_dir / "summary_tier1_asr.csv", index=False)
        print("\n=== TIER 1 LLM ATTACK SUCCESS RATE BY TECHNIQUE & EXTRACTOR ===")
        print(asr_df.to_string(index=False))

        # Chart 1: Tier 1 ASR by Technique & Extractor
        attacks_asr = asr_df[asr_df["label"] == "attack_hidden"]
        piv = attacks_asr.pivot(index="technique", columns="extractor", values="asr_pct").fillna(0)
        ax = piv.plot(kind="bar", figsize=(10, 5), width=0.78, color=["#355c7d", "#6c5b7b", "#f67280"])
        ax.set_ylabel("Attack Success Rate (% Runs Obeyed)")
        ax.set_xlabel("Hiding Technique (Tier 1 Core Corpus)")
        ax.set_title("Tier 1 LLM Resume Screener Attack Success Rate by Technique & Extractor")
        ax.set_ylim(0, 115)
        plt.xticks(rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="PDF Extractor")
        plt.tight_layout()
        plt.savefig(res_dir / "chart1_tier1_asr.png", dpi=200)
        plt.close()

    print(f"\nAll summaries and charts saved in {res_dir.resolve()}")


if __name__ == "__main__":
    main()
