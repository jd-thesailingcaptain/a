#!/usr/bin/env python3
r"""analyze_results.py - Merge all experiment CSVs against manifest.csv and generate tables + charts.

Runs on: Windows native (C:\research, with (venv) active).
Reads:   pdfs/manifest.csv + any available files in results/
Writes:  results/merged_master.csv
         results/summary_overall_metrics.csv
         results/summary_by_technique.csv
         results/summary_by_surface.csv
         results/summary_rq3_ablation.csv (if both NLP and passthrough CSVs exist)
         results/summary_tier1_asr.csv (if tier1_results.csv exists)
         results/chart1_tier1_asr.png (if tier1_results.csv exists)
         results/chart2_detector_comparison.png
         results/chart3_rq3_ablation.png (if both NLP and passthrough CSVs exist)

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


def clean_filename(df: pd.DataFrame) -> pd.DataFrame:
    if "filename" in df.columns:
        df["filename"] = df["filename"].astype(str).str.strip()
    return df


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
    manifest = clean_filename(pd.read_csv(args.manifest))

    detectors = []
    master = manifest.copy()

    classic_path = res_dir / "classic_results.csv"
    if classic_path.exists():
        classic_raw = clean_filename(pd.read_csv(classic_path))
        cols = [c for c in ["filename", "classic_flagged", "active_keyword_sum", "flagged_keywords"] if c in classic_raw.columns]
        master = master.merge(classic_raw[cols], on="filename", how="left")
        detectors.append(("classic_flagged", "Classic Tools (pdfid)"))

    pl_path = res_dir / "phantomlint_results.csv"
    has_pl_nlp = pl_path.exists()
    if has_pl_nlp:
        pl_raw = clean_filename(pd.read_csv(pl_path))
        cols = [c for c in ["filename", "pl_hidden_flagged", "pl_suspicious_flagged", "pl_canary_caught"] if c in pl_raw.columns]
        master = master.merge(pl_raw[cols], on="filename", how="left")
        detectors.append(("pl_hidden_flagged", "PhantomLint (NLP)"))

    pl_pass_path = res_dir / "phantomlint_passthrough.csv"
    has_pl_pass = pl_pass_path.exists()
    if has_pl_pass:
        plp_raw = clean_filename(pd.read_csv(pl_pass_path))
        rename_map = {
            "pl_hidden_flagged": "pl_pass_hidden_flagged",
            "pl_suspicious_flagged": "pl_pass_suspicious_flagged",
            "pl_canary_caught": "pl_pass_canary_caught",
        }
        keep_cols = ["filename"] + [c for c in rename_map if c in plp_raw.columns]
        plp = plp_raw[keep_cols].rename(columns=rename_map)
        master = master.merge(plp, on="filename", how="left")
        detectors.append(("pl_pass_hidden_flagged", "PhantomLint (Passthrough)"))

    heur_path = res_dir / "heuristic_results.csv"
    if heur_path.exists():
        heur = clean_filename(pd.read_csv(heur_path)).drop(
            columns=["tier", "label", "technique", "variant", "surface"], errors="ignore"
        )
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
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        surfaces = surf_df["surface"].tolist()
        x = np.arange(len(surfaces))
        width = 0.8 / len(detectors)
        palette = ["#8c969e", "#2b5c8f", "#e07a2f", "#1b8a5a"]

        for i, (_, dname) in enumerate(detectors):
            vals = surf_df[f"{dname} (%)"].tolist()
            bars = ax.bar(
                x + (i - (len(detectors) - 1) / 2) * width,
                vals,
                width,
                label=dname,
                color=palette[i % len(palette)],
            )
            for b in bars:
                h = b.get_height()
                ax.annotate(
                    f"{h:.0f}%",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax.set_ylabel("Detection Rate / Recall (%)")
        ax.set_title("Hidden Prompt Injection Detection Rate by PDF Surface")
        ax.set_xticks(x)
        ax.set_xticklabels(surfaces, rotation=15, ha="right")
        ax.set_ylim(0, 118)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, ncol=2 if len(detectors) >= 4 else 1)
        plt.tight_layout()
        plt.savefig(res_dir / "chart2_detector_comparison.png", dpi=200)
        plt.close(fig)

    # 3b. RQ3 Ablation: PhantomLint NLP vs. Passthrough (Hidden Attacks Only)
    if has_pl_nlp and has_pl_pass:
        hidden_only = master[master["label"] == "attack_hidden"].copy()
        hidden_only["_nlp"] = as_bool(hidden_only["pl_hidden_flagged"])
        hidden_only["_pass"] = as_bool(hidden_only["pl_pass_hidden_flagged"])

        abl_rows = []
        for (tech, surf), grp in hidden_only.groupby(["technique", "surface"], sort=False):
            n = len(grp)
            nlp_hit = int(grp["_nlp"].sum())
            pass_hit = int(grp["_pass"].sum())
            both_hit = int((grp["_nlp"] & grp["_pass"]).sum())
            nlp_gate_loss = int((grp["_pass"] & ~grp["_nlp"]).sum())
            ocr_miss = int((~grp["_pass"] & ~grp["_nlp"]).sum())
            abl_rows.append({
                "technique": tech,
                "surface": surf,
                "n": n,
                "NLP_caught": nlp_hit,
                "NLP_%": round(nlp_hit / n * 100.0, 1),
                "Passthrough_caught": pass_hit,
                "Passthrough_%": round(pass_hit / n * 100.0, 1),
                "NLP_gate_loss_n": nlp_gate_loss,
                "NLP_gate_loss_%": round(nlp_gate_loss / n * 100.0, 1),
                "OCR_or_surface_miss_n": ocr_miss,
                "OCR_or_surface_miss_%": round(ocr_miss / n * 100.0, 1),
                "_both_%": round(both_hit / n * 100.0, 1),
            })

        n_tot = len(hidden_only)
        nlp_tot = int(hidden_only["_nlp"].sum())
        pass_tot = int(hidden_only["_pass"].sum())
        both_tot = int((hidden_only["_nlp"] & hidden_only["_pass"]).sum())
        loss_tot = int((hidden_only["_pass"] & ~hidden_only["_nlp"]).sum())
        miss_tot = int((~hidden_only["_pass"] & ~hidden_only["_nlp"]).sum())
        abl_rows.append({
            "technique": "OVERALL",
            "surface": "all",
            "n": n_tot,
            "NLP_caught": nlp_tot,
            "NLP_%": round(nlp_tot / n_tot * 100.0, 1) if n_tot else 0.0,
            "Passthrough_caught": pass_tot,
            "Passthrough_%": round(pass_tot / n_tot * 100.0, 1) if n_tot else 0.0,
            "NLP_gate_loss_n": loss_tot,
            "NLP_gate_loss_%": round(loss_tot / n_tot * 100.0, 1) if n_tot else 0.0,
            "OCR_or_surface_miss_n": miss_tot,
            "OCR_or_surface_miss_%": round(miss_tot / n_tot * 100.0, 1) if n_tot else 0.0,
            "_both_%": round(both_tot / n_tot * 100.0, 1) if n_tot else 0.0,
        })

        abl_df = pd.DataFrame(abl_rows)
        abl_export = abl_df.drop(columns=["_both_%"])
        abl_export.to_csv(res_dir / "summary_rq3_ablation.csv", index=False)
        print("\n=== RQ3 ABLATION: PHANTOMLINT NLP GATE VS. OCR PASSTHROUGH ===")
        print(abl_export.to_string(index=False))

        # Chart 3: Stacked Attribution of PhantomLint Detection & Misses by Technique
        plot_df = abl_df[abl_df["technique"] != "OVERALL"].copy()
        x = np.arange(len(plot_df))
        both_pct = plot_df["_both_%"].to_numpy()
        gate_pct = plot_df["NLP_gate_loss_%"].to_numpy()
        miss_pct = plot_df["OCR_or_surface_miss_%"].to_numpy()

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.bar(x, both_pct, label="Caught by PhantomLint (NLP)", color="#2b5c8f")
        ax.bar(x, gate_pct, bottom=both_pct, label="Lost to NLP Classifier Gate (Caught in Passthrough)", color="#e07a2f")
        ax.bar(x, miss_pct, bottom=both_pct + gate_pct, label="Missed by OCR / Surface Blind Spot", color="#c94c4c")

        ax.set_ylabel("Share of Hidden Attacks (%)")
        ax.set_xlabel("Hiding Technique")
        ax.set_title("RQ3 Ablation: PhantomLint NLP Classifier Bottleneck vs. OCR/Surface Blind Spots")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["technique"].tolist(), rotation=25, ha="right")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        plt.tight_layout()
        plt.savefig(res_dir / "chart3_rq3_ablation.png", dpi=200, bbox_inches="tight")
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