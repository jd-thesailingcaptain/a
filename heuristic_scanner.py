#!/usr/bin/env python3
"""heuristic_scanner.py - Multi-surface static PDF anomaly detector for hidden prompt injections.

Inspects both visual layout attributes (PyMuPDF) and structural PDF objects (pikepdf) across
surfaces outside PhantomLint's text-block scope (metadata, annotations, OCG layers, ToUnicode CMaps).

Runs on: Windows native (C:\research, with (venv) active).
Reads:   pdfs/manifest.csv
Writes:  results/heuristic_results.csv

Usage:
  python heuristic_scanner.py --manifest pdfs/manifest.csv --out results/heuristic_results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import pikepdf

try:
    import pymupdf as mu
except ImportError:
    import fitz as mu

INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|earlier)\s+(instructions|guidance|prompts?)"
    r"|system\s+notice|note\s+to\s+(the\s+)?ai|well\s+qualified"
    r"|recommend\s+hiring|score\s*10|decision\s*[:=]?\s*hire|canary-\d+)",
    re.IGNORECASE,
)


def int_to_rgb(color_int: int) -> tuple[int, int, int]:
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return r, g, b


def white_distance(rgb: tuple[int, int, int]) -> float:
    return math.sqrt(sum((255 - c) ** 2 for c in rgb))


def rects_overlap_ratio(text_rect: mu.Rect, cover_rect: mu.Rect) -> float:
    inter = text_rect & cover_rect
    if inter.is_empty or text_rect.get_area() <= 0:
        return 0.0
    return inter.get_area() / text_rect.get_area()


def scan_pdf(pdf_path: Path) -> dict:
    flags = {
        "flag_low_contrast": False,
        "flag_tiny_font": False,
        "flag_render_mode_3": False,
        "flag_offpage": False,
        "flag_occluded": False,
        "flag_hidden_ocg": False,
        "flag_meta_annot": False,
        "flag_cmap_mismatch": False,
    }
    reasons = []

    # --- Pass 1: Layout & Visual Geometry via PyMuPDF ---
    with mu.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, 1):
            crop = page.cropbox
            media = page.mediabox
            # Collect image bounding boxes on the page for Z-order occlusion checks
            image_rects = []
            for img_info in page.get_image_info(xrefs=True):
                bbox = img_info.get("bbox")
                if bbox:
                    image_rects.append(mu.Rect(bbox))

            text_dict = page.get_text("dict", flags=mu.TEXTFLAGS_SEARCH)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if not txt:
                            continue
                        size = float(span.get("size", 12.0))
                        rgb = int_to_rgb(int(span.get("color", 0)))
                        bbox = mu.Rect(span["bbox"])

                        # 1. Near-white / low-contrast ink (distance < 60 from pure white covers d <= 32)
                        if white_distance(rgb) < 60.0 and len(txt) >= 4:
                            flags["flag_low_contrast"] = True
                            reasons.append(f"p{page_num}:low_contrast_rgb{rgb}")

                        # 2. Sub-threshold font size (< 4.5 pt avoids 5-6 pt footers, catches <= 4 pt)
                        if size < 4.5 and len(txt) >= 4:
                            # Allow benign tiny revision stamps like "rev 3 - 2026-11" if short
                            if size <= 4.0 and (len(txt) > 20 or INJECTION_PATTERNS.search(txt)):
                                flags["flag_tiny_font"] = True
                                reasons.append(f"p{page_num}:tiny_font_{size:.2f}pt")

                        # 3. Outside CropBox or MediaBox
                        visible_box = crop & media
                        inter = bbox & visible_box
                        if inter.is_empty or (bbox.get_area() > 0 and inter.get_area() / bbox.get_area() < 0.25):
                            flags["flag_offpage"] = True
                            reasons.append(f"p{page_num}:offpage_bbox")

                        # 4. Occluded behind embedded image
                        for irect in image_rects:
                            if rects_overlap_ratio(bbox, irect) > 0.50:
                                flags["flag_occluded"] = True
                                reasons.append(f"p{page_num}:behind_image")

    # --- Pass 2: Low-Level PDF Structure via pikepdf ---
    with pikepdf.open(str(pdf_path)) as pdf:
        # 5. Text rendering mode 3 (invisible text: "3 Tr") in content streams
        for page_num, page in enumerate(pdf.pages, 1):
            contents = page.get("/Contents")
            if contents is not None:
                streams = contents if isinstance(contents, pikepdf.Array) else [contents]
                for st in streams:
                    try:
                        raw = st.read_bytes().decode("latin-1", errors="ignore")
                        if re.search(r"(?:^|\s)3\s+Tr(?:\s|$)", raw):
                            flags["flag_render_mode_3"] = True
                            reasons.append(f"p{page_num}:render_mode_3")
                    except Exception:
                        pass

        # 6. Hidden Optional Content Groups (OCG /OFF or /BaseState /OFF)
        oc_props = pdf.Root.get("/OCProperties")
        if oc_props is not None:
            default_cfg = oc_props.get("/D")
            if default_cfg is not None:
                off_list = default_cfg.get("/OFF")
                base_state = str(default_cfg.get("/BaseState", "/ON"))
                if (off_list is not None and len(off_list) > 0) or base_state == "/OFF":
                    flags["flag_hidden_ocg"] = True
                    reasons.append("ocg_hidden_layer")

        # 7. Metadata (/Info + XMP) and Annotations (/Annots)
        suspicious_meta_values = []
        if pdf.docinfo:
            for k, v in pdf.docinfo.items():
                val = str(v).strip()
                if len(val) > 80 or INJECTION_PATTERNS.search(val):
                    suspicious_meta_values.append(f"docinfo{k}")

        try:
            with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as meta:
                for k, v in meta.items():
                    val = str(v).strip()
                    if len(val) > 80 or INJECTION_PATTERNS.search(val):
                        suspicious_meta_values.append(f"xmp:{k}")
        except Exception:
            pass

        for page_num, page in enumerate(pdf.pages, 1):
            annots = page.get("/Annots")
            if annots:
                for annot in annots:
                    subtype = str(annot.get("/Subtype", ""))
                    # Ignore normal clickable URI links with no /Contents payload
                    for field_key in ("/Contents", "/V", "/TU"):
                        val = str(annot.get(field_key, "")).strip()
                        if val and (len(val) > 60 or INJECTION_PATTERNS.search(val)):
                            suspicious_meta_values.append(f"p{page_num}:annot{subtype}{field_key}")

        if suspicious_meta_values:
            flags["flag_meta_annot"] = True
            reasons.extend(suspicious_meta_values)

        # 8. ToUnicode CMap mismatch (rendered glyph name != mapped Unicode character)
        for page_num, page in enumerate(pdf.pages, 1):
            res = page.get("/Resources")
            fonts = res.get("/Font") if res else None
            if not fonts:
                continue
            for fname, fobj in fonts.items():
                enc = fobj.get("/Encoding")
                tounicode = fobj.get("/ToUnicode")
                if enc is None or tounicode is None or not isinstance(enc, pikepdf.Dictionary):
                    continue
                diffs = enc.get("/Differences")
                if not diffs:
                    continue
                # Build code -> glyph char map from /Differences
                glyph_map = {}
                cur_code = None
                simple_glyphs = {"space": " ", "period": ".", "comma": ",", "hyphen": "-"}
                for item in diffs:
                    if isinstance(item, int):
                        cur_code = int(item)
                    elif cur_code is not None:
                        gname = str(item).lstrip("/")
                        glyph_map[cur_code] = simple_glyphs.get(gname, gname if len(gname) == 1 else "?")
                        cur_code += 1
                # Parse bfchar pairs from ToUnicode stream
                cmap_raw = tounicode.read_bytes().decode("latin-1", errors="ignore")
                mismatches = 0
                for m in re.finditer(r"<([0-9A-Fa-f]{2})>\s*<([0-9A-Fa-f]{4})>", cmap_raw):
                    code = int(m.group(1), 16)
                    uni_char = chr(int(m.group(2), 16))
                    if code in glyph_map and glyph_map[code] != "?" and glyph_map[code] != uni_char:
                        mismatches += 1
                if mismatches >= 5:
                    flags["flag_cmap_mismatch"] = True
                    reasons.append(f"p{page_num}:cmap_mismatch({mismatches}_chars)")

    # Deduplicate reasons while preserving order
    unique_reasons = list(dict.fromkeys(reasons))
    triggered = sum(1 for v in flags.values() if v)
    return {
        "heuristic_flagged": triggered > 0,
        "rules_triggered": triggered,
        **flags,
        "reasons": ";".join(unique_reasons),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pdfs/manifest.csv")
    ap.add_argument("--out", default="results/heuristic_results.csv")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out_rows = []
    for row in rows:
        sub = "tier1_core" if str(row["tier"]) == "1" else "tier2_stress"
        pdf_path = base_dir / sub / row["filename"]
        res = scan_pdf(pdf_path)
        out_rows.append({
            "filename": row["filename"],
            "tier": row["tier"],
            "label": row["label"],
            "technique": row["technique"],
            "variant": row["variant"],
            "surface": row["surface"],
            **res,
        })

    fieldnames = list(out_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    hidden_attacks = [r for r in out_rows if r["label"] == "attack_hidden"]
    cleans = [r for r in out_rows if r["label"] == "clean"]
    tp = sum(1 for r in hidden_attacks if r["heuristic_flagged"])
    fp = sum(1 for r in cleans if r["heuristic_flagged"])
    print(f"Scanned {len(out_rows)} PDFs -> {out_path.resolve()}")
    print(f"Hidden attacks caught: {tp}/{len(hidden_attacks)} ({100*tp/max(len(hidden_attacks),1):.1f}%)")
    print(f"False positives on clean controls: {fp}/{len(cleans)} ({100*fp/max(len(cleans),1):.1f}%)")


if __name__ == "__main__":
    main()
