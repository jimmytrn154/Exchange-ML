#!/usr/bin/env python3
"""Rebuild the paper from completed evidence; never train, infer, or fit scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
HELDOUT = ROOT / "outputs/stage3_h1_v2/heldout"
SENSITIVITY = ROOT / "outputs/evaluator_sensitivity_v1"
MANIFESTS = (
    (ROOT / "outputs/stage3_h1_v2", "e1_manifest.json"),
    (ROOT / "outputs/stage3_h1_v2/perturbations", "e2_manifest.json"),
    (HELDOUT, "e3_manifest.json"),
    (ROOT / "outputs/stage3_h1_v2/e4_e5", "e4_e5_manifest.json"),
    (ROOT / "outputs/stage3_h1_v2/e5_supplement_v1", "e5_supplement_manifest.json"),
    (SENSITIVITY, "sensitivity_manifest.json"),
)
SCORE_NAMES = {
    "mean_entropy": "Mean entropy",
    "p95_entropy": "95th-percentile entropy",
    "pairwise_dice": "Pairwise Dice disagreement",
    "presence": r"Coverage $A$ (Jaccard)",
    "count_variance": "Instance-count variance",
    "union_nvi": "Union-normalized VI",
    "hungarian": "Hungarian matching",
    "fa_v1": r"Combined $D_{\rm vox}$",
    "fa_v2": r"Combined $D_{\rm joint}$",
    "q_joint_unweighted": r"Grouping $Q_{\rm joint}$",
    "partition_v2": r"Grouping $wQ_{\rm joint}$",
    "partition_v1": r"Grouping $wQ_{\rm vox}$",
}
SCORE_GROUPS = (
    ("Voxel uncertainty", ("mean_entropy", "p95_entropy")),
    ("Foreground overlap", ("pairwise_dice", "presence")),
    ("Instance level", ("count_variance", "union_nvi", "hungarian")),
    ("Coverage/grouping decomposition", ("fa_v1", "fa_v2", "q_joint_unweighted",
                                         "partition_v2", "partition_v1")),
)
ENDPOINT_NAMES = {
    "original_merge_split": "Merge/split (primary)",
    "one_to_one_one_minus_f1": r"One-to-one matched $1-F_1$",
    "significant_0p01_merge_split": r"Merge/split, overlap $\geq 1\%$",
    "significant_0p1_merge_split": r"Merge/split, overlap $\geq 10\%$",
    "missing_volume": "Missing volume",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_hashes(base: Path, hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        path = base / name  # Path also handles absolute external data paths.
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed evidence: {path}")


def verify_manifest(folder: Path, filename: str) -> None:
    manifest = json.loads((folder / filename).read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete experiment: {folder / filename}")
    verify_hashes(folder, manifest["output_sha256"])
    verify_hashes(folder, manifest.get("cache_sha256", {}))


def table_end() -> list[str]:
    return [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def random_aurc(split: str, risk: str) -> float:
    rows = [row for row in read_rows(ROOT / f"outputs/stage3_h1_v2/{split}/comparison.csv")
            if row["stratum"] == "pooled" and row["risk_source"] == "member_mean"
            and row["risk"] == risk]
    values = {round(float(row["random_aurc"]), 12) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Random AURC is not unique for {split}/{risk}")
    return values.pop()


def build_main_table() -> None:
    records = {row["score"]: row for row in read_rows(HELDOUT / "main_table.csv")}
    if len(records) != 12 or set(records) != set(SCORE_NAMES):
        raise ValueError("Expected exactly the 12 fixed score controls")
    if sorted(k for _, keys in SCORE_GROUPS for k in keys) != sorted(SCORE_NAMES):
        raise ValueError("Every score must appear in exactly one group")
    columns = ("validation_merge_split_aurc", "heldout_merge_split_aurc",
               "heldout_one_minus_f1_aurc")
    best = {c: min(float(row[c]) for row in records.values()) for c in columns}
    table = [
        r"\begin{table}[t]",
        r"\caption{Structural-failure ranking by all 12 prediction-only scores (AURC, lower is better). "
        r"M+S: merge-plus-split risk per reference fragment; $1-F_1$: instance detection error. "
        r"Dev.: 80 bones from 40 development CTs; Held-out: 200 bones from 100 separate CTs. "
        r"Random: expected AURC of a random ranking. Best value per column in bold.}",
        r"\label{tab:main}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccc}",
        r"\toprule", r" & Dev. & \multicolumn{2}{c}{Held-out} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}",
        r"Score & M+S & M+S & $1-F_1$ \\", r"\midrule",
        r"Random ranking & " + " & ".join(f"{v:.3f}" for v in (
            random_aurc("validation", "merge_split"), random_aurc("heldout", "merge_split"),
            random_aurc("heldout", "one_minus_f1"))) + r" \\",
    ]
    for group, keys in SCORE_GROUPS:
        table.append(r"\midrule")
        table.append(r"\multicolumn{4}{l}{\textit{" + group + r"}} \\")
        for key in keys:
            cells = []
            for c in columns:
                value = float(records[key][c])
                text = f"{value:.3f}"
                cells.append(r"\textbf{" + text + "}" if value == best[c] else text)
            table.append(r"\quad " + SCORE_NAMES[key] + " & " + " & ".join(cells) + r" \\")
    (PAPER / "main_table.tex").write_text("\n".join(table + table_end()) + "\n")


def build_endpoint_table() -> None:
    records = [row for row in read_rows(SENSITIVITY / "paired_differences.csv")
               if row["dataset_split"] == "heldout" and row["stratum"] == "pooled"
               and row["risk_source"] == "member_mean" and row["score"] == "fa_v2"
               and row["baseline"] == "partition_v1"]
    selected = {row["endpoint"]: row for row in records}
    if len(records) != len(selected) or not set(ENDPOINT_NAMES) <= set(selected):
        raise ValueError("Missing or duplicate endpoint contrast")
    table = [
        r"\begin{table}[t]",
        r"\caption{Dependence on the failure definition. Paired held-out AURC difference between the combined score and the grouping-only score, $D_{\rm joint}-wQ_{\rm vox}$, under alternative risk definitions computed from the same predictions (100 CTs, 200 bones). Positive values favor grouping only. Intervals: 95\% paired case-bootstrap.}",
        r"\label{tab:sensitivity}", r"\centering\footnotesize",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Failure definition & $\Delta$AURC & 95\% CI \\", r"\midrule",
    ]
    for key, label in ENDPOINT_NAMES.items():
        row = selected[key]
        table.append(f"{label} & {float(row['delta_aurc']):+.4f} & "
                     f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]" + r" \\")
    (PAPER / "endpoint_table.tex").write_text("\n".join(table + table_end()) + "\n")


def ensure_overview(refresh: bool, offline: bool = False) -> dict[str, Any]:
    """Verify the real-CT overview asset whose panels Fig. 1 reuses.

    ``offline`` skips re-hashing the CT volume and member predictions, which
    live only on the compute server; the rendered asset is still verified.
    """
    receipt_path = PAPER / "evidence/overview.json"
    if refresh or not receipt_path.exists():
        subprocess.run([sys.executable, "-B", str(PAPER / "make_overview.py")],
                       cwd=ROOT, check=True)
    receipt = json.loads(receipt_path.read_text())
    if not offline:
        verify_hashes(ROOT, receipt["source_sha256"])
        producer = receipt["transformation"]
        verify_hashes(ROOT, {producer["script"]: producer["sha256"]})
    verify_hashes(PAPER, {k: v for k, v in receipt["output_sha256"].items()
                          if k.startswith("figures/overview.png")})
    if not receipt["component_recomputation_matches_saved"]:
        raise ValueError("Overview components were not reconciled with saved scores")
    if receipt["gt_used_for_predictor_or_slice"]:
        raise ValueError("Overview selection must be prediction-only")
    return receipt


def ensure_figures() -> dict[str, Any]:
    """Redraw Fig. 1/2 from saved evidence and verify the receipt."""
    subprocess.run([sys.executable, "-B", str(PAPER / "make_figures.py")], cwd=ROOT, check=True)
    receipt = json.loads((PAPER / "evidence/figures.json").read_text())
    verify_hashes(ROOT, receipt["source_sha256"])
    verify_hashes(PAPER, {k: v for k, v in receipt["output_sha256"].items()})
    return receipt


def markdown_table(filename: str) -> str:
    lines = []
    for line in (PAPER / filename).read_text().splitlines():
        if "&" not in line:
            continue
        cells = line.replace(r"\\", "").strip().split("&")
        lines.append("| " + " | ".join(cell.strip() for cell in cells) + " |")
        if len(lines) == 1:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n".join(lines)


def write_markdown() -> None:
    source = (PAPER / "manuscript.tex").read_text()
    title = re.search(r"\\title\{([^}]+)\}", source).group(1)
    text = source[source.index(r"\begin{abstract}"):source.index(r"\bibliographystyle")]
    text = text.replace(r"\begin{abstract}", "## Abstract\n").replace(r"\end{abstract}", "\n")
    text = text.replace(r"\begin{IEEEkeywords}", "**Keywords:** ").replace(r"\end{IEEEkeywords}", "\n")
    text = re.sub(r"\\section\*?\{([^}]+)\}", r"\n## \1\n", text)
    text = re.sub(r"\\subsection\{([^}]+)\}", r"\n### \1\n", text)
    text = re.sub(r"\\cite\{([^}]+)\}", lambda match: "[" + ", ".join(match[1].split(",")) + "]", text)
    for basename in ("main_table", "endpoint_table"):
        text = text.replace(r"\input{" + basename + "}", "\n" + markdown_table(basename + ".tex") + "\n")
    def figure_md(match: re.Match) -> str:
        body = match.group(0)
        label = re.search(r"\\label\{(fig:[^}]+)\}", body).group(1)
        number, image = {"fig:concept": ("1", "figures/fig1_concept.png"),
                         "fig:evidence": ("2", "figures/fig2_evidence.png")}[label]
        caption = re.search(r"\\caption\{(.*)\}\s*\\label", body, re.S).group(1)
        return f"\n![Figure {number}]({image})\n\n**Figure {number}.** {caption}\n"
    text = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", figure_md, text, flags=re.S)
    text = text.replace(r"\begin{equation}", "\n$$\n").replace(r"\end{equation}", "\n$$\n")
    text = text.replace(r"\begin{enumerate}", "").replace(r"\end{enumerate}", "")
    counter = iter(range(1, 20))
    text = re.sub(r"\\item\s+", lambda _: f"\n{next(counter)}. ", text)
    for command, marker in (("emph", "*"), ("textbf", "**")):
        text = re.sub(r"\\" + command + r"\{([^}]+)\}", lambda match: marker + match[1] + marker, text)
    for label, number in (("tab:main", "I"), ("tab:sensitivity", "II"), ("fig:concept", "1"),
                          ("fig:evidence", "2")):
        text = text.replace(r"\ref{" + label + "}", number)
    text = text.replace(r"\clearpage", "").replace(r"\%", "%").replace("~", " ")
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    (PAPER / "manuscript.md").write_text(
        f"# {title}\n\nAnonymous scientific draft for author review.\n\n" + text +
        "\n## References\n\n```bibtex\n" + (PAPER / "references.bib").read_text() + "```\n"
    )


def write_evidence(overview: dict[str, Any], figures: dict[str, Any]) -> None:
    relatives = [
        "outputs/stage3_h1_v2/heldout/e3_manifest.json",
        "outputs/stage3_h1_v2/heldout/main_table.csv",
        "outputs/evaluator_sensitivity_v1/sensitivity_manifest.json",
        "outputs/evaluator_sensitivity_v1/paired_differences.csv",
    ]
    sources = {name: sha256(ROOT / name) for name in relatives}
    sources.update({str((folder / name).relative_to(ROOT)): sha256(folder / name)
                    for folder, name in MANIFESTS})
    sources.update(overview["source_sha256"])
    sources.update(figures["source_sha256"])
    transform = {"script": "paper/build_paper.py", "sha256": sha256(Path(__file__))}
    trace = [
        {"artifact_id": "fig:concept", "source_data": ["paper/figures/overview.png"],
         "transformation": figures["producer"],
         "caption_claim": "Real ensemble output (case 004, left hip, development) and schematic score construction.",
         "limitations": ["Panels b and c are schematic", "Instance colours carry no cross-model correspondence"]},
        {"artifact_id": "fig:evidence", "source_data": sorted(figures["source_sha256"]),
         "transformation": figures["producer"],
         "caption_claim": "Controlled dissociation, development grouping share, held-out risk-coverage and score-risk correlations.",
         "limitations": ["Risk-coverage curves recomputed from saved scores and verified against saved AURCs"]},
    ] + [
        {"artifact_id": "tab:main", "source_data": [relatives[1]],
         "transformation": transform, "caption_claim": "All 12 fixed scores under the same bone-level AURC convention.",
         "limitations": ["Pelvic CTs contribute three bones; femur CTs contribute one", "One backend family"]},
        {"artifact_id": "tab:sensitivity", "source_data": [relatives[3]],
         "transformation": {**transform, "filter": "heldout/pooled/member_mean/fa_v2-partition_v1"},
         "caption_claim": "Paired AURC contrasts depend on the failure definition.",
         "limitations": ["Supplementary analysis of already inspected cases", "Marginal intervals", "No clinical threshold validation", "Absolute AURCs across endpoints are not directly comparable"]},
    ]
    generated = ["main_table.tex", "endpoint_table.tex", "manuscript.md",
                 "figures/fig1_concept.pdf", "figures/fig1_concept.png",
                 "figures/fig2_evidence.pdf", "figures/fig2_evidence.png",
                 "figures/overview.png", "evidence/overview.json", "evidence/figures.json"]
    evidence = {
        "role": "reporting_only_completed_evidence_and_one_real_saved_case_no_new_experiment",
        "source_sha256": sources,
        "producer": transform,
        "manuscript_input_sha256": {name: sha256(PAPER / name) for name in
                                    ("manuscript.tex", "references.bib", "IEEEtran.cls")},
        "generated_sha256": {name: sha256(PAPER / name) for name in generated},
        "figure_table_trace": trace,
        "results_not_used_in_manuscript": ["incomplete_external_RibFrac", "development_triage_policy"],
    }
    (PAPER / "evidence/manuscript_sources.json").write_text(json.dumps(evidence, indent=2) + "\n")


def compile_pdf(compiler: str | None) -> None:
    executable = compiler or shutil.which("tectonic")
    if not executable:
        fallback = Path.home() / ".local/bin/tectonic"
        if fallback.is_file():
            executable = str(fallback)
    if not executable:
        raise FileNotFoundError("Install Tectonic or specify --tectonic /path/to/tectonic")
    build = PAPER / "build"
    build.mkdir(exist_ok=True)
    subprocess.run([executable, "--only-cached", "--keep-logs", "--outdir", str(build),
                    "manuscript.tex"], cwd=PAPER, check=True)
    # Preserve the last successful PDF if compilation fails.
    shutil.copy2(build / "manuscript.pdf", PAPER / "manuscript.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Compile the IEEE PDF with cached Tectonic")
    parser.add_argument("--refresh-overview", action="store_true", help="Reread one saved CT and its predictions to redraw the overview")
    parser.add_argument("--tectonic", help="Optional path to the Tectonic executable")
    parser.add_argument("--offline", action="store_true",
                        help="Skip hashing server-only CT/prediction volumes and experiment caches")
    args = parser.parse_args()
    if not args.offline:
        for folder, name in MANIFESTS:
            verify_manifest(folder, name)
    overview = ensure_overview(args.refresh_overview, args.offline)
    figures = ensure_figures()
    build_main_table()
    build_endpoint_table()
    write_markdown()
    write_evidence(overview, figures)
    if args.compile:
        compile_pdf(args.tectonic)
    print("Verified evidence; built two tables, two figures, Markdown, and source receipt.")
    print("No training, inference, score fitting, or experiment bootstrap was run.")


if __name__ == "__main__":
    main()
