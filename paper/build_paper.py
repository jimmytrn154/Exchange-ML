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
    "pairwise_dice": "Foreground Dice disagreement",
    "presence": "Presence (Jaccard disagreement)",
    "union_nvi": "Union-normalized VI",
    "count_variance": "Instance-count variance",
    "hungarian": "Hungarian matching disagreement",
    "fa_v1": r"$D_{\rm vox}$ (combined)",
    "partition_v1": r"$wQ_{\rm vox}$ (partition only)",
    "partition_v2": r"$wQ_{\rm joint}$ (partition only)",
    "fa_v2": r"$D_{\rm joint}$ (combined)",
    "q_joint_unweighted": r"$Q_{\rm joint}$ (unweighted VI/H)",
}
ENDPOINT_NAMES = {
    "original_merge_split": "Original M+S",
    "one_to_one_one_minus_f1": r"One-to-one $1-F_1$",
    "significant_0p01_merge_split": r"Significant overlap, $\tau=.01$",
    "significant_0p1_merge_split": r"Significant overlap, $\tau=.10$",
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


def build_main_table() -> None:
    records = read_rows(HELDOUT / "main_table.csv")
    if len(records) != 12 or {row["score"] for row in records} != set(SCORE_NAMES):
        raise ValueError("Expected exactly the 12 fixed score controls")
    table = [
        r"\begin{table}[t]",
        r"\caption{Bone-level AURC for all 12 scores; lower is better. M+S denotes normalized merge-plus-split risk. Subscripts vox and joint denote normalization by voxel count and joint entropy, respectively. Dev.: 40 development CTs; Eval.: 100 separate evaluation CTs.}",
        r"\label{tab:main}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrr}",
        r"\toprule", r"Score & Dev. M+S & Eval. M+S & Eval. $1-F_1$ \\",
        r"\midrule",
    ]
    for row in records:
        values = [float(row[key]) for key in (
            "validation_merge_split_aurc", "heldout_merge_split_aurc",
            "heldout_one_minus_f1_aurc",
        )]
        table.append(SCORE_NAMES[row["score"]] + " & " +
                     " & ".join(f"{value:.4f}" for value in values) + r" \\")
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
        r"\caption{Supplementary endpoint sensitivity on the same 100 evaluation CTs: joint-entropy combined score minus voxel-normalized partition-only score ($D_{\rm joint}-wQ_{\rm vox}$). Negative AURC differences favor the combined score. Intervals are marginal 95\% paired CT-bootstrap intervals.}",
        r"\label{tab:sensitivity}", r"\centering\footnotesize",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Failure definition & $\Delta$AURC & 95\% CI \\", r"\midrule",
    ]
    for key, label in ENDPOINT_NAMES.items():
        row = selected[key]
        table.append(f"{label} & {float(row['delta_aurc']):+.4f} & "
                     f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]" + r" \\")
    (PAPER / "endpoint_table.tex").write_text("\n".join(table + table_end()) + "\n")


def ensure_overview(refresh: bool) -> dict[str, Any]:
    receipt_path = PAPER / "evidence/overview.json"
    if refresh or not receipt_path.exists():
        subprocess.run([sys.executable, "-B", str(PAPER / "make_overview.py")],
                       cwd=ROOT, check=True)
    receipt = json.loads(receipt_path.read_text())
    verify_hashes(ROOT, receipt["source_sha256"])
    verify_hashes(PAPER, receipt["output_sha256"])
    producer = receipt["transformation"]
    verify_hashes(ROOT, {producer["script"]: producer["sha256"]})
    if not receipt["component_recomputation_matches_saved"]:
        raise ValueError("Overview components were not reconciled with saved scores")
    if receipt["gt_used_for_predictor_or_slice"]:
        raise ValueError("Overview selection must be prediction-only")
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
    caption = re.search(r"\\caption\{(.*?)\}\s*\\label\{fig:overview\}", source, re.S).group(1)
    figure = ("\n![Real CT, saved predictions, disagreement components, and evaluation]"
              "(figures/overview.png)\n\n**Figure 1.** " + caption + "\n")
    text = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", lambda _: figure, text, flags=re.S)
    text = text.replace(r"\begin{equation}", "\n$$\n").replace(r"\end{equation}", "\n$$\n")
    text = text.replace(r"\begin{enumerate}", "").replace(r"\end{enumerate}", "")
    counter = iter(range(1, 20))
    text = re.sub(r"\\item\s+", lambda _: f"\n{next(counter)}. ", text)
    for command, marker in (("emph", "*"), ("textbf", "**")):
        text = re.sub(r"\\" + command + r"\{([^}]+)\}", lambda match: marker + match[1] + marker, text)
    for label, number in (("tab:main", "I"), ("tab:sensitivity", "II"), ("fig:overview", "1")):
        text = text.replace(r"\ref{" + label + "}", number)
    text = text.replace(r"\clearpage", "").replace(r"\%", "%").replace("~", " ")
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    (PAPER / "manuscript.md").write_text(
        f"# {title}\n\nAnonymous scientific draft for author review.\n\n" + text +
        "\n## References\n\n```bibtex\n" + (PAPER / "references.bib").read_text() + "```\n"
    )


def write_evidence(overview: dict[str, Any]) -> None:
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
    transform = {"script": "paper/build_paper.py", "sha256": sha256(Path(__file__))}
    trace = overview["figure_table_trace"] + [
        {"artifact_id": "tab:main", "source_data": [relatives[1]],
         "transformation": transform, "caption_claim": "All 12 fixed scores under the same bone-level AURC convention.",
         "limitations": ["Pelvic CTs contribute three bones; femur CTs contribute one", "One backend family"]},
        {"artifact_id": "tab:sensitivity", "source_data": [relatives[3]],
         "transformation": {**transform, "filter": "heldout/pooled/member_mean/fa_v2-partition_v1"},
         "caption_claim": "Paired AURC contrasts depend on the failure definition.",
         "limitations": ["Supplementary analysis of already inspected cases", "Marginal intervals", "No clinical threshold validation", "Absolute AURCs across endpoints are not directly comparable"]},
    ]
    generated = ["main_table.tex", "endpoint_table.tex", "manuscript.md",
                 "figures/overview.pdf", "figures/overview.png", "evidence/overview.json"]
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
    args = parser.parse_args()
    for folder, name in MANIFESTS:
        verify_manifest(folder, name)
    overview = ensure_overview(args.refresh_overview)
    build_main_table()
    build_endpoint_table()
    write_markdown()
    write_evidence(overview)
    if args.compile:
        compile_pdf(args.tectonic)
    print("Verified evidence; built two tables, real-data overview, Markdown, and source receipt.")
    print("No training, inference, score fitting, or experiment bootstrap was run.")


if __name__ == "__main__":
    main()
