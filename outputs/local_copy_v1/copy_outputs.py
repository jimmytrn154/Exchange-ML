"""Copy v1 artifacts into this checkout, verifying bytes before replacing links.

Does not modify sources, overwrite different files, or copy training datasets.
Run again after the source owner grants read access to the pending sources.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/local_copy_v1"
PLAN = AUDIT / "plan.json"
REPORT = AUDIT / "manifest.json"
BUFFER = 8 * 1024 * 1024


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_plan() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    mapping = ROOT / "__docs__/storage_map.tsv"
    for line in mapping.read_text().splitlines()[1:]:
        source, old = line.split("\t")
        relative = old.split("/Exchange-ML/", 1)[1]
        if "/nnUNet_raw/" in relative or "/nnUNet_preprocessed/" in relative:
            continue
        jobs.append({"source": source, "destination": relative})
    members = ROOT / "outputs/stage3_h1/members"
    for member in sorted(members.iterdir()):
        for current, directories, files in os.walk(member, followlinks=False):
            for name in directories + files:
                destination = Path(current) / name
                if destination.is_symlink():
                    relative = str(destination.relative_to(ROOT))
                    if not any(job["destination"] == relative for job in jobs):
                        jobs.append({"source": os.readlink(destination), "destination": relative})
    source_root = Path("/mnt/sdb/shared/chuong")
    for number in range(1, 4):
        for kind in ("validation_predictions", "validation_probabilities"):
            jobs.append({
                "source": str(source_root / "archive_2026-10-20_pre_F9" / f"m0{number}_infer" / kind),
                "destination": f"outputs/archive/2026-10-20_pre_F9_inference/member_0{number}/{kind}",
            })
    jobs.extend([
        {"source": str(source_root / "archive_2026-10-20_pre_F9/stage2_validation_predictions"),
         "destination": "outputs/archive/2026-10-20_pre_F9_inference/stage2_backend/validation_predictions"},
        {"source": str(source_root / "m03_frag_crashed_2026-10-19"),
         "destination": "outputs/archive/2026-10-19_member_03_fragment_interrupted/Dataset702_PENGWINStage2Fragment"},
        {"source": str(source_root / "b/m02_frag_crashed"),
         "destination": "outputs/archive/2026-10-16_member_02_fragment_quota_stop/Dataset702_PENGWINStage2Fragment"},
    ])
    return jobs


def inventory(source: Path) -> tuple[bool, list[tuple[Path, str, os.stat_result]]]:
    info = source.stat()
    if stat.S_ISREG(info.st_mode):
        with source.open("rb"):
            pass
        return False, [(source, "", info)]
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Unsupported source: {source}")
    files = []
    def onerror(error: OSError) -> None:
        raise error
    for base, directories, names in os.walk(source, onerror=onerror):
        for name in directories + names:
            item = Path(base) / name
            if item.is_symlink():
                raise ValueError(f"Nested source symlink requires review: {item}")
        for name in sorted(names):
            item = Path(base) / name
            info = item.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Unsupported source: {item}")
            with item.open("rb"):
                pass
            files.append((item, str(item.relative_to(source)), info))
    return True, files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, before: os.stat_result) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(BUFFER):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during copy: {source}")
    expected = digest.hexdigest()
    if sha256(destination) != expected:
        raise RuntimeError(f"SHA-256 mismatch: {destination}")
    shutil.copystat(source, destination)
    return expected


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    if not PLAN.exists():
        save(PLAN, make_plan())
    jobs = json.loads(PLAN.read_text())
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"jobs": {}}
    report.update({"checkout": str(ROOT), "sources_modified": False,
                   "verification": "SHA-256 of source bytes during copy equals SHA-256 reread from destination",
                   "excluded": "Source datasets and nnUNet_raw/nnUNet_preprocessed training data"})
    for index, job in enumerate(jobs, 1):
        relative = job["destination"]
        destination = ROOT / relative
        source = Path(job["source"])
        if report["jobs"].get(relative, {}).get("status") == "verified":
            continue
        item = {**job, "status": "pending", "files": []}
        report["jobs"][relative] = item
        try:
            if not destination.absolute().is_relative_to(ROOT / "outputs"):
                raise ValueError(f"Destination outside outputs: {destination}")
            if not destination.parent.resolve().is_relative_to(ROOT / "outputs"):
                raise ValueError(f"Destination parent follows a link outside outputs: {destination}")
            is_directory, files = inventory(source)
            size = sum(info.st_size for _, _, info in files)
            item["source_bytes"] = size
            item["source_file_count"] = len(files)
            if shutil.disk_usage(ROOT).free < size + 2 * 1024**3:
                raise OSError(f"Insufficient free space for {size} bytes")
            staging = destination.with_name(destination.name + ".v1-copy-staging")
            if staging.exists():
                raise FileExistsError(f"Staging from interrupted job retained for review: {staging}")
            original_link = os.readlink(destination) if destination.is_symlink() else None
            if destination.exists() and original_link is None:
                raise FileExistsError(f"Existing destination retained: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if is_directory:
                staging.mkdir()
            print(f"[{index}/{len(jobs)}] COPYING {relative}: {len(files)} files, {size / 1024**3:.2f} GiB", flush=True)
            item["original_symlink"] = original_link
            save(REPORT, report)
            for file_source, file_relative, info in files:
                target = staging / file_relative if is_directory else staging
                digest = copy_file(file_source, target, info)
                target_relative = destination / file_relative if is_directory else destination
                item["files"].append({"path": str(target_relative.relative_to(ROOT)),
                                      "bytes": info.st_size, "sha256": digest})
            if original_link is not None:
                if not destination.is_symlink() or os.readlink(destination) != original_link:
                    raise RuntimeError(f"Destination link changed: {destination}")
                destination.unlink()
            try:
                staging.rename(destination)
            except BaseException:
                if original_link is not None and not os.path.lexists(destination):
                    destination.symlink_to(original_link)
                raise
            item["status"] = "verified"
            print(f"[{index}/{len(jobs)}] VERIFIED {relative}", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            item["error"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{len(jobs)}] PENDING {relative}: {item['error']}", flush=True)
        report["updated_at_unix"] = time.time()
        save(REPORT, report)
    verified = [job for job in report["jobs"].values() if job["status"] == "verified"]
    report["summary"] = {"verified_jobs": len(verified), "pending_jobs": len(jobs) - len(verified),
                         "verified_files": sum(len(job["files"]) for job in verified),
                         "verified_bytes": sum(job["source_bytes"] for job in verified)}
    save(REPORT, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
