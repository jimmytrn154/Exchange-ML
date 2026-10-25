# Storage layout

Last checked: 2026-10-21.

The repository at `/home/24chuong.ta/chun/Exchange-ML` (disk `/dev/sda3`) holds
code, configs, docs, splits, and small manifests and metrics. Bulk data —
source datasets, nnU-Net raw/preprocessed data, checkpoints, predictions, and
probability maps — lives on `/mnt/sdb/shared/chuong/` and is reached through
symlinks at the paths the scripts expect. No script or config refers to
`/mnt/sdb`; the links are transparent to the code.

## Why

The user quota on `/dev/sda3` is 100 GB soft / 293 GB hard. On 2026-10-16
usage reached 282 GB, the grace period had expired, and every write failed
(training crashed mid-epoch). The bulk data was moved to `/mnt/sdb` (3.5 TB)
and linked back. Usage after the move: about 92 GB, so **headroom on
`/dev/sda3` is only about 8 GB**. Incident record:
`__docs__/reports/stage3/stage3_fix_progress.md`, session log 2026-10-16.

## What is where

| Repo path | Target on `/mnt/sdb/shared/chuong/` | Content | Size (2026-10-21) |
|---|---|---|---|
| `dataset/` | `dataset/` | PENGWIN and RibFrac source data | 134 GB |
| 39 paths under `outputs/` (file or folder links) | `b/001` … `b/039` | nnU-Net raw `imagesTr`/`labelsTr` and preprocessed `nnUNetPlans_3d_fullres`/`gt_segmentations` (Datasets 701, 702); Stage 2 checkpoints and `validation_predictions`; checkpoints and predictions in `outputs/archive/` and `outputs/smoke_f4/`; four D6 `.npy` probability dumps | `b/` 59 GB in total (with the two `m02_frag*` folders) |
| `outputs/stage3_h1/members/member_01/nnUNet_results` | `m01_results/` | member_01 Anatomy and Fragment runs | 1.8 GB |
| `outputs/stage3_h1/members/member_02/nnUNet_results/Dataset702_PENGWINStage2Fragment` | `b/m02_frag/` | member_02 Fragment run (its Anatomy checkpoints are `b/021`, `b/022`) | in `b/` |
| `outputs/stage3_h1/members/member_03/nnUNet_results` | `m03_results/` | member_03 Anatomy and Fragment runs | 1.8 GB |
| `outputs/stage3_h1/members/member_0X/validation_predictions` | `m0X_infer/validation_predictions/` | member validation label maps (X = 1, 2, 3) | 3.1–3.3 GB per member, with probabilities |
| `outputs/stage3_h1/members/member_0X/validation_probabilities` | `m0X_infer/validation_probabilities/` | member foreground probability maps | (as above) |

The full `b/NNN` ↔ original repo path list is
[`storage_map.tsv`](storage_map.tsv), a copy of `/mnt/sdb/shared/chuong/b/MAP.tsv`.
`b/` was filled by moving each item (copy → `diff -rq` → delete → link), so a
`b/NNN` folder is exactly what used to be at its repo path.

Kept on `/mnt/sdb` only (no repo link):

| Folder | Content |
|---|---|
| `archive_2026-10-20_pre_F9/` (12 GB) | Stage 2 and member validation outputs made before the D8/F9 fix; the matching small manifests and metrics are in `outputs/archive/2026-10-20_pre_F9_inference/` |
| `b/m02_frag_crashed/` | member_02 Fragment run lost to the 2026-10-16 quota stop |
| `m03_frag_crashed_2026-10-19/` (445 MB) | member_03 Fragment run lost to the 2026-10-19 server reboot |
| `logs/` | empty |

Still on `/dev/sda3`: everything else in the repo, including the member_02
Anatomy folder (`debug.json`, logs, `validation/`), training logs in each
`fold_0/`, `outputs/logs/`, all manifests, metrics CSV/JSON files, and
diagnostics.

Read-only reference, not ours: `/mnt/sdb/shared/dang.cpm/Exchange-ML/` holds
the pre-fix (2026-10-05) H1 member outputs used in diagnostics. Never write
there.

## Permissions

`b/` and `dataset/` are `drwx------` (owner only). The top folder
`/mnt/sdb/shared/chuong` and the `m0X_*`, `archive_*`, `m03_frag_crashed_*`,
and `logs/` folders are `drwxrwxr-x`, so other users on the server can read
them. `/mnt/sdb/shared` itself is world-writable with the sticky bit.

## Rules

- Before a run writes large outputs to a new repo path (a new member, a new
  inference, a new archive), create the target under `/mnt/sdb/shared/chuong/`
  and link it first. Example from the member runs:

  ```bash
  M=outputs/stage3_h1/members/member_0X; S=/mnt/sdb/shared/chuong/m0X_infer
  mkdir -p $S/validation_predictions $S/validation_probabilities
  ln -s $S/validation_predictions $M/validation_predictions
  ln -s $S/validation_probabilities $M/validation_probabilities
  ```

- To archive outputs, move them within `/mnt/sdb` (instant) and recreate the
  empty target, so the repo link stays valid; move the small repo-side files
  to `outputs/archive/<date>_<reason>/`.
- Do not replace a link with a real folder (`rm` of a link then `mkdir`);
  that puts the data back on `/dev/sda3`.
- Check quota with `quota -s`; keep `/dev/sda3` below 100 GB.
- Git (checked 2026-10-21): `dataset` is ignored as `/dataset` (the
  `dataset/` pattern alone does not match a symlink). The 28 file-level links
  (`*.pth` checkpoints, `.npy` dumps) are ignored like the files they replace.
  The 21 folder-level links (nnU-Net raw/preprocessed folders, all
  `validation_predictions`/`validation_probabilities`, member `nnUNet_results`)
  are **committed as symlinks** (mode `120000`, absolute `/mnt/sdb/...`
  targets); Git stores only the link, never the data behind it. When adding a
  new folder link, commit it the same way.

## Checks

```bash
cd /home/24chuong.ta/chun/Exchange-ML
find outputs dataset -type l | wc -l   # 49 links on 2026-10-21
find -L outputs dataset -type l        # broken links; must print nothing
quota -s                               # /dev/sda3 usage
du -sh /mnt/sdb/shared/chuong/*        # bulk data sizes
```

## Recreating the layout elsewhere

A fresh clone has none of the bulk data. It does contain the 21 committed
folder links, which dangle on any machine without `/mnt/sdb/shared/chuong/`
(the scripts then fail on missing folders; remove or replace a dangling link
before regenerating into it). Either regenerate the data
with the stage scripts (Stage 2 preparation, training, inference) at the repo
paths, or copy the folders above from `/mnt/sdb/shared/chuong/` and recreate
each link from the tables and [`storage_map.tsv`](storage_map.tsv)
(`ln -s <sdb_path> <original_repo_path>`).
