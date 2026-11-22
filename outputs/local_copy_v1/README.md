# Local v1 artifacts

Status: complete for all 49 planned source locations.

Copied and SHA-256 verified **807 files, 44,697,656,493 bytes**, including
**35 checkpoints** across current and historical runs. Sources are unchanged.
`manifest.json` records exact source/destination paths and per-file hashes;
`plan.json` retains the mapping. No imports remain pending.

Current inputs include all three members' Anatomy/Fragment checkpoints,
120 instance maps, 120 probability maps, Stage 2 checkpoints and 40 Stage 2
validation predictions. Older outputs are kept under their archive paths.

`validation.json` records the final local-file/size checks, six member
inference-checkpoint fingerprint matches, and E1 input availability.
40 validation cases comprise 20 pelvic + 20 femur / 80 annotated bone units.
GT is readable through `dataset -> ../dataset`; dataset contents were not copied
or changed. No training, inference, or E1 experiment was executed.

Source `chuong/b/` permits reads at the known paths but still does not permit
listing its directory, so completeness is defined by the recorded 49-location
plan and storage map. Raw/preprocessed training datasets are excluded.

The retained helper `copy_outputs.py` skips verified jobs and can be used to
inspect or resume this plan if needed. Its SHA-256 checks occur during transfer;
rerunning a completed plan skips existing verified jobs.
