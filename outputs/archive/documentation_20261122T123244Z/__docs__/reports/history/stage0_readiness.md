# Stage 0 — Readiness and frozen protocol

Status: **PASS**

Date: 2026-09-21 (Asia/Ho_Chi_Minh)

## Configuration

The following settings are frozen from `__docs__/paper.md` and `__docs__/plan.md`:

- PENGWIN public labeled cases only: 50 static test / 40 validation / 250 AL pool.
- Exact anatomy quota per split: test 25 pelvic + 25 femur; validation 20 + 20; pool 125 + 125.
- Initial labeled set: 40 cases per trajectory; five whole-volume rounds of 20; budgets `40, 60, 80, 100, 120, 140`.
- Three trajectory seeds: `20260920`, `20260921`, `20260922`. All acquisition methods share the same initial set within a trajectory.
- Main ensemble size `M = 3`; `M = 2,3,4` is limited to the Round-0 ablation.
- H1 risks: `1 - Instance F1` and normalized merge-plus-split error; `1 - Dice` is the semantic control.
- H1 analyses: risk–coverage/AURC, top-10% and top-20% failure enrichment, and fragment-count/foreground-volume-controlled association. Spearman correlation is secondary.
- H2 co-primary endpoints: trajectory-level AULC for Instance F1 and merge-plus-split error.
- Semantic non-inferiority margin: **1.0 absolute Dice point** below the best non-structural uncertainty baseline at matched budget.
- Ground truth may define the static split and initial sets but must never be exposed to acquisition.

The internal research evaluation contract is frozen in `configs/study_protocol.json`. It specifies the matching, filtering, Instance F1, normalized merge/split risk, H1 risk–coverage/top-k analyses, H2 AULC, and Dice guardrail. It is explicitly not represented as the official challenge evaluator. The available ABBC implementation identifies itself as `task1_official_aligned_proxy_v2`; later organizer clarification is an external dependency and any conflict must be reported before changing this frozen contract.

## Result

### Data and semantics

- PENGWIN contains 340 `image.mha` and 340 `label.mha` files: 170 pelvic and 170 femur subjects.
- Documented subject-local instance labels are background `0`, sacrum `1–50`, left hip `51–100`, right hip `101–150`, and femur `151–200`.
- Case `001` has a 3D compressed `MET_SHORT` label header with size `512 x 512 x 401` and spacing approximately `0.78125 x 0.78125 x 0.8 mm`.
- The completed dataset-wide preparation run read all 340 image/label pairs with SimpleITK, verified paired geometry, validated anatomy-specific label ranges, and generated the GT audit successfully.
- All 1,360 documented Task 2 click JSON files are present. They remain outside Task 1 acquisition.
- RibFrac has 420 extracted training pairs, 80 extracted validation labels, and 160 extracted unlabeled test images. `dataset/RibFrac/ribfrac-val-images.zip` contains all 80 validation images (`RibFrac421`-`RibFrac500`); those images are not yet extracted and are not needed until Stage 6/H3.

### Publication follow-up (not a Stage 0 blocker)

On 2026-09-21 the user confirmed that the current data are usable for this project and explicitly accepted Stage 0 as passed. Publication/embargo clarification is tracked separately and does not block Stage 1 or Stage 2 implementation.

- Workspace ACL write access for `24chuong.ta` was restored during this stage.
- The official PENGWIN rule states a 12-month publication embargo after challenge-result announcement. The challenge timeline places result announcement on 24 August 2026, which makes the stated embargo extend past the 26 October 2026 ISBI deadline.
- The PENGWIN Zenodo record does not display a populated license value.
- No written exception or clarification is present in the workspace.

Required organizer request (not sent by the agent):

```text
To: yanzhenliu@buaa.edu.cn
Cc: sangyudi@rossumrobot.cn
Subject: Written clarification requested: PENGWIN 2026 public training data in ISBI 2027 submission

Dear PENGWIN 2026 organizers,

We plan an independent ISBI 2027 submission (deadline 26 October 2026) using
only the 340 public labeled Task 1 training cases. We would create and disclose
a fixed internal train/validation/test split and would not use or report the
hidden challenge test set or leaderboard results.

Could you confirm in writing whether this submission is permitted before the
stated 12-month publication embargo expires? If the embargo applies, may we
receive an explicit exception? Please also confirm the applicable dataset
license/citation terms and provide or confirm the exact Task 1 Instance F1 and
merge/split specification (filtering, connectivity, matching threshold, and
normalization), or an official evaluator if available.
```

### Implemented preparation

`scripts/stage0_prepare_pengwin_split.py` was added and its dataset-wide path was executed by the user. It:

1. verify the 340-case public inventory and image/label geometry;
2. validate label IDs against pelvic/femur ranges;
3. compare each case's label-fragment count with all four click files;
4. compute GT fragment count, affected-bone signature, and foreground volume;
5. generate the fixed 50/40/250 split and three 40-case initial sets using exact anatomy quotas and deterministic distribution balancing;
6. keep the acquisition-safe split manifest separate from the GT-only audit metadata.

The syntax check, `--help` check, synthetic allocator check, user-owned full run, and JSON-only output validation passed after the compatibility fix described below.

The first user-typed dataset-wide run failed during case metadata inspection before producing outputs. The installed SimpleITK `ImageFileReader` provides `GetPixelID()` but not `GetPixelIDTypeAsString()`. The script was corrected to resolve the numeric pixel ID through the module-level `GetPixelIDValueAsString()` when available, with a numeric-string fallback. The user reran the command successfully.

Generated outputs:

- `splits/pengwin_stage0_split.json` (8.1 KiB)
- `splits/pengwin_stage0_gt_audit.json` (607 KiB; acquisition code must never load this GT-derived file)

JSON-only smoke validation passed:

- split sizes are exactly 50 test / 40 validation / 250 pool;
- assignments are disjoint and exhaustive over all 340 audited cases;
- anatomy quotas are exactly 25/25 in test, 20/20 in validation, and 125/125 in the pool;
- all three initial sets contain 40 unique pool cases, exactly 20 pelvic and 20 femur;
- all four click counts equal the label-fragment count for every audited case;
- audited fragment counts range from 2 to 30, with mean `7.1382352941176475`;
- the manifest's self-recorded SHA-256 was reproduced as `0ef703ed51a3d64c992cf3e5e40d22f39d4f4d83166b0df69ef9aa09cd2ccf4a`.

## Gateway

**PASS.** The user has accepted the data as usable for the project. The 340-case PENGWIN audit, fixed split, initial trajectory sets, leakage guard, internal metric contract and artifact validator are present and validated. Stage 1 and Stage 2 work are authorized.

This project gate is not itself written organizer clearance. The publication/embargo question must still be resolved before submission or any claim that the dataset is publication-cleared.

## Tracked follow-up items (non-blocking for Stage 0)

- Written PENGWIN publication/data-use clearance remains absent, and the apparent embargo covers the ISBI deadline.
- The RibFrac validation-image archive must be integrity-checked and extracted before Stage 6/H3.
- Official PENGWIN instance-metric semantics remain unconfirmed. Stage 1 implements the frozen internal research contract without claiming it is official.

## Agent-typed commands

The following lightweight/read-only or syntax commands were run. The first repository-discovery command initially failed during sandbox setup and was repeated outside that sandbox. The first `mkdir` attempt failed before the ACL was fixed; the later attempt succeeded. The ownership command hung and was terminated.

```text
pwd && rg --files -g 'AGENTS.md' -g '__docs__/plan.md' -g '__docs__/rule.md' -g '!node_modules' -g '!vendor'
cat __docs__/rule.md __docs__/plan.md
ls -la
find __docs__ -maxdepth 2 -type f -print
find . -maxdepth 2 -type f -not -path './.git/*' -print | sort | head -300
cat __docs__/paper.md
git status --short
wc -l __docs__/paper.md __docs__/data_description.md
cat __docs__/data_description.md
find dataset -maxdepth 4 -type d -print | sort | head -300
find dataset -maxdepth 4 -type f -print | sort | head -300
find baselines -maxdepth 2 -type d -print | sort
find baselines -maxdepth 2 -type f -print | sort | head -300
rg -n "^#|non-infer|margin|static split|H1|H2|AURC|top-10|top-20|Instance F1|merge|split|Dice" __docs__/paper.md
sed -n '330,760p' __docs__/paper.md
find dataset/PENGWIN26_task1_2 -type f -name 'image.mha' | wc -l
find dataset/PENGWIN26_task1_2 -type f -name 'label.mha' | wc -l
find dataset/RibFrac -type f -name '*-image.nii.gz' | wc -l
find dataset/RibFrac -type f -name '*-label.nii.gz' | wc -l
find dataset/RibFrac -maxdepth 2 -type d -print | sort
file dataset/PENGWIN26_task1_2/PENGWIN26_task1_2_train_part1/001/image.mha dataset/PENGWIN26_task1_2/PENGWIN26_task1_2_train_part1/001/label.mha
head -c 1024 dataset/PENGWIN26_task1_2/PENGWIN26_task1_2_train_part1/001/image.mha | strings
head -c 1024 dataset/PENGWIN26_task1_2/PENGWIN26_task1_2_train_part1/001/label.mha | strings
python -c "import SimpleITK as s, numpy as n; p='dataset/PENGWIN26_task1_2/PENGWIN26_task1_2_train_part1/001/label.mha'; im=s.ReadImage(p); a=s.GetArrayViewFromImage(im); print('size=', im.GetSize()); print('spacing=', im.GetSpacing()); print('pixel_id=', im.GetPixelIDTypeAsString()); print('labels=', n.unique(a).tolist())"
rg -n -i "merge.?error|split.?error|instance.?f1|topology consistency|fracture dice|local dice" baselines __docs__ -g '*.py' -g '*.md' -g '*.json' -g '*.yaml' -g '*.yml'
find dataset/PENGWIN26_task1_2 -maxdepth 3 -type f \( -iname 'README*' -o -iname '*.txt' -o -iname '*.json' \) -print | sort | head -200
find dataset/RibFrac -maxdepth 2 -type f \( -iname 'README*' -o -iname '*.txt' -o -iname '*.csv' \) -print | sort
head -40 dataset/RibFrac/urls.txt
head -20 dataset/RibFrac/ribfrac-val-info.csv
cat dataset/PENGWIN26_task1_2/README.md
sed -n '520,930p' baselines/pengwin2026-task1-abbc/code_task1/eval.py
sed -n '1180,1225p' baselines/pengwin2026-task1-abbc/code_task1/eval.py
sed -n '45,80p' baselines/pengwin2026-task1-abbc/README.md
find dataset/PENGWIN26_task1_2 -maxdepth 2 -type d -name '*click*' -print
find dataset/PENGWIN26_task1_2 -maxdepth 4 -type f -name 'peripelvic-fragment-clicks.json' | head -8
find dataset/PENGWIN26_task1_2 -maxdepth 4 -type f -name 'peripelvic-fragment-clicks.json' | wc -l
mkdir -p scripts reports
id && stat -c '%A %U %G %a %n' . __docs__
find dataset -type f -name 'RibFrac421-image.nii.gz' -print
find dataset -type f \( -name 'RibFrac4[2-9][1-9]-image.nii.gz' -o -name 'RibFrac500-image.nii.gz' \) | wc -l
find dataset/RibFrac -maxdepth 2 -type f -name '*-image.nii.gz' | sed 's#dataset/RibFrac/##' | sort | tail -15
mkdir -p scripts reports
python -m py_compile scripts/stage0_prepare_pengwin_split.py
python scripts/stage0_prepare_pengwin_split.py --help
python -c "import importlib.util; p='scripts/stage0_prepare_pengwin_split.py'; s=importlib.util.spec_from_file_location('stage0',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); records=[{'case_id':f'{i:03d}','anatomy':'pelvic','fragment_count':2+i%9,'foreground_volume_mm3':1000.0+i,'bone_groups':['sacrum']} for i in range(1,171)]+[{'case_id':f'{i:03d}','anatomy':'femur','fragment_count':2+i%9,'foreground_volume_mm3':1000.0+i,'bone_groups':['femur']} for i in range(251,421)]; m.attach_strata(records); pelvic=[r for r in records if r['anatomy']=='pelvic']; a,b,c=m.best_partition(pelvic,25,20,1,100); pool=c+[r for r in records if r['anatomy']=='femur'][:125]; initial=m.select_initial_set(pool,1,100); assert (len(a),len(b),len(c),len(initial))==(25,20,125,40); assert not ({r['case_id'] for r in a}&{r['case_id'] for r in b}); print('synthetic split smoke: PASS')"
find scripts reports -maxdepth 2 -type f -print | sort
rg -n "^Status:|^## Gateway|BLOCKED|stage0_prepare" reports/stage0_readiness.md
rm scripts/__pycache__/stage0_prepare_pengwin_split.cpython-313.pyc
rmdir scripts/__pycache__
python -c "from pathlib import Path; compile(Path('scripts/stage0_prepare_pengwin_split.py').read_text(), 'scripts/stage0_prepare_pengwin_split.py', 'exec'); print('syntax: PASS')"
python -B -c "import importlib.util; from pathlib import Path; s=importlib.util.spec_from_file_location('stage0','scripts/stage0_prepare_pengwin_split.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); R=type('R',(),{'SetFileName':lambda self,p:None,'ReadImageInformation':lambda self:None,'GetSize':lambda self:(1,2,3),'GetSpacing':lambda self:(1.,1.,1.),'GetOrigin':lambda self:(0.,0.,0.),'GetDirection':lambda self:(1.,0.,0.,0.,1.,0.,0.,0.,1.),'GetPixelID':lambda self:7}); S=type('S',(),{'ImageFileReader':R,'GetPixelIDValueAsString':staticmethod(lambda value:'pixel-'+str(value))}); info=m.image_information(S,Path('unused')); assert info['pixel_id']=='pixel-7' and info['pixel_id_value']==7; print('SimpleITK compatibility smoke: PASS')"
ls -lh splits/pengwin_stage0_split.json splits/pengwin_stage0_gt_audit.json
python -c "import json,hashlib; from pathlib import Path; mp=Path('splits/pengwin_stage0_split.json'); ap=Path('splits/pengwin_stage0_gt_audit.json'); m=json.loads(mp.read_text()); a=json.loads(ap.read_text()); s=m['split']; assert {k:len(v) for k,v in s.items()}=={'pool':250,'test':50,'validation':40}; all_ids=s['test']+s['validation']+s['pool']; assert len(all_ids)==len(set(all_ids))==340; assert a['n_cases']==len(a['records'])==340; by_id={r['case_id']:r for r in a['records']}; assert set(all_ids)==set(by_id); assert m['split_summary']['test']['anatomy_counts']=={'femur':25,'pelvic':25}; assert m['split_summary']['validation']['anatomy_counts']=={'femur':20,'pelvic':20}; assert m['split_summary']['pool']['anatomy_counts']=={'femur':125,'pelvic':125}; pool=set(s['pool']); assert len(m['trajectories'])==3; [(lambda ids: (assertions := (len(ids)==len(set(ids))==40 and set(ids)<=pool and sum(by_id[x]['anatomy']=='pelvic' for x in ids)==20 and sum(by_id[x]['anatomy']=='femur' for x in ids)==20)) or (_ for _ in ()).throw(AssertionError('trajectory validation failed')))(t['initial_labeled_case_ids']) for t in m['trajectories']]; expected=m.pop('content_sha256_without_this_field'); actual=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest(); assert actual==expected,(actual,expected); assert all(all(c==r['fragment_count'] for c in r['click_counts'].values()) for r in a['records']); print('manifest validation: PASS'); print('fragment_count=',a['summary']['fragment_count']); print('anatomy_counts=',a['summary']['anatomy_counts']); print('trajectory_seeds=',[t['seed'] for t in m['trajectories']]); print('sha256=',expected)"
find dataset -type f -name 'RibFrac421-image.nii.gz' -print
sed -n '55,115p' reports/stage0_readiness.md
cat __docs__/rule.md
sed -n '94,127p' __docs__/plan.md
find . -maxdepth 2 -type f -not -path './dataset/*' -not -path './baselines/*' -print | sort
mkdir -p configs
python -B scripts/validate_stage0_artifacts.py
python -c "from pathlib import Path; compile(Path('scripts/validate_stage0_artifacts.py').read_text(), 'scripts/validate_stage0_artifacts.py', 'exec'); print('validator syntax: PASS')"
find configs scripts reports splits -maxdepth 2 -type f -print | sort
sed -n '1,125p' reports/stage0_readiness.md
```

Official web evidence checked on 2026-09-20:

- <https://pengwin2026.grand-challenge.org/rules/>
- <https://pengwin2026.grand-challenge.org/timeline/>
- <https://pengwin2026.grand-challenge.org/organizers/>
- <https://zenodo.org/records/19732767>

## User-typed commands

Dataset-wide preparation command executed by the user. The first run failed before output generation because `ImageFileReader.GetPixelIDTypeAsString()` was unavailable. After the compatibility correction, the second run completed and wrote both manifests.

```bash
python scripts/stage0_prepare_pengwin_split.py --dataset-root dataset/PENGWIN26_task1_2 --output-dir splits --seed 20260920 --trajectory-seeds 20260920 20260921 20260922
```

If the official RibFrac validation images do not exist elsewhere on the system:

```bash
wget -c https://zenodo.org/records/3893496/files/ribfrac-val-images.zip -O dataset/RibFrac/ribfrac-val-images.zip
unzip -n dataset/RibFrac/ribfrac-val-images.zip -d dataset/RibFrac
```

The organizer email above is also a required user action, not a shell command.

## Next step

Stage 0 is closed as PASS. This report's earlier Stage 1 blocker was superseded
on 2026-09-21 by the completed FA-IPD construct tests and 340-case audit; see
`__docs__/reports/stage1_instance_audit.md`. Publication clearance, official
metric clarification and RibFrac validation-image extraction remain tracked for
their respective later gates.
