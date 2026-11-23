#!/usr/bin/env python3
"""CPU-only prediction-rank dual-channel CT policy; full analysis is user-run."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from exchange_ml.triage_policy import calibrated_datasets, report
from run_count_rank import sha256, signature, require_hash, read_json, save_json, write_csv, lock
CODE = ('scripts/run_triage_policy.py', 'scripts/triage_policy.sbatch', 'scripts/submit_triage_policy.sh',
        'src/exchange_ml/triage_policy.py', 'scripts/run_count_rank.py',
        'src/exchange_ml/evaluator_sensitivity.py', 'src/exchange_ml/e1_statistics.py',
        'src/exchange_ml/e5_supplement_statistics.py', 'src/exchange_ml/partition_cv.py',
        'src/exchange_ml/e1_scoring.py', 'src/exchange_ml/metrics.py', 'src/exchange_ml/__init__.py')


def load_config(path: Path) -> dict:
    config = read_json(path)
    fixed = {'version':1, 'role':'posthoc_dual_channel_policy_not_confirmation',
        'parent_root':'outputs/evaluator_sensitivity_v1', 'candidate':'guard_dual_channel', 'baseline':'guard_p1',
        'channels':['partition_v1','presence'], 'calibration':'per_anatomy_prediction_only_midrank_ECDF',
        'combination':'maximum_no_fitted_weight', 'outer_folds':5, 'fold_seed':20261122,
        'review_fractions':[.1,.2], 'bootstrap_resamples':10000, 'bootstrap_seed':20261123,
        'followup_target':'pooled/member_mean/missing_volume/20pct_CT/candidate-minus-guard_p1',
        'original_E3_primary_modified':False, 'confirmation_ready':False}
    if any(config.get(k) != v for k,v in fixed.items()):
        raise ValueError('configuration differs from fixed dual-channel follow-up')
    return config


def output_path(config: dict, override: Path | None, smoke: bool) -> Path:
    dest = (override or ROOT/config['smoke_output_root' if smoke else 'output_root']).resolve()
    prefix = 'smoke_triage_policy' if smoke else 'triage_policy_development_'
    if dest.parent != (ROOT/'outputs').resolve() or not dest.name.startswith(prefix):
        raise ValueError('use a dedicated direct output child with the triage namespace')
    full, small = ((ROOT/config[k]).resolve() for k in ('output_root','smoke_output_root'))
    if full == small or full.is_relative_to(small) or small.is_relative_to(full) or dest == (full if smoke else small):
        raise ValueError('smoke and full output roots must be separate')
    return dest


def source_inputs(config: dict) -> tuple[dict, dict]:
    base = ROOT/config['parent_root'];path = base/'sensitivity_manifest.json';m = read_json(path)
    if m['status'] != 'complete' or m['provenance']['smoke_only'] or m['signature'] != signature(m['provenance']):
        raise ValueError('completed real evaluator-sensitivity parent required')
    if m['summary']['case_counts'] != {'validation':40,'heldout':100} or m['summary']['unit_counts'] != {'validation':80,'heldout':200}:
        raise ValueError('exact original PENGWIN cohort required')
    for name,digest in m['provenance']['code_sha256'].items(): require_hash(ROOT/name,digest)
    for name,digest in {**m['output_sha256'],**m['cache_sha256']}.items(): require_hash(base/name,digest)
    require_hash(ROOT/'configs/evaluator_sensitivity.json',m['provenance']['config_sha256'])
    datasets = {'validation':[], 'heldout':[]}
    with (base/'risks.csv').open(newline='') as handle:
        for raw in csv.DictReader(handle):
            split = raw['dataset_split']
            if split not in datasets: raise ValueError('unexpected source split')
            row = {k:raw[k] for k in ('case_id','scan_type','bone')}
            if raw['all_members_empty'] not in ('True','False'): raise ValueError('explicit source flag required')
            row['all_members_empty'] = raw['all_members_empty']=='True'
            row.update({k:float(v) for k,v in raw.items() if k not in ('dataset_split','case_id','scan_type','bone','all_members_empty')})
            datasets[split].append(row)
    if {s:len(rows) for s,rows in datasets.items()} != {'validation':80,'heldout':200}:
        raise ValueError('source table membership differs from parent')
    return datasets, {'parent_manifest_sha256':sha256(path), 'source_sha256':m['output_sha256']['risks.csv'],
                      'parent_output_and_cache_hashes_verified':len(m['output_sha256'])+len(m['cache_sha256'])}


def synthetic_inputs() -> dict:
    from exchange_ml.evaluator_sensitivity import ENDPOINTS,SOURCES
    result = {}
    rng = np.random.default_rng(55123)
    for split,n in [('validation',20),('heldout',10)]:
        rows=[]
        for i in range(n):
            scan='pelvic' if i<n//2 else 'femur'
            for bone in (('sacrum','left_hip','right_hip') if scan=='pelvic' else ('femur',)):
                row={'case_id':f'{split}_synthetic_{i:03}', 'scan_type':scan, 'bone':bone,
                     'partition_v1':float(rng.uniform(0,.1)), 'presence':float(rng.uniform()),
                     'fa_v2':float(rng.uniform()), 'all_members_empty':i==0}
                for s in SOURCES:
                    for e in ENDPOINTS:row[f'{s}_{e}']=float(i==0) if e=='missing_volume' else float(rng.uniform())
                rows.append(row)
        result[split]=rows
    return result


def run(config: dict, config_path: Path, output: Path, mode: str) -> None:
    with ExitStack() as stack:
        parent = ROOT/config['parent_root']
        stack.enter_context(lock(parent/'.sensitivity.lock',shared=True))
        datasets,inputs=source_inputs(config)
        print('DUAL-CHANNEL PREFLIGHT PASS:40 validation +100 inspected heldout; no volume reads',flush=True)
        if mode=='check':return
        smoke=mode=='smoke'
        prov={'config':config,'config_sha256':sha256(config_path),'inputs':inputs,'smoke_only':smoke,
              'output_root':str(output),'code_sha256':{p:sha256(ROOT/p) for p in CODE},
              'software':{p:importlib.metadata.version(p) for p in ('numpy','scipy')},'python':sys.version}
        ident=signature(prov);mp=output/'triage_manifest.json'
        if output.exists() and not mp.exists() and any(output.iterdir()):raise ValueError('foreign output artifacts')
        output.mkdir(parents=True,exist_ok=True);stack.enter_context(lock(output/'.triage.lock'))
        if mp.exists():
            old=read_json(mp)
            if old.get('signature')!=ident or old.get('provenance')!=prov:raise ValueError('run identity drift; use new output root')
            if old['status']=='complete':
                for name,digest in old['output_sha256'].items():require_hash(output/name,digest)
                print('DUAL-CHANNEL ALREADY COMPLETE (verified)');return
        state={'status':'running','signature':ident,'provenance':prov,'started_at_unix':time.time()}
        save_json(mp,state)
        try:
            if smoke:datasets=synthetic_inputs()
            calibrated,models=calibrated_datasets(datasets['validation'],datasets['heldout'],folds=config['outer_folds'],seed=config['fold_seed'])
            save_json(output/'calibration.json',models)
            combined={k:[] for k in ('ct_review','paired_differences','ct_omission','case_scores')}
            for split,rows in calibrated.items():
                tables=report(rows,resamples=32 if smoke else config['bootstrap_resamples'],seed=config['bootstrap_seed'])
                for name,table in tables.items():combined[name].extend({'dataset_split':split,**r} for r in table)
            for name,table in combined.items():write_csv(output/(name+'.csv'),table)
            summary={'smoke_only':smoke,'role':config['role'],'counts':{s:len({r['case_id'] for r in rows}) for s,rows in calibrated.items()},
                'prediction_only_calibration':True,'validation_calibration':'5-fold_CT_OOF','heldout_calibration':'all40_validation_reference',
                'fresh_confirmation':False,'original_E3_primary_modified':False,'new_success_threshold':None,
                'candidate':config['candidate'],'baseline':config['baseline'],'followup_target':config['followup_target'],
                'limitations':['posthoc_rules_on_previously_inspected_cohorts','CI_conditional_on_fixed_calibration_and_rules',
                               'empty_guard_assumes_expected_PENGWIN_anatomy_not_RibFrac','no_measured_reading_time_or_perfect_correction',
                               'midrank_calibration_changes_ordering_between_anatomies']}
            save_json(output/'summary.json',summary)
            require_hash(parent/'sensitivity_manifest.json',inputs['parent_manifest_sha256'])
            state.update(status='complete',summary=summary,completed_at_unix=time.time(),output_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.suffix in ('.csv','.json') and p!=mp})
            save_json(mp,state)
        except BaseException as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}');save_json(mp,state);raise
    print('DUAL-CHANNEL '+('SYNTHETIC SMOKE' if smoke else 'FULL')+' COMPLETE',flush=True)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','smoke','full'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/triage_policy.json')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();config=load_config(args.config)
    run(config,args.config,output_path(config,args.output_root,args.mode=='smoke'),args.mode)

if __name__=='__main__':main()
