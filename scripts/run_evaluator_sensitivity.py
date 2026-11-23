#!/usr/bin/env python3
"""CPU-only supplementary evaluator sensitivity; full volume reads are user-run."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from exchange_ml.e1_data import MEMBERS, bone_risk, clean_prediction, mask_bone, read_volume
from exchange_ml.evaluator_sensitivity import ENDPOINTS, SOURCES, overlap_metrics, report
from run_e4_e5 import load_inputs, locked, read_json, require_hash, signature, state, write_csv
from run_e1 import save_json, sha256

CODE=('scripts/run_evaluator_sensitivity.py','scripts/evaluator_sensitivity.sbatch',
      'scripts/submit_evaluator_sensitivity.sh','src/exchange_ml/evaluator_sensitivity.py',
      'src/exchange_ml/e5_supplement_statistics.py','src/exchange_ml/e1_statistics.py',
      'scripts/run_e4_e5.py','scripts/run_e1.py','scripts/run_e3.py',
      'src/exchange_ml/e1_data.py','src/exchange_ml/e1_scoring.py',
      'src/exchange_ml/instances.py','src/exchange_ml/metrics.py')
MODELS=(*MEMBERS,'stage2')


def load_config(path: Path)->dict:
    config=read_json(path)
    expected={'version':1,'role':'post_E3_supplementary_not_fresh_confirmation','iou_threshold':.1,
              'significant_overlap_thresholds':[.01,.1],'overlap_denominator':'maximum_object_volume',
              'bootstrap_resamples':10000,'bootstrap_seed':20261122,'review_fractions':[.1,.2],
              'dataset':'PENGWIN','scorer_modified':False,'original_endpoints_modified':False}
    if any(config.get(key)!=value for key,value in expected.items()):
        raise ValueError('configuration differs from declared supplementary sensitivity protocol')
    if config.get('e4_config')!='configs/e4_e5.json':raise ValueError('fixed E1/E3 parent required')
    return config


def output_path(config:dict,override:Path|None,smoke:bool)->Path:
    output=(override or ROOT/config['smoke_output_root' if smoke else 'output_root']).resolve()
    # Restrict to new direct children of repo outputs; E6 uses external roots.
    prefix='smoke_evaluator_sensitivity' if smoke else 'evaluator_sensitivity_'
    if not output.name.startswith(prefix) or output.parent != (ROOT/'outputs').resolve() or output.name in {
        'stage3_h1_v2','stage3_h1','stage2_backend','archive','local_copy_v1',
        'partition_v3_development_v1','partition_v4_development_v1','external_ribfrac',
        'smoke_partition_v3','smoke_partition_v4','smoke_e1','smoke_e3','smoke_e6'}:
        raise ValueError('dedicated output root must not overlap source/data/frozen outputs')
    full=(ROOT/config['output_root']).resolve();small=(ROOT/config['smoke_output_root']).resolve()
    if (full==small or full.is_relative_to(small) or small.is_relative_to(full)
            or output==(full if smoke else small)):
        raise ValueError('smoke/full outputs must be separate')
    return output


def validate_cache(record:dict,run_signature:str,states:dict,hashes:dict)->None:
    """Reject accidental source, identity or result corruption before reuse."""
    if (record.get('signature')!=run_signature or record.get('input_states')!=states
            or record.get('input_sha256')!=hashes
            or record.get('result_sha256')!=signature(record.get('result'))):
        raise ValueError('cached case source/identity/payload drift')


def process_case(case:dict,spec:dict,expected:list[dict],cleanup:dict)->dict:
    """At most one prediction and GT volume resident, all scores reused frozen."""
    paths=spec['paths'];gt,geom=read_volume(paths['gt'],None)
    spacing=tuple(geom['spacing'][::-1]);bones=('sacrum','left_hip','right_hip') if case['scan_type']=='pelvic' else ('femur',)
    expected_by={(r['bone'],r['model_id']):r for r in expected}
    models=[];perbone={bone:{} for bone in bones};empty={bone:[] for bone in bones}
    for model in MODELS:
        pred,_=read_volume(paths[model],geom)
        pred=clean_prediction(pred,spacing,cleanup['prune_mm3'],cleanup['connectivity'])
        for bone in bones:
            # Flag takes predictions and declared anatomy only, never GT values.
            pm=mask_bone(pred,bone);flag=not bool(pm.any())
            if model in MEMBERS:empty[bone].append(flag)
            gm=mask_bone(gt,bone);active=pm|gm
            if active.any():
                from scipy.ndimage import find_objects
                box=find_objects(active.astype(np.uint8),max_label=1)[0]
                p=np.where(pm[box],pred[box],0);g=np.where(gm[box],gt[box],0)
            else:p=g=np.zeros((1,1,1),np.uint8)
            original=bone_risk(pred,gt,spacing,bone,cleanup['iou_threshold'])
            previous=expected_by[bone,model]
            for name in ('merge_split','one_minus_f1','missing_volume','merge_errors','split_errors','n_GT','n_pred'):
                if previous[name] is None or not np.isclose(original[name],float(previous[name]),rtol=0,atol=1e-10):
                    raise ValueError(f'original risk reconciliation failed {case["case_id"]}/{bone}/{model}/{name}')
            if original['n_GT']==0:raise ValueError('expected annotated anatomy lacks GT')
            new=overlap_metrics(p,g)
            row={**case,'bone':bone,'model_id':model,
                 'original_merge_split':original['merge_split'],'original_one_minus_f1':original['one_minus_f1'],
                 **new}
            models.append(row);perbone[bone][model]=row
        del pred
    rows=[]
    for bone in bones:
        row={**case,'bone':bone,'all_members_empty':all(empty[bone])}
        for endpoint in ENDPOINTS:
            row['member_mean_'+endpoint]=float(np.mean([perbone[bone][m][endpoint] for m in MEMBERS]))
            row['stage2_'+endpoint]=perbone[bone]['stage2'][endpoint]
        rows.append(row)
    return {'risks':rows,'model_risks':models}


def synthetic_cases()->dict[str,list[dict]]:
    """Tiny fake volumes only. Identical structure counts/flags across scores."""
    rng=np.random.default_rng(20261122);rows=[]
    for i in range(10):
        scan='pelvic' if i<5 else 'femur';cid=f'SYN{i:03d}'
        for bone in (('sacrum','left_hip','right_hip') if scan=='pelvic' else ('femur',)):
            gt=np.ones((4,4,4),np.uint8);pred=gt.copy()
            if i==0:pred[:]=0
            elif i%2:pred.ravel()[-1:]=2
            else:pred.ravel()[32:]=2
            new=overlap_metrics(pred,gt);row={'case_id':cid,'scan_type':scan,'bone':bone,'all_members_empty':i==0}
            from exchange_ml.e1_scoring import SCORE_NAMES
            row.update({name:float(rng.uniform()) for name in SCORE_NAMES})
            for name in ('mean_entropy','p95_entropy'):row[name]*=float(np.log(2))
            for source in SOURCES:
                row[source+'_original_merge_split']=float(i!=0)
                row[source+'_original_one_minus_f1']=float(i==0)
                row.update({source+'_'+name:new[name] for name in ENDPOINTS if not name.startswith('original_')})
            rows.append(row)
    return {'synthetic':rows}


def preflight(config:dict)->tuple:
    e4_path=ROOT/config['e4_config']
    cases,data,specs,inputs=load_inputs(e4_path,read_json(e4_path))
    # load_inputs validates all original E1/E3 table/model identities and sizes.
    return cases,data,specs,inputs


def run(config_path:Path,config:dict,output:Path,smoke:bool,check_only:bool)->None:
    with ExitStack() as stack:
        base=ROOT/'outputs/stage3_h1_v2'
        stack.enter_context(locked(base/'.e1.lock',shared=True))
        held=base/'heldout';stack.enter_context(locked(held/'.evaluate.lock',shared=True))
        for model in MODELS:stack.enter_context(locked(held/'models'/model/'.lock',shared=True))
        cases,data,specs,inputs=preflight(config)
        print('EVALUATOR SENSITIVITY PREFLIGHT PASS: 40 validation + 100 heldout cases; original sources unchanged',flush=True)
        if check_only:return
        identity={'config':config,'config_sha256':sha256(config_path),'parent':inputs['identity'],
                  'output_root':str(output),
                  'code_sha256':{name:sha256(ROOT/name) for name in CODE},
                  'software':{name:importlib.metadata.version(name) for name in ('numpy','scipy','SimpleITK')},
                  'smoke_only':smoke}
        run_signature=signature(identity)
        if output.exists() and not (output/'sensitivity_manifest.json').exists():
            if any(output.iterdir()):raise ValueError('foreign artifacts without matching manifest')
        output.mkdir(parents=True,exist_ok=True)
        stack.enter_context(locked(output/'.sensitivity.lock'))
        path=output/'sensitivity_manifest.json'
        if path.exists():
            old=read_json(path)
            if old['signature']!=run_signature:raise ValueError('identity changed; retain old run and choose a new output root')
            if old['status']=='complete':
                for name,digest in {**old['output_sha256'],**old['cache_sha256']}.items():require_hash(output/name,digest)
                print('EVALUATOR SENSITIVITY ALREADY COMPLETE',flush=True);return
        elif any(p.name!='.sensitivity.lock' for p in output.iterdir()):raise ValueError('artifacts without matching manifest')
        manifest={'status':'running','signature':run_signature,'provenance':identity,'started_at_unix':time.time(),
                  'role':'synthetic_implementation_check' if smoke else config['role'],
                  'original_E3_primary_modified':False,'fresh_confirmation':False,'volumes_read':not smoke,
                  'cache_sha256':old.get('cache_sha256',{}) if path.exists() else {}}
        save_json(path,manifest)
        try:
            if smoke:
                datasets=synthetic_cases();models=[]
            else:
                datasets={};models=[]
                for split in ('validation','heldout'):
                    datasets[split]=[]
                    scores={(r['case_id'],r['bone']):r for r in data[split]['scores']}
                    for index,case in enumerate(cases[split],1):
                        cid=case['case_id'];spec=specs[split,cid]
                        states={key:state(p) for key,p in spec['paths'].items() if key!='image'}
                        # Content hashes, not only sizes/mtime, bind every reused source.
                        for key,digest in spec['hashes'].items():require_hash(spec['paths'][key],digest)
                        cache=output/'case_cache'/split/(cid+'.json')
                        if cache.exists():
                            relative=str(cache.relative_to(output))
                            if relative in manifest['cache_sha256']:
                                require_hash(cache,manifest['cache_sha256'][relative])
                            record=read_json(cache)
                            validate_cache(record,run_signature,states,spec['hashes'])
                        else:
                            expected=[r for r in data[split]['model_risks'] if r['case_id']==cid]
                            result=process_case(case,spec,expected,inputs['protocol'])
                            if any(state(spec['paths'][key])!=value for key,value in states.items()):raise ValueError('source changed during case')
                            record={'signature':run_signature,'input_states':states,'input_sha256':spec['hashes'],
                                    'result':result,'result_sha256':signature(result)}
                            save_json(cache,record)
                        manifest['cache_sha256'][str(cache.relative_to(output))]=sha256(cache)
                        save_json(path,manifest)
                        for row in record['result']['risks']:
                            original=scores[cid,row['bone']]
                            from exchange_ml.e1_scoring import SCORE_NAMES
                            datasets[split].append({**row,**{name:original[name] for name in SCORE_NAMES}})
                        models.extend({'dataset_split':split,**r} for r in record['result']['model_risks'])
                        print(f'{split} {index}/{len(cases[split])}: {cid} complete',flush=True)
            all_tables={name:[] for name in ('comparison','paired_differences','ct_review','ct_review_differences','ct_omission')}
            for split,rows in datasets.items():
                tables=report(rows,resamples=32 if smoke else config['bootstrap_resamples'],seed=config['bootstrap_seed'],review_fractions=config['review_fractions'])
                for name,values in tables.items():all_tables[name].extend({'dataset_split':split,**r} for r in values)
            for name,rows in all_tables.items():write_csv(output/(name+'.csv'),rows)
            write_csv(output/'risks.csv',[{'dataset_split':split,**r} for split,rows in datasets.items() for r in rows])
            if models:write_csv(output/'model_risks.csv',models)
            summary={'smoke_only':smoke,'case_counts':{split:len({r['case_id'] for r in rows}) for split,rows in datasets.items()},
                     'unit_counts':{split:len(rows) for split,rows in datasets.items()},
                     'thresholds':list(config['significant_overlap_thresholds']),'threshold_selection':'both_fixed_reported_no_winner',
                     'clinical_severity_threshold_validated':False,'input_scores_refit':False,
                     'CT_cost':'equal_one_CT_one_review','review_implies_perfect_correction':False}
            save_json(output/'summary.json',summary)
            manifest.update(status='complete',summary=summary,completed_at_unix=time.time(),
                   output_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.suffix in ('.csv','.json') and p!=path},
                   cache_sha256={str(p.relative_to(output)):sha256(p) for p in (output/'case_cache').rglob('*.json')})
            save_json(path,manifest);print('EVALUATOR SENSITIVITY '+('SYNTHETIC SMOKE' if smoke else 'FULL')+' COMPLETE',flush=True)
        except BaseException as error:
            manifest.update(status='failed',error=f'{type(error).__name__}: {error}');save_json(path,manifest);raise


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('check','smoke','full'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/evaluator_sensitivity.json')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();config=load_config(args.config)
    run(args.config,config,output_path(config,args.output_root,args.mode=='smoke'),args.mode=='smoke',args.mode=='check')

if __name__=='__main__':main()
