from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from exchange_ml.triage_policy import (fit_reference,apply_reference,calibrated_datasets,case_arrays,report,
                                      CANDIDATE,BASELINE)
from run_triage_policy import synthetic_inputs,load_config,output_path

class TriageTests(unittest.TestCase):
    def setUp(self):self.datasets=synthetic_inputs()

    def test_midrank_equal_values_and_out_of_range_reference_scores(self):
        rows=self.datasets['validation']
        for r in rows:r['partition_v1']=.1;r['presence']=.2
        model=fit_reference(rows);test=copy.deepcopy(rows)
        observed=apply_reference(model,test)
        self.assertTrue(all(r['calibrated_p1']==.5 and r['calibrated_presence']==.5 for r in observed))
        for r in test:r['partition_v1']=0;r['presence']=1
        observed=apply_reference(model,test)
        self.assertTrue(all(r['calibrated_p1']==0 and r['calibrated_presence']==1 and r['dual_channel']==1 for r in observed))

    def test_calibration_uses_no_GT_outcomes(self):
        first=self.datasets['validation'];second=copy.deepcopy(first)
        for r in second:
            r['n_GT']=999
            for k in list(r):
                if k.startswith(('member_mean_','stage2_')):r[k]=999
        self.assertEqual(fit_reference(first),fit_reference(second))
        model=fit_reference(first)
        a=apply_reference(model,first);b=apply_reference(model,second)
        self.assertEqual([(r['calibrated_p1'],r['calibrated_presence'],r['dual_channel']) for r in a],
                         [(r['calibrated_p1'],r['calibrated_presence'],r['dual_channel']) for r in b])

    def test_outer_CT_groups_excluded_from_references_and_heldout_never_fits(self):
        data,models=calibrated_datasets(self.datasets['validation'],self.datasets['heldout'])
        ids={r['case_id'] for r in self.datasets['validation']}
        for record in models['fold_models']:
            self.assertFalse(set(record['model']['train_case_ids']) & set(record['test_case_ids']))
            self.assertEqual(set(record['model']['train_case_ids'])|set(record['test_case_ids']),ids)
        self.assertEqual(set(models['reference']['train_case_ids']),ids)
        self.assertEqual(len(data['validation']),40)
        assigned={}
        for r in data['validation']:
            self.assertTrue(r['case_id'] not in assigned or assigned[r['case_id']]==r['outer_fold'])
            assigned[r['case_id']]=r['outer_fold']

    def test_overlap_or_incomplete_anatomy_rejected(self):
        with self.assertRaises(ValueError):calibrated_datasets(self.datasets['validation'],self.datasets['validation'])
        with self.assertRaises(ValueError):fit_reference(self.datasets['validation'][1:])
        with self.assertRaises(ValueError):fit_reference(self.datasets['validation']+[self.datasets['validation'][0]])

    def test_fixed_guard_same_for_candidate_and_baseline(self):
        data,_=calibrated_datasets(self.datasets['validation'],self.datasets['heldout'])
        _,_,_,f1=case_arrays(data['heldout'],BASELINE)
        _,_,_,f2=case_arrays(data['heldout'],CANDIDATE)
        np.testing.assert_array_equal(f1,f2)
        self.assertEqual(f1.sum(),1)
        tables=report(data['heldout'],resamples=16)
        target=[r for r in tables['paired_differences'] if r['contrast_role']=='declared_followup_target_not_original_E3_primary']
        self.assertEqual(len(target),1)
        self.assertEqual(target[0]['baseline'],BASELINE)
        self.assertEqual(target[0]['review_fraction'],.2)
        self.assertEqual(target[0]['endpoint'],'missing_volume')

    def test_guard_budget_overflow_is_undefined_not_silently_clipped(self):
        data,_=calibrated_datasets(self.datasets['validation'],self.datasets['heldout'])
        for r in data['heldout']:r['all_members_empty']=True
        tables=report(data['heldout'],resamples=8)
        guarded=[r for r in tables['ct_review'] if r['policy']==CANDIDATE]
        self.assertTrue(all(r['retained_risk'] is None and r['status']=='guard_budget_infeasible' for r in guarded))

    def test_output_protections_and_fixed_design(self):
        config=load_config(ROOT/'configs/triage_policy.json')
        for dest in [ROOT/'outputs/evaluator_sensitivity_v1',ROOT/'outputs/stage3_h1_v2',ROOT/'__docs__',
                     Path('/tmp/Exchange-ML-E6-fast-v1'),ROOT/'outputs/smoke_triage_policy']:
            with self.assertRaises(ValueError):output_path(config,dest,False)
        with TemporaryDirectory() as directory:
            p=Path(directory)/'config.json';bad={**config,'channels':['partition_v1','n_GT']};p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_config(p)

    def test_real_launcher_afterok_cpu_only_and_failure_closed(self):
        for fail_at,mode,n in [(0,'full',2),(1,'full',1),(0,'smoke',1),(0,'check',1),(0,'bad',0)]:
            with TemporaryDirectory() as directory:
                folder=Path(directory);capture=folder/'calls.json';stub=folder/'sbatch'
                stub.write_text(f'#!{sys.executable}\nimport json,os,pathlib,sys\np=pathlib.Path(os.environ["TRIAGE_CAPTURE"])\nc=json.loads(p.read_text()) if p.exists() else []\nc.append(sys.argv[1:]);p.write_text(json.dumps(c))\nif len(c)==int(os.environ["TRIAGE_FAIL"]):sys.exit(1)\nprint(str(100+len(c))+";cluster")\n')
                stub.chmod(0o755)
                env=dict(os.environ,PATH=str(folder)+os.pathsep+os.environ['PATH'],TRIAGE_CAPTURE=str(capture),TRIAGE_FAIL=str(fail_at))
                result=subprocess.run(['bash',str(ROOT/'scripts/submit_triage_policy.sh'),mode],env=env,capture_output=True,text=True)
                calls=json.loads(capture.read_text()) if capture.exists() else []
                self.assertEqual(len(calls),n)
                self.assertEqual(result.returncode==0,not fail_at and mode!='bad',result.stderr)
                if n==2:self.assertIn('--dependency=afterok:101',calls[1])
        directives=[x for x in (ROOT/'scripts/triage_policy.sbatch').read_text().splitlines() if x.startswith('#SBATCH')]
        self.assertFalse(any('--gres' in x or '--gpus' in x for x in directives))

if __name__=='__main__':unittest.main()
