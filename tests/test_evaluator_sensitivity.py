from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from exchange_ml.evaluator_sensitivity import (ct_arrays, guarded_retention,
                                              overlap_metrics, report)
from run_evaluator_sensitivity import (load_config, output_path, process_case,
                                       signature, synthetic_cases, validate_cache)


class OverlapSensitivityTests(unittest.TestCase):
    def test_significant_fraction_is_both_sided_not_minimum_volume(self):
        gt=np.ones((4,4,4),np.uint8);tiny=gt.copy();tiny.ravel()[-1]=2
        result=overlap_metrics(tiny,gt)
        self.assertEqual(result['significant_0p01_merge_split'],1)
        self.assertEqual(result['significant_0p1_merge_split'],0)
        balanced=gt.copy();balanced.ravel()[32:]=2
        self.assertEqual(overlap_metrics(balanced,gt)['significant_0p1_merge_split'],1)

    def test_many_to_one_cannot_obtain_perfect_one_to_one_f1(self):
        gt=np.ones((4,4,4),np.uint8);gt.ravel()[32:]=2
        result=overlap_metrics(np.ones_like(gt),gt)
        self.assertEqual(result['one_to_one_true_positives'],1)
        self.assertAlmostEqual(result['one_to_one_one_minus_f1'],1/3)
        self.assertEqual(result['significant_0p1_merge_split'],.5)

    def test_prediction_id_permutation_and_directional_count_symmetry(self):
        gt=np.ones((4,4,4),np.uint16);pred=gt.copy();pred.ravel()[32:]=17
        first=overlap_metrics(pred,gt);second=overlap_metrics(np.where(pred==17,93,pred),gt)
        self.assertEqual(first,second)
        reverse=overlap_metrics(gt,pred)
        self.assertEqual(first['significant_0p1_splits'],reverse['significant_0p1_merges'])

    def test_empty_omission_and_valid_negative_have_distinct_endpoints(self):
        gt=np.ones((4,4,4),np.uint8);zero=gt*0
        missing=overlap_metrics(zero,gt);negative=overlap_metrics(zero,zero)
        self.assertEqual(missing['missing_volume'],1)
        self.assertEqual(missing['one_to_one_one_minus_f1'],1)
        self.assertEqual(missing['significant_0p1_merge_split'],0)
        self.assertIsNone(negative['missing_volume'])
        self.assertIsNone(negative['significant_0p1_merge_split'])
        self.assertEqual(negative['one_to_one_one_minus_f1'],0)

    def test_shapes_labels_thresholds_checked(self):
        a=np.zeros((2,2,2),np.uint8)
        for bad in [a.astype(float),np.full_like(a,-1,dtype=int),np.zeros((2,2),int)]:
            with self.assertRaises(ValueError):overlap_metrics(bad,a)
        with self.assertRaises(ValueError):overlap_metrics(a,a,thresholds=[0])


class CTPolicyTests(unittest.TestCase):
    def test_exact_cutoff_ties_match_enumerated_expected_review(self):
        scores=np.array([[0.,1.,1.,2.]])
        y=np.array([[[1.],[2.],[6.],[8.]]])
        value,feasible=guarded_retention(scores,y,np.zeros_like(scores,bool),2,guarded=False)
        # Keep score0 and one of tied score1, each with equal probability.
        self.assertAlmostEqual(value[0,0],((1+2)/2+(1+6)/2)/2)
        self.assertTrue(feasible[0])

    def test_guard_priority_and_infeasibility_are_explicit(self):
        scores=np.array([[0.,1.,2.,3.]])
        y=np.arange(4.)[None,:,None];flags=np.array([[True,False,False,False]])
        value,feasible=guarded_retention(scores,y,flags,3,guarded=True)
        self.assertTrue(feasible[0]);self.assertEqual(value[0,0],2)
        value,feasible=guarded_retention(scores,y,np.ones_like(flags),3,guarded=True)
        self.assertFalse(feasible[0]);self.assertTrue(np.isnan(value).all())

    def test_complete_anatomy_and_explicit_prediction_only_flags(self):
        rows=synthetic_cases()['synthetic'];meta,x,y,f=ct_arrays(rows,'partition_v1')
        self.assertEqual(len(meta),10);self.assertTrue(f[0]);self.assertEqual(f.sum(),1)
        with self.assertRaises(ValueError):ct_arrays(rows[1:],'partition_v1')

    def test_no_omission_positives_is_undefined(self):
        rows=[r for r in synthetic_cases()['synthetic'] if r['case_id']!='SYN000']
        tables=report(rows,resamples=8,seed=7)
        self.assertTrue(all(r['status']=='no_positive_cts' and r['sensitivity'] is None
                            for r in tables['ct_omission']))
        self.assertTrue(tables['paired_differences'])
        self.assertTrue(all(r['baseline']=='partition_v1' for r in tables['paired_differences']))

    def test_zero_positive_bootstrap_draws_are_reported_not_dropped(self):
        tables=report(synthetic_cases()['synthetic'],resamples=64,seed=7)
        selected=[r for r in tables['ct_omission'] if r['stratum']=='pooled']
        self.assertTrue(selected)
        self.assertTrue(all(0<r['bootstrap_positive_fraction']<1 for r in selected))
        self.assertTrue(all(r['ci_low'] is None and r['ci_status']=='no_positive_or_infeasible_bootstrap_draws'
                            for r in selected))


class RunnerProtocolTests(unittest.TestCase):
    def test_fixed_threshold_config_and_changed_identity_rejected(self):
        config=load_config(ROOT/'configs/evaluator_sensitivity.json')
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as directory:
            changed=copy.deepcopy(config);changed['significant_overlap_thresholds']=[.05]
            p=Path(directory)/'config.json';p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):load_config(p)

    def test_root_protection_and_smoke_full_separation(self):
        config=load_config(ROOT/'configs/evaluator_sensitivity.json')
        for path in [ROOT,ROOT/'outputs',ROOT/'outputs/stage3_h1_v2',ROOT/'configs',
                     ROOT/'outputs/smoke_e2',ROOT/'outputs/smoke_ribfrac_fast',
                     Path('/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1')]:
            with self.assertRaises(ValueError):output_path(config,path,False)
        with self.assertRaises(ValueError):output_path(config,ROOT/config['output_root'],True)
        self.assertNotEqual(output_path(config,None,False),output_path(config,None,True))

    def test_cache_payload_input_and_root_identity_corruption_rejected(self):
        payload={'risks':[{'value':1.0}]}
        record={'signature':'bound_root_identity','input_states':{'x':{'bytes':2}},
                'input_sha256':{'x':'hash'},'result':payload,'result_sha256':signature(payload)}
        validate_cache(record,'bound_root_identity',record['input_states'],record['input_sha256'])
        changed=copy.deepcopy(record);changed['result']['risks'][0]['value']=2.0
        with self.assertRaises(ValueError):validate_cache(changed,'bound_root_identity',record['input_states'],record['input_sha256'])
        with self.assertRaises(ValueError):validate_cache(record,'different_root_identity',record['input_states'],record['input_sha256'])

    def test_one_synthetic_volume_end_to_end_and_original_reconciliation(self):
        from tempfile import TemporaryDirectory
        import SimpleITK as sitk
        from exchange_ml.e1_data import MEMBERS,bone_risk
        with TemporaryDirectory() as directory:
            paths={};gt=np.full((4,4,4),151,np.uint8);pred=gt.copy();pred.ravel()[32:]=152
            for name,array in [('gt',gt),*[(m,pred) for m in (*MEMBERS,'stage2')]]:
                image=sitk.GetImageFromArray(array);image.SetSpacing((10.,10.,10.))
                p=Path(directory)/(name+'.mha');sitk.WriteImage(image,str(p));paths[name]=p
            expected=[{'bone':'femur','model_id':m,**bone_risk(pred,gt,(10.,10.,10.),'femur',.1)}
                      for m in (*MEMBERS,'stage2')]
            case={'case_id':'SYN','scan_type':'femur'};cleanup={'prune_mm3':1000,'connectivity':26,'iou_threshold':.1}
            result=process_case(case,{'paths':paths},expected,cleanup)
            self.assertEqual(len(result['risks']),1);self.assertEqual(len(result['model_risks']),4)
            self.assertFalse(result['risks'][0]['all_members_empty'])
            expected[0]['merge_split']+=1
            with self.assertRaisesRegex(ValueError,'reconciliation'):
                process_case(case,{'paths':paths},expected,cleanup)

    def test_slurm_requests_cpu_only_and_gates_full_on_smoke(self):
        batch=(ROOT/'scripts/evaluator_sensitivity.sbatch').read_text()
        submit=(ROOT/'scripts/submit_evaluator_sensitivity.sh').read_text()
        self.assertNotIn('--gres',batch);self.assertNotIn('CUDA_VISIBLE_DEVICES',batch)
        self.assertIn('--dependency="afterok:$smoke"',submit)
        self.assertIn('run_evaluator_sensitivity.py "$mode"',batch)

if __name__=='__main__':unittest.main()
