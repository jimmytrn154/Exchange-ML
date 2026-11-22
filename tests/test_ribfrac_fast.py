from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from exchange_ml.ribfrac_fast_backend import batch_budget, configure_paths, load_parent_checkpoint, sha256, unpack_one
import run_ribfrac_fast as runner


class RibfracFastTests(unittest.TestCase):
    def test_mmap_lifetimes_and_single_process_rng_contract(self):
        with patch.dict(os.environ,{'nnUNet_keep_files_open':'true'}):
            configure_paths(Path('/tmp/fast-paths'), 'member_01')
            self.assertEqual(os.environ['nnUNet_keep_files_open'],'false')
            self.assertEqual(os.environ['nnUNet_n_proc_DA'],'0')

    def test_patch_budget_is_preserved_with_explicit_rounding(self):
        self.assertEqual(batch_budget(5), {'batch_size': 5, 'iterations_per_epoch': 100,
                         'validation_iterations_per_epoch': 20, 'train_patches_per_epoch': 500,
                         'validation_patches_per_epoch': 100})
        six = batch_budget(6)
        self.assertEqual((six['iterations_per_epoch'], six['train_patches_per_epoch']), (83,498))
        self.assertEqual(six['validation_patches_per_epoch'],102)
        for invalid in (1,2,4,7):
            with self.assertRaises(ValueError): batch_budget(invalid)

    def test_cache_preserves_all_values_dtype_geometry_grid_and_source_bytes(self):
        with TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source';target=folder/'cache';source.mkdir();target.mkdir()
            image=np.random.default_rng(7).normal(size=(1,5,6,7)).astype(np.float32)
            labels=(image>0).astype(np.int8)
            path=source/'case';np.savez_compressed(path.with_suffix('.npz'),data=image,seg=labels)
            path.with_suffix('.pkl').write_bytes(b'properties-preserved')
            expected={s:sha256(path.with_suffix(s)) for s in ('.npz','.pkl')}
            result=unpack_one(path,target/'case',expected)
            self.assertEqual(len(result),4)
            for name,value in [('case.npy',image),('case_seg.npy',labels)]:
                output=np.load(target/name,mmap_mode='r')
                self.assertIsInstance(output,np.memmap)
                np.testing.assert_array_equal(output,value)
                self.assertEqual(output.dtype,value.dtype)
            self.assertEqual(expected,{s:sha256(path.with_suffix(s)) for s in expected})
            self.assertEqual(result,unpack_one(path,target/'case',expected))
            self.assertFalse(list(target.glob('*.tmp')))
            (target/'case.npz').unlink();(target/'case.npz').write_bytes(b'wrong alias')
            with self.assertRaisesRegex(ValueError,'alias drift'):unpack_one(path,target/'case',expected)

    def test_warmstart_bridge_preserves_parent_and_changes_only_explicit_identity(self):
        import torch
        with TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source.pth'
            payload={'e6_identity':'parent','e6_rng':{'marker':'rng'},'current_epoch':30,
                     'init_args':{'plans':{'batch_size':2}},'network_weights':{'w':torch.ones(2)},
                     'optimizer_state':{'marker':'momentum'},'grad_scaler_state':{'scale':128}}
            torch.save(payload,source);digest=sha256(source)
            trainer=SimpleNamespace(output_folder=str(folder/'new'),e6_identity='new-id',
                                    my_init_kwargs={'plans':{'batch_size':6}})
            def load(instance,filename):
                self.assertIsInstance(filename,str)
                state=torch.load(filename,map_location='cpu',weights_only=False)
                self.assertEqual(state['e6_identity'],'new-id')
                self.assertEqual(state['optimizer_state'],payload['optimizer_state'])
                self.assertEqual(state['grad_scaler_state'],payload['grad_scaler_state'])
                np.testing.assert_array_equal(state['network_weights']['w'].numpy(),np.ones(2))
                instance.current_epoch=state['current_epoch'];instance.my_init_kwargs=state['init_args']
            with patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.nnUNetTrainer.load_checkpoint',load):
                load_parent_checkpoint(trainer,source,'parent')
            self.assertEqual(trainer.current_epoch,30)
            self.assertEqual(trainer.my_init_kwargs['plans']['batch_size'],6)
            self.assertEqual(trainer._e6_resume_rng,payload['e6_rng'])
            self.assertEqual(sha256(source),digest)
            with self.assertRaisesRegex(ValueError,'identity/RNG'):load_parent_checkpoint(trainer,source,'wrong')

    def test_cache_metadata_includes_full_training_start_prerequisites(self):
        with TemporaryDirectory() as directory:
            source=Path(directory)/'source';target=Path(directory)/'cache';source.mkdir()
            original={'configurations':{'3d_fullres':{'batch_size':2,'patch_size':[96,160,160]}}}
            (source/'nnUNetPlans.json').write_text(json.dumps(original))
            for name in ('dataset.json','splits_final.json','dataset_fingerprint.json'):
                (source/name).write_text(json.dumps({'marker':name}))
            before={p.name:p.read_bytes() for p in source.iterdir()}
            outputs=runner.cache_metadata(source,target,5)
            self.assertEqual(len(outputs),4)
            self.assertEqual(json.loads((target/'nnUNetPlans.json').read_text())['configurations']['3d_fullres']['batch_size'],5)
            self.assertEqual((target/'dataset_fingerprint.json').read_bytes(),before['dataset_fingerprint.json'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            runner.verify_outputs(Path(directory),{str(Path(name).relative_to(directory)):digest for name,digest in outputs.items()})

    def test_external_inference_evaluation_and_risk_semantics_unchanged(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=functions(ROOT/'scripts/run_e6.py'),functions(ROOT/'scripts/run_ribfrac_fast.py')
        for name in ('infer','evaluate','verify_training','verify_preprocessing','model_folder'):
            self.assertEqual(old[name],new[name],name)

    def test_actual_cpu_optimizer_warmstart_and_larger_batch_backward(self):
        import importlib
        import random
        import torch
        from exchange_ml.e6_backend import trainer_class
        module=importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
        architecture={'network_class_name':'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                      'arch_kwargs':{'n_stages':3,'features_per_stage':[4,8,16],
                       'conv_op':'torch.nn.modules.conv.Conv3d','kernel_sizes':[[3,3,3]]*3,
                       'strides':[[1,1,1],[2,2,2],[2,2,2]],'n_conv_per_stage':[1]*3,
                       'n_conv_per_stage_decoder':[1]*2,'conv_bias':True,
                       'norm_op':'torch.nn.modules.instancenorm.InstanceNorm3d',
                       'norm_op_kwargs':{'eps':1e-5,'affine':True},'dropout_op':None,
                       'nonlin':'torch.nn.LeakyReLU','nonlin_kwargs':{'inplace':True}},
                      '_kw_requires_import':['conv_op','norm_op','nonlin']}
        plans={'dataset_name':'Dataset706_RibFracE6','plans_name':'nnUNetPlans',
               'original_median_spacing_after_transp':[1]*3,'original_median_shape_after_transp':[16]*3,
               'transpose_forward':[0,1,2],'transpose_backward':[0,1,2],
               'configurations':{'3d_fullres':{'data_identifier':'toy','batch_size':2,
                                  'batch_dice':True,'patch_size':[16]*3,'spacing':[1]*3,'architecture':architecture}}}
        dataset={'channel_names':{'0':'CT'},'labels':{'background':0,'fracture':1},'file_ending':'.nii.gz'}
        before_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with TemporaryDirectory() as directory, patch.dict(os.environ,{'nnUNet_compile':'false'}):
                folder=Path(directory)
                with patch.object(module,'nnUNet_preprocessed',str(folder/'preprocessed')),patch.object(module,'nnUNet_results',str(folder/'results')):
                    cls=trainer_class();old=cls(plans,'3d_fullres',0,dataset,False,torch.device('cpu'))
                    old.initialize();old.set_deep_supervision_enabled(True)
                    def batch(n):
                        return {'data':torch.randn(n,1,16,16,16),
                                'target':[torch.randint(0,2,(n,1,16,16,16)),torch.randint(0,2,(n,1,8,8,8))]}
                    self.assertTrue(np.isfinite(old.train_step(batch(2))['loss']))
                    source=folder/'source.pth'
                    torch.save({'network_weights':old.network.state_dict(),'optimizer_state':old.optimizer.state_dict(),
                                'grad_scaler_state':None,'current_epoch':30,'logging':old.logger.get_checkpoint(),
                                '_best_ema':old._best_ema,'init_args':old.my_init_kwargs,'e6_identity':'old',
                                'e6_rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                                          'torch':torch.get_rng_state(),'cuda':[]}},source)
                    digest=sha256(source)
                    changed=copy.deepcopy(plans);changed['configurations']['3d_fullres']['batch_size']=5
                    new=cls(changed,'3d_fullres',0,dataset,False,torch.device('cpu'));new.e6_identity='new'
                    load_parent_checkpoint(new,source,'old')
                    self.assertEqual((new.batch_size,new.current_epoch),(5,30))
                    self.assertEqual(new.my_init_kwargs['plans']['configurations']['3d_fullres']['batch_size'],5)
                    for name,value in old.network.state_dict().items():torch.testing.assert_close(value,new.network.state_dict()[name],rtol=0,atol=0)
                    self.assertEqual(len(old.optimizer.state),len(new.optimizer.state))
                    new.set_deep_supervision_enabled(True)
                    self.assertTrue(np.isfinite(new.train_step(batch(5))['loss']))
                    self.assertEqual(sha256(source),digest)
        finally:torch.set_num_threads(before_threads)

    def test_output_protects_parent_dataset_sources_and_old_results(self):
        config={'dataset_root':str(ROOT.parent/'dataset/RibFrac'),
                'speed_revision':{'parent_root':'/tmp/original-e6'},'output_root':'/tmp/new-e6'}
        for path in (Path('/tmp/original-e6'),Path('/tmp/original-e6/child'),ROOT,ROOT/'outputs',
                     ROOT/'src/child',ROOT/'outputs/partition_v3_development_v1',ROOT.parent/'dataset'):
            with self.assertRaises(ValueError):runner.output_path(config,path)
        self.assertEqual(runner.output_path(config,Path('/tmp/new-e6')),Path('/tmp/new-e6'))

    def test_launcher_dependencies_resources_and_failure_do_not_cancel_old_jobs(self):
        with TemporaryDirectory() as directory:
            folder=Path(directory);fake=folder/'sbatch';capture=folder/'calls.json'
            fake.write_text(f'#!{sys.executable}\nimport pathlib,json,os,sys\np=pathlib.Path(os.environ["FAST_CAPTURE"])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a))\nif len(a)==int(os.environ.get("FAST_FAIL","0")):sys.exit(1)\nprint(str(700+len(a))+";cluster")\n')
            fake.chmod(0o755)
            env=dict(os.environ,PATH=str(folder)+os.pathsep+os.environ['PATH'],FAST_CAPTURE=str(capture))
            result=subprocess.run(['bash',str(ROOT/'scripts/submit_ribfrac_fast.sh'),'full'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=json.loads(capture.read_text());self.assertEqual(len(calls),6)
            self.assertEqual([c[-1] for c in calls],['benchmark','cache','switch','train','infer','evaluate'])
            for i,call in enumerate(calls):
                if i:self.assertIn(f'--dependency=afterok:{700+i}',call)
                self.assertEqual('--gres=gpu:1' in call,i in (0,3,4))
                self.assertEqual('--array=0-2%2' in call,i in (3,4))
            capture.unlink();env['FAST_FAIL']='1'
            failed=subprocess.run(['bash',str(ROOT/'scripts/submit_ribfrac_fast.sh'),'full'],env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(len(json.loads(capture.read_text())),1)

    def test_handoff_rejects_bad_preconditions_before_cancelling(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config={'members':{},'speed_revision':{'parent_root':str(root/'parent'),'old_jobs':[1,2,3]}}
            with patch.object(runner,'verify_preprocessing',side_effect=ValueError('bad cache')),patch.object(runner.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'bad cache'):runner.switch_campaign(config,root)
                run.assert_not_called()
            wrong=SimpleNamespace(returncode=0,stdout='UserId=someone(99999) Command=wrong')
            with patch.object(runner,'verify_preprocessing'),patch.object(runner.subprocess,'run',return_value=wrong) as run:
                with self.assertRaisesRegex(ValueError,'ownership/command'):runner.switch_campaign(config,root)
                self.assertTrue(all(call.args[0][0]=='scontrol' for call in run.call_args_list))


if __name__=='__main__':unittest.main()
