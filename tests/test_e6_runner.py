from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_e6 import stage_state, save_json, output_path, load_config, identity

class E6RunnerTests(unittest.TestCase):
    def test_manifest_resume_rejects_config_code_or_software_drift(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';provenance={'code':'first','software':'2.5.1'}
            state=stage_state(path,provenance);state['status']='complete';save_json(path,state)
            before=path.read_bytes()
            self.assertEqual(stage_state(path,provenance),state)
            self.assertEqual(path.read_bytes(),before)
            for changed in ({'code':'second','software':'2.5.1'},{'code':'first','software':'changed'}):
                with self.assertRaisesRegex(ValueError,'identity drift'):stage_state(path,changed)

    def test_roots_protect_dataset_and_old_experiments(self):
        config={'dataset_root':str(ROOT.parent/'dataset/RibFrac'),'output_root':'outputs/external_ribfrac'}
        for root in (ROOT,ROOT/'outputs',ROOT/'outputs/stage3_h1_v2/test',ROOT.parent/'dataset',ROOT.parent/'dataset/RibFrac/test'):
            with self.assertRaises(ValueError):output_path(config,root)
        self.assertEqual(output_path(config,Path('/tmp/e6-test')),Path('/tmp/e6-test'))

    def test_config_refuses_external_tuning_and_member_collisions(self):
        config=load_config(ROOT/'configs/e6.json')
        self.assertEqual(config['backend']['minimum_component_mm3'],0)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';config['members']['member_02']=config['members']['member_01'];save_json(path,config)
            with self.assertRaisesRegex(ValueError,'independently'):load_config(path)

if __name__=='__main__':unittest.main()
