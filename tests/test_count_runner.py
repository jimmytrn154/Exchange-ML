from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_count_rank as runner
from exchange_ml import count_rank, count_rank_reporting


def fixture(folder: Path) -> dict:
    config = runner.read_json(ROOT / 'configs/count_rank.json')
    config.update(parent_root='parent_v3', v4_root='parent_v4', output_root='outputs/full', smoke_output_root='outputs/smoke')
    for name in ('configs', 'parent_e1/validation', 'parent_v3', 'parent_v4'):
        (folder / name).mkdir(parents=True)
    rows, cases = [], []
    for index in range(40):
        scan = 'pelvic' if index < 20 else 'femur'
        cid = f'{index:03}'
        cases.append({'case_id': cid, 'scan_type': scan})
        for bone in (('sacrum','left_hip','right_hip') if scan == 'pelvic' else ('femur',)):
            rows.append({'case_id': cid, 'scan_type': scan, 'bone': bone,
                         **{name: .1 for name in runner.SCORE_NAMES}, 'graph_degree': .2,
                         'predicted_count_mean': index / 3 + 1, 'all_members_empty': False, 'n_GT': index + 1,
                         **{name: .3 for name in runner.RISKS}, 'pairwise_p1_g': index - 10})
    folds = runner.grouped_folds(rows, 5, 20261122)
    rows = [dict(row, outer_fold=folds[row['case_id']]) for row in rows]
    runner.write_csv(folder / 'configs/cases.csv', cases)
    runner.save_json(folder / 'configs/split.json', {'split': {'validation': [c['case_id'] for c in cases]}})
    e1config = {'split': 'configs/split.json', 'scan_metadata': 'configs/cases.csv'}
    runner.save_json(folder / 'configs/e1.json', e1config)
    for name in ('scores', 'risks'):
        runner.write_csv(folder / f'parent_e1/validation/{name}.csv', rows)
    tables = {f'validation/{name}.csv': runner.sha256(folder / f'parent_e1/validation/{name}.csv') for name in ('scores', 'risks')}
    e1prov = {'config_sha256': runner.sha256(folder / 'configs/e1.json'), 'code_sha256': {},
              'split_sha256': runner.sha256(folder / 'configs/split.json'),
              'scan_metadata_sha256': runner.sha256(folder / 'configs/cases.csv')}
    runner.save_json(folder / 'parent_e1/e1_manifest.json', {'status':'complete','smoke_only':False,
                     'provenance':e1prov,'signature':runner.signature(e1prov),
                     'case_ids':[c['case_id'] for c in cases],'output_sha256':tables})
    v3config = {'parent_root':'parent_e1','parent_config':'configs/e1.json', 'cv':{'outer_folds':5,'seed':20261122}}
    runner.save_json(folder / 'configs/partition_v3.json', v3config)
    for name in ('graph_features','oof_scores'):
        runner.write_csv(folder / f'parent_v3/{name}.csv',rows)
    v3prov = {'smoke_only':False,'config':v3config,'config_sha256':runner.sha256(folder/'configs/partition_v3.json'),
              'parent_manifest_sha256':runner.sha256(folder/'parent_e1/e1_manifest.json'),'source_tables':tables,'code_sha256':{}}
    runner.save_json(folder/'parent_v3/partition_manifest.json',{'status':'complete','summary':{'smoke_only':False},
                     'provenance':v3prov,'signature':runner.signature(v3prov), 'cache_sha256':{},
                     'output_sha256':{f'{name}.csv':runner.sha256(folder/f'parent_v3/{name}.csv') for name in ('graph_features','oof_scores')}})
    v4config = runner.read_json(ROOT/'configs/partition_v4.json')
    runner.save_json(folder/'configs/partition_v4.json',v4config)
    runner.write_csv(folder/'parent_v4/oof_scores.csv', rows)
    v4prov = {'smoke_only':False,'config':v4config,'config_sha256':runner.sha256(folder/'configs/partition_v4.json'),
              'code_sha256':{},'parent_manifest_sha256':runner.sha256(folder/'parent_v3/partition_manifest.json')}
    runner.save_json(folder/'parent_v4/rank_manifest.json', {'status':'complete','provenance':v4prov,
                     'signature':runner.signature(v4prov),'output_sha256':{'oof_scores.csv':runner.sha256(folder/'parent_v4/oof_scores.csv')}})
    (folder/'parent_v3/.partition.lock').touch();(folder/'parent_v4/.rank.lock').touch()
    runner.save_json(folder/'configs/count_rank.json', config)
    return config


class CountRunnerTests(unittest.TestCase):
    def test_real_parent_contract_count_and_frozen_v4_control(self):
        with TemporaryDirectory() as directory:
            folder=Path(directory);config=fixture(folder)
            with patch.object(runner,'ROOT',folder):
                inputs=runner.source_inputs(config)
                self.assertEqual(len(inputs['rows']),80)
                self.assertEqual(inputs['rows'][0]['predicted_count_mean'],1.)
                self.assertEqual(inputs['rows'][0]['pairwise_p1_g'],-10.)
                self.assertEqual(inputs['rows'][0]['n_GT'],1)
                self.assertFalse('n_GT' in count_rank.FEATURES)
                # A rehashed changed v4 score column still cannot alter frozen raw controls.
                p=folder/'parent_v4/oof_scores.csv';rows=runner.read_csv(p);rows[0]['partition_v1']='.2';runner.write_csv(p,rows)
                m=folder/'parent_v4/rank_manifest.json';value=runner.read_json(m);value['output_sha256']['oof_scores.csv']=runner.sha256(p);runner.save_json(m,value)
                with self.assertRaisesRegex(ValueError,'raw control'):
                    runner.source_inputs(config)

    def test_nested_smoke_resume_and_complete_rerun_are_verified(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder=Path(directory);config=fixture(folder);output=folder/'outputs/smoke'
            with patch.object(runner,'ROOT',folder),patch.object(runner,'CODE',()):
                with patch.object(count_rank_reporting,'report',side_effect=RuntimeError('report interruption')):
                    with self.assertRaisesRegex(RuntimeError,'report interruption'):
                        runner.run(config,folder/'configs/count_rank.json',output,smoke=True)
                self.assertEqual(runner.read_json(output/'rank_manifest.json')['stage'],'cv_complete')
                with patch.object(count_rank,'nested_cv',side_effect=AssertionError('must reuse checkpoint')):
                    runner.run(config,folder/'configs/count_rank.json',output,smoke=True)
                    before={p.name:p.read_bytes() for p in output.iterdir() if p.is_file()}
                    runner.run(config,folder/'configs/count_rank.json',output,smoke=True)
                    self.assertEqual(before,{p.name:p.read_bytes() for p in output.iterdir() if p.is_file()})
                self.assertTrue(all(r['case_id'].startswith('synthetic_') for r in runner.read_csv(output/'oof_scores.csv')))
                (output/'comparison.csv').write_text('corruption')
                with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
                    runner.run(config,folder/'configs/count_rank.json',output,smoke=True)

    def test_output_roots_and_single_family_configuration(self):
        config=runner.read_json(ROOT/'configs/count_rank.json')
        for path in (ROOT,ROOT/'outputs',ROOT/'src/new',ROOT/'outputs/partition_v4_development_v1/sub',
                     Path('/tmp/Exchange-ML-E6-fast-v1/sub'),ROOT/'outputs/smoke_count_ablation/sub'):
            with self.subTest(path=path),self.assertRaises(ValueError):runner.output_path(config,path,False)
        with TemporaryDirectory() as directory:
            p=Path(directory)/'config.json';runner.save_json(p,dict(config,confirmation_ready=True))
            with self.assertRaisesRegex(ValueError,'cannot authorize'):runner.load_config(p)
            runner.save_json(p,dict(config,features=['partition_v1','n_GT']))
            with self.assertRaisesRegex(ValueError,'fixed'):runner.load_config(p)

    def test_launcher_afterok_and_fail_closed_submission(self):
        for mode,fail_at,expected in (('full',0,2),('smoke',0,1),('full',1,1),('invalid',0,0)):
            with self.subTest(mode=mode,fail_at=fail_at),TemporaryDirectory() as directory:
                folder=Path(directory);capture=folder/'calls.json';fake=folder/'sbatch'
                fake.write_text(f'#!{sys.executable}\nimport json,os,pathlib,sys\np=pathlib.Path(os.environ["COUNT_CAPTURE"])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a))\nif len(a)==int(os.environ["COUNT_FAIL"]):sys.exit(1)\nprint(str(100+len(a))+";cluster")\n')
                fake.chmod(0o755)
                env=dict(os.environ,PATH=str(folder)+os.pathsep+os.environ['PATH'],COUNT_CAPTURE=str(capture),COUNT_FAIL=str(fail_at))
                result=subprocess.run(['bash',str(ROOT/'scripts/submit_count_rank.sh'),mode],env=env,text=True,capture_output=True)
                calls=json.loads(capture.read_text()) if capture.exists() else []
                self.assertEqual(len(calls),expected)
                self.assertEqual(result.returncode==0,fail_at==0 and mode!='invalid',result.stderr)
                if len(calls)==2:self.assertEqual(calls[1],['--parsable','--dependency=afterok:101','scripts/count_rank.sbatch','full'])
        directives=[line for line in (ROOT/'scripts/count_rank.sbatch').read_text().splitlines() if line.startswith('#SBATCH')]
        self.assertFalse(any('--gres' in line or '--gpus' in line for line in directives))


if __name__=='__main__':unittest.main()
