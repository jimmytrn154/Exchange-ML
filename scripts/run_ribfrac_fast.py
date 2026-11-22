#!/usr/bin/env python3
"""E6 performance revision: measured batch5/6, NPY cache and explicit warmstart.

Inference/evaluation functions are preserved from frozen run_e6.py. Original
E6 outputs remain untouched. Full conversion/handoff/training are user-run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exchange_ml.ribfrac_fast_backend import (configure_paths, disk_estimate, fingerprint, internal_split,
    lock, make_predictor, make_trainer, read_json, reconstruct_instances, require_cuda,
    require_disk, require_hash, planning_threads, save_json, seed_everything, sha256, signature, software, write_volume)
from exchange_ml.e6_data import binary_target, inventory, require_geometry, geometry
from exchange_ml.ribfrac_fast_backend import parent_metadata, unpack_one, benchmark, profile_case


def load_config(path: Path) -> dict:
    from exchange_ml.ribfrac_fast_backend import parent_metadata, batch_budget
    settings = read_json(path)
    if (settings['target_train_patches_per_epoch'], settings['target_validation_patches_per_epoch']) != (500, 100):
        raise ValueError('This revision preserves the original 500/100 patch budget approximately')
    if (len(settings['old_jobs']) != 3 or len(set(settings['old_jobs'])) != 3
            or any(type(job) is not int or job < 1 for job in settings['old_jobs'])):
        raise ValueError('Exactly three distinct original job IDs required')
    parent, _, config = parent_metadata(settings)
    config = json.loads(json.dumps(config))
    root = Path(os.environ.get('RIBFRAC_FAST_ROOT', settings['output_root'])).resolve()
    selection = root / 'speed_selection.json'
    if selection.exists():
        picked = read_json(selection)
        if picked['status'] == 'complete':
            if picked['settings'] != settings or picked['backend_sha256'] != sha256(ROOT / 'src/exchange_ml/ribfrac_fast_backend.py'):
                raise ValueError('Batch benchmark settings/code drift; choose a new output root')
            budget = batch_budget(picked['selected_batch'])
            config['backend'].update({k: budget[k] for k in ('batch_size','iterations_per_epoch','validation_iterations_per_epoch')})
    config['output_root'] = str(root)
    config['speed_revision'] = settings
    config['scope'] += ' Performance revision: larger batch, NPY cache, recorded mixed-batch warmstart prefixes.'
    config['speed_selection_sha256'] = sha256(selection) if selection.exists() and picked['status'] == 'complete' else None
    if (config['dataset_id'], config['dataset_name'], config['train_ids'], config['external_ids']) != (
            706, 'Dataset706_RibFracE6', [1, 420], [421, 500]):
        raise ValueError('E6 requires Dataset706 and all 420 training / 80 external cases')
    if config['members'] != {'member_01': 32001, 'member_02': 32002, 'member_03': 32003}:
        raise ValueError('E6 requires three independently seeded members')
    if config['backend']['connectivity'] != 26 or config['backend']['minimum_component_mm3'] != 0:
        raise ValueError('E6 must retain small lesions with fixed 26-connectivity')
    if config['backend']['threshold'] != .5 or config['backend']['checkpoint'] != 'checkpoint_final.pth':
        raise ValueError('Do not tune E6 threshold or checkpoint on external validation')
    if config['backend']['augmentation_workers'] != 0 or config['backend']['unpack_dataset']:
        raise ValueError('E6 uses single-process augmentation and compressed preprocessed inputs')
    if config['backend']['nnunet_version'] != software()['nnunetv2']:
        raise ValueError('Installed nnunetv2 version differs from config')
    return config


def output_path(config: dict, override: Path | None) -> Path:
    root = (override or Path(os.environ.get('E6_OUTPUT_ROOT', config['output_root']))).resolve()
    dataset = Path(config['dataset_root']).resolve()
    parent = Path(config['speed_revision']['parent_root']).resolve()
    if root == parent or parent in root.parents or root in parent.parents:
        raise ValueError('Performance fork must not overlap original E6 outputs')
    for name in ('src', 'scripts', 'configs', 'splits', 'tests', '__docs__', 'paper',
                 'outputs/partition_v3_development_v1', 'outputs/partition_v4_development_v1',
                 'outputs/smoke_partition_v3', 'outputs/smoke_partition_v4'):
        protected = (ROOT / name).resolve()
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError('Performance output overlaps protected source/artifacts')
    if root == ROOT or root == ROOT / 'outputs' or root in dataset.parents or dataset == root or dataset in root.parents:
        raise ValueError('E6 output root must not overlap the dataset or replace the repository/outputs root')
    for old in ('stage3_h1', 'stage3_h1_v2', 'stage2_backend', 'archive', 'local_copy_v1'):
        protected = ROOT / 'outputs' / old
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError('E6 output root overlaps protected existing artifacts')
    return root


def identity(config: dict, root: Path, *, smoke: bool = False) -> dict:
    files = [Path(__file__), *sorted((ROOT / 'src/exchange_ml').glob('e6_*.py')),
             ROOT / 'src/exchange_ml/e1_statistics.py', ROOT / 'src/exchange_ml/e1_scoring.py',
             ROOT / 'src/exchange_ml/ribfrac_fast_backend.py',
             ROOT / 'scripts/ribfrac_fast.sbatch', ROOT / 'scripts/submit_ribfrac_fast.sh']
    return {'config': config, 'output_root': str(root), 'software': software(),
            'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in files},
            'smoke_only': smoke, 'role': 'synthetic_smoke_only' if smoke else 'external_ribfrac_performance_revision'}


def stage_state(path: Path, provenance: dict) -> dict:
    expected = signature(provenance)
    if path.exists():
        value = read_json(path)
        if value['signature'] != expected or value['provenance'] != provenance:
            raise ValueError(f'E6 identity drift: {path}; keep old outputs and use a new output root')
        return value
    return {'status': 'running', 'signature': expected, 'provenance': provenance,
            'started_at_unix': time.time(), 'records': {}}


def verify_outputs(folder: Path, hashes: dict) -> None:
    for name, value in hashes.items():
        require_hash(folder / name, value['sha256'] if isinstance(value, dict) else value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Cannot write an undefined E6 table: {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)






def preprocessing_identity(config: dict, root: Path) -> dict:
    parent = Path(config['speed_revision']['parent_root'])
    source = parent / 'preprocessing_manifest.json'
    return {**identity(config, root), 'parent_preprocessing_sha256': sha256(source),
            'parent_preparation_sha256': sha256(parent / 'preparation_manifest.json')}






def model_folder(config: dict, root: Path, member: str) -> Path:
    return (root / 'backend/results' / member / config['dataset_name'] /
            f"E6Trainer__{config['backend']['plans_name']}__{config['backend']['configuration']}" / 'fold_0')


def verify_preprocessing(config: dict, root: Path) -> dict:
    path = root / 'preprocessing_manifest.json'
    manifest = read_json(path)
    expected = preprocessing_identity(config, root)
    if manifest['status'] != 'complete' or manifest['provenance'] != expected or manifest['signature'] != signature(expected):
        raise ValueError('Complete matching E6 preprocessing identity required')
    verify_outputs(root, manifest['output_sha256'])
    return manifest


def train(config: dict, root: Path, member: str) -> None:
    configure_paths(root, member)
    require_cuda()
    print(f'Verifying full NPY-cache hashes before training {member}; this may take several minutes', flush=True)
    preprocessing = verify_preprocessing(config, root)
    seed = config['members'][member]
    provenance = {**identity(config, root), 'member': member, 'seed': seed,
                  'preprocessing_manifest_sha256': sha256(root / 'preprocessing_manifest.json'),
                  'internal_split': preprocessing['internal_split']}
    folder = model_folder(config, root, member)
    manifest_path = root / 'models' / member / 'training_manifest.json'
    with lock(root / 'models' / member / '.train.lock'):
        manifest = stage_state(manifest_path, provenance)
        final = folder / 'checkpoint_final.pth'
        if manifest['status'] == 'complete':
            require_hash(final, manifest['checkpoint']['sha256'])
            print(f'E6 TRAINING ALREADY COMPLETE: {member}', flush=True)
            return
        save_json(manifest_path, manifest)
        seed_everything(seed)
        import torch
        torch.backends.cudnn.benchmark = config['speed_revision']['cudnn_benchmark']
        trainer = make_trainer(root, config, member)
        trainer.e6_identity = signature(provenance)
        latest = folder / 'checkpoint_latest.pth'
        if final.exists():
            trainer.load_checkpoint(str(final))
            if trainer.current_epoch < config['backend']['epochs']:
                raise ValueError('Final checkpoint precedes configured epochs')
        else:
            if latest.exists():
                trainer.load_checkpoint(str(latest))
            elif (root / 'warmstarts' / member / 'checkpoint.pth').exists():
                from exchange_ml.ribfrac_fast_backend import load_parent_checkpoint
                record = read_json(root / 'warmstarts' / member / 'source.json')
                source = root / 'warmstarts' / member / 'checkpoint.pth'
                require_hash(source, record['sha256'])
                load_parent_checkpoint(trainer, source, record['parent_training_signature'])
            elif any(folder.glob('checkpoint*.pth')):
                raise ValueError('Partial E6 checkpoints without latest/final; inspect manually')
            started = time.monotonic()
            trainer.run_training()
            manifest.setdefault('run_segments', []).append({'seconds': time.monotonic()-started,
                'ending_epoch': trainer.current_epoch, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
        import torch
        state = torch.load(str(final), map_location='cpu')
        if state['current_epoch'] < config['backend']['epochs']:
            raise ValueError('Training ended before fixed final epoch')
        manifest.update(status='complete', completed_at_unix=time.time(), checkpoint=fingerprint(final),
                        checkpoint_path=str(final), final_epoch=state['current_epoch'],
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                        rng_resume='single-process augmentation; hardware-dependent kernels may differ')
        save_json(manifest_path, manifest)
    print(f'E6 TRAINING COMPLETE: {member}; fixed final checkpoint', flush=True)


def verify_training(config: dict, root: Path, member: str) -> dict:
    path = root / 'models' / member / 'training_manifest.json'
    manifest = read_json(path)
    preprocessing = read_json(root / 'preprocessing_manifest.json')
    expected = {**identity(config, root), 'member': member, 'seed': config['members'][member],
                'preprocessing_manifest_sha256': sha256(root / 'preprocessing_manifest.json'),
                'internal_split': preprocessing['internal_split']}
    if manifest['status'] != 'complete' or manifest['provenance'] != expected or manifest['signature'] != signature(expected):
        raise ValueError(f'Complete matching training identity required: {member}')
    for rel in ('dataset.json', config['backend']['plans_name'] + '.json', 'splits_final.json'):
        file = root / 'backend/nnUNet_preprocessed' / config['dataset_name'] / rel
        require_hash(file, preprocessing['output_sha256'][str(file.relative_to(root))])
    if manifest['provenance']['seed'] != config['members'][member]:
        raise ValueError('Member seed differs from fixed E6 seed')
    checkpoint = model_folder(config, root, member) / 'checkpoint_final.pth'
    require_hash(checkpoint, manifest['checkpoint']['sha256'])
    if manifest['checkpoint_path'] != str(checkpoint):
        raise ValueError('Training checkpoint path differs')
    require_hash(root / 'preprocessing_manifest.json', manifest['provenance']['preprocessing_manifest_sha256'])
    return manifest


def infer(config: dict, root: Path, member: str) -> None:
    configure_paths(root, member)
    require_cuda()
    training = verify_training(config, root, member)
    cases = {cid: r for cid, r in inventory(Path(config['dataset_root'])).items() if r['split'] == 'validation'}
    folder = root / 'models' / member
    provenance = {**identity(config, root), 'member': member,
                  'training_manifest_sha256': sha256(folder / 'training_manifest.json'),
                  'checkpoint_sha256': training['checkpoint']['sha256'], 'case_ids': sorted(cases)}
    path = folder / 'inference_manifest.json'
    with lock(folder / '.infer.lock'):
        manifest = stage_state(path, provenance)
        was_complete = manifest['status'] == 'complete'
        if was_complete and set(manifest['records']) != set(cases):
            raise ValueError('Completed inference lacks exact 80-case membership')
        if not was_complete:
            save_json(path, manifest)
        predictor = None
        for cid, record in sorted(cases.items()):
            prediction_path, probability_path = folder / 'predictions' / f'{cid}.mha', folder / 'probabilities' / f'{cid}.mha'
            if cid in manifest['records']:
                saved = manifest['records'][cid]
                require_hash(record['image'], saved['source_image']['sha256'])
                require_hash(prediction_path, saved['prediction']['sha256'])
                require_hash(probability_path, saved['probability']['sha256'])
                continue
            started = time.monotonic()
            if predictor is None:
                predictor = make_predictor(root, config, member, model_folder(config, root, member) / 'checkpoint_final.pth')
            before = fingerprint(record['image'])
            image = sitk.ReadImage(str(record['image']))
            ct = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
            _, probabilities = predictor.predict_single_npy_array(
                ct[None], {'spacing': tuple(image.GetSpacing()[::-1])}, save_or_return_probabilities=True)
            if probabilities.shape != (2, *ct.shape):
                raise ValueError(f'Expected true binary softmax [2,Z,Y,X], got {probabilities.shape}')
            probability = np.asarray(probabilities[1], dtype=np.float32)
            labels = reconstruct_instances(probability, threshold=config['backend']['threshold'],
                connectivity=26, minimum_component_mm3=0, spacing_zyx=image.GetSpacing()[::-1])
            pred_fp = write_volume(prediction_path, labels, image)
            prob_fp = write_volume(probability_path, probability, image)
            require_hash(record['image'], before['sha256'])
            manifest['records'][cid] = {'source_image': before, 'prediction': pred_fp, 'probability': prob_fp,
                'seconds_including_first_model_load': time.monotonic()-started,
                'n_pred_instances': int(labels.max()), 'all_empty': not bool(labels.any())}
            save_json(path, manifest)
        if not was_complete:
            manifest.update(status='complete', completed_at_unix=time.time())
            save_json(path, manifest)
    print(f'E6 INFERENCE COMPLETE: {member}; 80 external cases', flush=True)


def evaluate(config: dict, root: Path) -> None:
    from itertools import combinations
    from exchange_ml.e6_evaluation import score_case, evaluate_predictions, report_tables
    from exchange_ml.e6_data import validate_labels, read_header
    cases = {cid: r for cid, r in inventory(Path(config['dataset_root'])).items() if r['split'] == 'validation'}
    members = list(config['members'])
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(lock(root / '.evaluate.lock'))
        manifests = {}
        for member in members:
            stack.enter_context(lock(root / 'models' / member / '.infer.lock'))
            training = verify_training(config, root, member)
            path = root / 'models' / member / 'inference_manifest.json'
            manifest = read_json(path)
            expected = {**identity(config, root), 'member': member,
                'training_manifest_sha256': sha256(path.parent / 'training_manifest.json'),
                'checkpoint_sha256': training['checkpoint']['sha256'], 'case_ids': sorted(cases)}
            if (manifest['status'] != 'complete' or set(manifest['records']) != set(cases)
                    or manifest['provenance'] != expected or manifest['signature'] != signature(expected)):
                raise ValueError(f'Incomplete or changed external inference identity: {member}')
            manifests[member] = manifest
        hashes = [m['provenance']['checkpoint_sha256'] for m in manifests.values()]
        if len(set(hashes)) != 3:
            raise ValueError('E6 requires three distinct independently seeded model checkpoints')
        provenance = {**identity(config, root), 'inference_manifest_sha256': {
            m: sha256(root / 'models' / m / 'inference_manifest.json') for m in members},
            'external_metadata_sha256': {str(p): sha256(p) for p in sorted({r['metadata'] for r in cases.values()})}}
        path = root / 'e6_manifest.json'
        result = stage_state(path, provenance)
        was_complete = result['status'] == 'complete'
        if not was_complete:
            save_json(path, result)
        scores, pairs, risks, model_risks = [], [], [], []
        for cid, record in sorted(cases.items()):
            started = time.monotonic()
            # Verify all prediction/probability/image inputs even on completed resume.
            source_hashes = {}
            for member in members:
                saved = manifests[member]['records'][cid]
                require_hash(record['image'], saved['source_image']['sha256'])
                for kind, directory in [('prediction', 'predictions'), ('probability', 'probabilities')]:
                    file = root / 'models' / member / directory / f'{cid}.mha'
                    require_hash(file, saved[kind]['sha256'])
                    source_hashes[str(file.relative_to(root))] = saved[kind]['sha256']
            cache_path = root / 'cache' / f'{cid}.json'
            if cid in result['records'] and 'cache_sha256' in result['records'][cid]:
                previous = result['records'][cid]
                require_hash(record['label'], previous['gt']['sha256'])
                require_hash(cache_path, previous['cache_sha256'])
                cache = read_json(cache_path)
                if cache['source_hashes'] != source_hashes:
                    raise ValueError(f'E6 case-cache source identity drift: {cid}')
            else:
                if was_complete:
                    raise ValueError(f'Completed E6 manifest lacks case cache: {cid}')
                partitions, probabilities = [], []
                ct_geometry = read_header(record['image'])
                for member in members:
                    folder = root / 'models' / member
                    pred_image = sitk.ReadImage(str(folder / 'predictions' / f'{cid}.mha'))
                    prob_image = sitk.ReadImage(str(folder / 'probabilities' / f'{cid}.mha'))
                    require_geometry(geometry(pred_image), ct_geometry)
                    require_geometry(geometry(prob_image), ct_geometry)
                    partitions.append(sitk.GetArrayFromImage(pred_image))
                    probabilities.append(sitk.GetArrayFromImage(prob_image))
                # Persist scores and pair components BEFORE reading GT voxels.
                score, pair = score_case(partitions, probabilities)
                pair = [{'member_i': first, 'member_j': second, **row}
                        for (first, second), row in zip(combinations(members, 2), pair)]
                score_path = root / 'cache' / f'{cid}.scores.json'
                save_json(score_path, {'case_id': cid, 'scores': score, 'pair_components': pair,
                                      'source_hashes': source_hashes, 'gt_voxels_loaded': False})
                result['records'][cid] = {'score_cache_sha256': sha256(score_path)}
                save_json(path, result)
                gt_fp = fingerprint(record['label'])
                gt_image = sitk.ReadImage(str(record['label']))
                require_geometry(geometry(gt_image), ct_geometry)
                gt = validate_labels(sitk.GetArrayFromImage(gt_image), record['label_codes'])
                evaluated = evaluate_predictions(cid, partitions, gt, tuple(gt_image.GetSpacing()[::-1]),
                    iou_threshold=config['evaluation']['iou_threshold'])
                require_hash(record['label'], gt_fp['sha256'])
                cache = {'case_id': cid, 'scores': score, 'pair_components': pair,
                         'source_hashes': source_hashes, **evaluated}
                save_json(cache_path, cache)
                result['records'][cid].update(gt=gt_fp, seconds=time.monotonic()-started,
                                              cache_sha256=sha256(cache_path))
                save_json(path, result)
            score_path = root / 'cache' / f'{cid}.scores.json'
            require_hash(score_path, result['records'][cid]['score_cache_sha256'])
            scores.append({'case_id': cid, **cache['scores']})
            pairs.extend({'case_id': cid, **row} for row in cache['pair_components'])
            model_risks.extend(cache['model_risks'])
            risks.extend(cache['risks'])
        if was_complete:
            verify_outputs(root, result['output_sha256'])
            print('E6 EVALUATION ALREADY COMPLETE', flush=True)
            return
        tables = report_tables(scores, risks, bootstrap_resamples=config['evaluation']['bootstrap_resamples'],
                               bootstrap_seed=config['evaluation']['bootstrap_seed'])
        data = {'scores': scores, 'pair_components': pairs, 'model_risks': model_risks,
                'risks': risks, 'comparison': tables['comparison'],
                'paired_differences': tables['paired_differences'], 'risk_coverage': tables['risk_coverage']}
        for name, rows in data.items():
            write_csv(root / f'{name}.csv', rows)
        summary = {**tables['summary'], 'smoke_only': False, 'n_external_cases': len(cases),
                   'scope': config['scope'], 'target': config['target'],
                   'checkpoint_selection': 'fixed final epoch; official validation excluded',
                   'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        save_json(root / 'summary.json', summary)
        outputs = {f'{name}.csv': sha256(root / f'{name}.csv') for name in data}
        outputs['summary.json'] = sha256(root / 'summary.json')
        result.update(status='complete', completed_at_unix=time.time(), output_sha256=outputs, summary=summary)
        save_json(path, result)
    print(f'E6 FULL COMPLETE: {len(cases)} official external cases; interpret effect and CI including negative results', flush=True)


def cache_metadata(source: Path, destination: Path, batch_size: int) -> dict:
    """Copy all metadata nnU-Net on_train_start needs, including fingerprint."""
    destination.mkdir(parents=True, exist_ok=True)
    plans = read_json(source / 'nnUNetPlans.json')
    plans['configurations']['3d_fullres']['batch_size'] = batch_size
    save_json(destination / 'nnUNetPlans.json', plans)
    for name in ('dataset.json', 'splits_final.json', 'dataset_fingerprint.json'):
        shutil.copyfile(source / name, destination / name)
    return {str(destination / name): sha256(destination / name) for name in
            ('nnUNetPlans.json', 'dataset.json', 'splits_final.json', 'dataset_fingerprint.json')}


def cache_inputs(config: dict, root: Path) -> None:
    """User-run full cache conversion. Parent .npz/.pkl are never modified."""
    settings = config['speed_revision']
    if not config['speed_selection_sha256']:
        raise ValueError('Run successful GPU batch benchmark before creating the new cache')
    parent, original, _ = parent_metadata(settings)
    source = parent / 'backend/nnUNet_preprocessed' / config['dataset_name']
    require_hash(source / 'dataset_fingerprint.json',
                 original['output_sha256'][str((source / 'dataset_fingerprint.json').relative_to(parent))])
    destination = root / 'backend/nnUNet_preprocessed' / config['dataset_name']
    plan = read_json(source / 'nnUNetPlans.json')
    identifier = plan['configurations']['3d_fullres']['data_identifier']
    input_data, output_data = source / identifier, destination / identifier
    expected_ids = sorted(original['internal_split']['train'] + original['internal_split']['val'])
    preparation = read_json(parent / 'preparation_manifest.json')
    if len(expected_ids) != 420 or len(set(expected_ids)) != 420 or set(expected_ids) != set(preparation['records']):
        raise ValueError('Cache must contain exactly the 420 training-source cases')
    estimate = 0
    for cid in expected_ids:
        with zipfile.ZipFile(input_data / f'{cid}.npz') as archive:
            estimate += sum(entry.file_size for entry in archive.infolist())
    require_disk({'sufficient': shutil.disk_usage(root).free > estimate * 1.05 + 30*2**30,
                  'estimated_required_bytes': int(estimate * 1.05 + 30*2**30),
                  'available_bytes': shutil.disk_usage(root).free})
    output_data.mkdir(parents=True, exist_ok=True)
    provenance = preprocessing_identity(config, root)
    path = root / 'preprocessing_manifest.json'
    with lock(root / '.cache.lock'):
        state = stage_state(path, provenance)
        if state['status'] == 'complete':
            verify_outputs(root, state['output_sha256'])
            print('FAST CACHE ALREADY COMPLETE', flush=True)
            return
        state.setdefault('output_sha256', {})
        state['internal_split'] = original['internal_split']
        save_json(path, state)
        for filename, digest in cache_metadata(source, destination, config['backend']['batch_size']).items():
            state['output_sha256'][str(Path(filename).relative_to(root))] = digest
        pending = []
        for cid in expected_ids:
            if cid in state['records']:
                wanted = {str((output_data / (cid + suffix)).relative_to(root))
                          for suffix in ('.npz', '.pkl', '.npy', '_seg.npy')}
                if set(state['records'][cid]) != wanted:
                    raise ValueError('Incomplete case cache schema; preserve artifacts for inspection')
                verify_outputs(root, state['records'][cid])
            else:
                pending.append(cid)
        def convert(cid):
            hashes = {suffix: original['output_sha256'][str((input_data / (cid+suffix)).relative_to(parent))]
                      for suffix in ('.npz', '.pkl')}
            return cid, unpack_one(input_data / cid, output_data / cid, hashes)
        with ThreadPoolExecutor(max_workers=settings['cache_workers']) as pool:
            for future in as_completed([pool.submit(convert, cid) for cid in pending]):
                cid, outputs = future.result()
                record = {str(Path(name).relative_to(root)): digest for name, digest in outputs.items()}
                state['records'][cid] = record
                state['output_sha256'].update(record)
                save_json(path, state)
                print(f'NPY cache {len(state["records"])}/420: {cid}', flush=True)
        state.update(status='complete', completed_at_unix=time.time(), train_cases=420,
                     role='bit_preserving_NPY_cache_no_replanning_or_external_case_reads')
        save_json(path, state)
    print('FAST CACHE COMPLETE', flush=True)


def switch_campaign(config: dict, root: Path) -> None:
    """User-run handoff only after benchmark/cache succeed; archive warmstarts."""
    verify_preprocessing(config, root)
    parent = Path(config['speed_revision']['parent_root'])
    receipt = root / 'handoff.json'
    with lock(root / '.handoff.lock'):
        if receipt.exists():
            saved = read_json(receipt)
            if saved['signature'] != signature(identity(config, root)):
                raise ValueError('Handoff identity drift')
            for record in saved['warmstarts'].values():
                if record.get('path'):
                    require_hash(Path(record['path']), record['sha256'])
            print('HANDOFF ALREADY COMPLETE', flush=True)
            return
        jobs = config['speed_revision']['old_jobs']
        expected_scripts = ('e6_train.sbatch', 'e6_infer.sbatch', 'e6_evaluate.sbatch')
        # Check that every already-started member has a readable, identity-bound
        # resume point BEFORE cancelling any job. Originals use atomic saves.
        import torch
        for member in config['members']:
            metadata = parent / 'models' / member / 'training_manifest.json'
            if not metadata.exists():
                continue
            model = model_folder(config, parent, member)
            source = model / ('checkpoint_final.pth' if (model/'checkpoint_final.pth').exists() else 'checkpoint_latest.pth')
            manifest = read_json(metadata)
            if manifest['signature'] != signature(manifest['provenance']):
                raise ValueError('Parent training manifest signature mismatch before handoff')
            checkpoint = torch.load(str(source), map_location='cpu', weights_only=False)
            if checkpoint.get('e6_identity') != manifest['signature'] or 'e6_rng' not in checkpoint:
                raise ValueError('Parent checkpoint not resumable; original jobs were not cancelled')
            del checkpoint
        active_jobs = []
        for job, script in zip(jobs, expected_scripts):
            query = subprocess.run(['scontrol', 'show', 'job', str(job), '-o'], text=True, capture_output=True)
            if query.returncode == 0:
                if f'({os.getuid()})' not in query.stdout or f'Command={ROOT}/scripts/{script}' not in query.stdout:
                    raise ValueError(f'Refuse cancellation: job {job} ownership/command differs')
                if not any(f'JobState={state}' in query.stdout for state in ('COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT')):
                    active_jobs.append(job)
        # Authorized change of batch requires replacing the original job chain.
        # Cancel all original array elements together, including queued member03.
        if active_jobs:
            subprocess.run(['scancel', *map(str, active_jobs)], check=True)
        records = {}
        for member in config['members']:
            folder = parent / 'models' / member
            if not folder.exists():
                records[member] = {'source': None, 'starting_epoch': 0}
                continue
            acquired = False
            for _ in range(120):
                try:
                    with lock(folder / '.train.lock'):
                        manifest_path = folder / 'training_manifest.json'
                        if not manifest_path.exists():
                            records[member] = {'source': None, 'starting_epoch': 0}
                            acquired = True
                            break
                        manifest = read_json(manifest_path)
                        if manifest['signature'] != signature(manifest['provenance']) or manifest['provenance']['seed'] != config['members'][member]:
                            raise ValueError('Parent training manifest identity mismatch')
                        model = model_folder(config, parent, member)
                        source = model / ('checkpoint_final.pth' if (model / 'checkpoint_final.pth').exists() else 'checkpoint_latest.pth')
                        if not source.exists():
                            raise ValueError(f'Parent {member} has no resumable latest checkpoint')
                        target = root / 'warmstarts' / member / 'checkpoint.pth'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        before = sha256(source)
                        temporary = target.with_suffix('.tmp')
                        shutil.copyfile(source, temporary)
                        require_hash(source, before); require_hash(temporary, before)
                        temporary.replace(target)
                        import torch
                        payload = torch.load(str(target), map_location='cpu', weights_only=False)
                        if payload.get('e6_identity') != manifest['signature'] or 'e6_rng' not in payload:
                            raise ValueError('Parent checkpoint identity/RNG mismatch')
                        record = {'path': str(target), 'source': str(source), 'sha256': before,
                                  'parent_training_signature': manifest['signature'],
                                  'parent_training_manifest_sha256': sha256(manifest_path),
                                  'starting_epoch': payload['current_epoch'],
                                  'phase_change': 'batch2_compressed_to_larger_batch_NPY_same_architecture'}
                        save_json(target.parent / 'source.json', record)
                        records[member] = record
                        acquired = True
                        break
                except RuntimeError as error:
                    if 'Another E6 writer owns' not in str(error):
                        raise
                    time.sleep(1)
            if not acquired:
                raise RuntimeError(f'Parent writer did not release checkpoint lock: {member}')
        save_json(receipt, {'signature': signature(identity(config, root)), 'old_jobs_cancelled': jobs,
                            'warmstarts': records, 'completed_at_unix': time.time()})
    print('HANDOFF COMPLETE; original checkpoints preserved', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Explicit E6 larger-batch/NPY-cache revision')
    parser.add_argument('stage', choices=['check', 'profile', 'benchmark', 'cache', 'switch', 'train', 'infer', 'evaluate'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ribfrac_fast.json')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--model', choices=['member_01', 'member_02', 'member_03'])
    args = parser.parse_args()
    settings = read_json(args.config)
    if args.output_root:
        os.environ['RIBFRAC_FAST_ROOT'] = str(args.output_root.resolve())
    config = load_config(args.config)
    root = output_path(config, Path(config['output_root']))
    if args.stage == 'check':
        parent, manifest, original = parent_metadata(settings)
        print(json.dumps({'status': 'parent_metadata_pass', 'parent': str(parent), 'output': str(root),
                          'original_batch': 2, 'requested_batches': [5,6], 'selected_batch': config['backend'].get('batch_size'),
                          'train_iterations': config['backend']['iterations_per_epoch'],
                          'warmstart_handoff_complete': (root/'handoff.json').exists()}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'profile':
        print(json.dumps(profile_case(settings, root/'io_profile.json'), indent=2))
    elif args.stage == 'benchmark':
        selection = root/'speed_selection.json'
        if selection.exists() and read_json(selection)['status'] == 'complete':
            print('BATCH BENCHMARK ALREADY COMPLETE (verified by config loader)')
        else:
            print(json.dumps(benchmark(settings, selection), indent=2))
    elif args.stage == 'cache': cache_inputs(config, root)
    elif args.stage == 'switch': switch_campaign(config, root)
    elif args.stage in ('train', 'infer'):
        if not args.model: parser.error('--model required')
        if not (root/'handoff.json').exists(): raise ValueError('Complete user-run handoff required')
        if read_json(root/'handoff.json')['signature'] != signature(identity(config, root)):
            raise ValueError('Handoff identity differs from requested training/inference')
        (train if args.stage == 'train' else infer)(config, root, args.model)
    else: evaluate(config, root)


if __name__ == '__main__':
    main()




