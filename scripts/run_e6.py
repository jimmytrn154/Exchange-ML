#!/usr/bin/env python3
"""E6 RibFrac prepare -> nnU-Net -> three independent models -> external scores.

Only check and small smoke are agent checks. Full stages are user-run Slurm
commands. Every stage writes a separate manifest and refuses identity drift.
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

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exchange_ml.e6_backend import (configure_paths, disk_estimate, fingerprint, internal_split,
    lock, make_predictor, make_trainer, read_json, reconstruct_instances, require_cuda,
    require_disk, require_hash, planning_threads, save_json, seed_everything, sha256, signature, software, write_volume)
from exchange_ml.e6_data import binary_target, inventory, require_geometry, geometry


def load_config(path: Path) -> dict:
    config = read_json(path)
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
    if root == ROOT or root == ROOT / 'outputs' or root in dataset.parents or dataset == root or dataset in root.parents:
        raise ValueError('E6 output root must not overlap the dataset or replace the repository/outputs root')
    for old in ('stage3_h1', 'stage3_h1_v2', 'stage2_backend', 'archive', 'local_copy_v1'):
        protected = ROOT / 'outputs' / old
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError('E6 output root overlaps protected existing artifacts')
    return root


def identity(config: dict, root: Path, *, smoke: bool = False) -> dict:
    files = [Path(__file__), *sorted((ROOT / 'src/exchange_ml').glob('e6_*.py')),
             ROOT / 'src/exchange_ml/e1_statistics.py', ROOT / 'src/exchange_ml/e1_scoring.py']
    return {'config': config, 'output_root': str(root), 'software': software(),
            'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in files},
            'smoke_only': smoke, 'role': 'synthetic_smoke_only' if smoke else 'external_ribfrac'}


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


def check(config: dict, root: Path) -> dict:
    cases = inventory(Path(config['dataset_root']))
    estimate = disk_estimate(list(cases.values()), root, config['storage'])
    result = {'status': 'ready_for_user_preparation' if estimate['sufficient'] else 'storage_insufficient',
              'train_cases': sum(r['split'] == 'train' for r in cases.values()),
              'external_cases': sum(r['split'] == 'validation' for r in cases.values()),
              'output_root': str(root), 'software': software(), 'storage': estimate,
              'full_results_exist': (root / 'e6_manifest.json').exists()}
    print(json.dumps(result, indent=2))
    return result


def prepare(config: dict, root: Path) -> None:
    cases = inventory(Path(config['dataset_root']))
    estimate = disk_estimate(list(cases.values()), root, config['storage'])
    if not (root / 'preparation_manifest.json').exists():
        require_disk(estimate)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / 'backend/nnUNet_raw' / config['dataset_name']
    provenance = identity(config, root)
    provenance['metadata_sha256'] = {str(p): sha256(p) for p in sorted({r['metadata'] for r in cases.values()})}
    manifest_path = root / 'preparation_manifest.json'
    with lock(root / '.prepare.lock'):
        manifest = stage_state(manifest_path, provenance)
        was_complete = manifest['status'] == 'complete'
        if was_complete and set(manifest['records']) != {cid for cid, r in cases.items() if r['split'] == 'train'}:
            raise ValueError('Completed preparation lacks 420 records')
        if not was_complete:
            save_json(manifest_path, manifest)
        for case_id, record in sorted(cases.items()):
            if record['split'] != 'train':
                continue  # No official validation labels/images enter nnU-Net raw data.
            image_path = raw / 'imagesTr' / f'{case_id}_0000.nii.gz'
            label_path = raw / 'labelsTr' / f'{case_id}.nii.gz'
            if case_id in manifest['records']:
                saved = manifest['records'][case_id]
                require_hash(record['image'], saved['source_image']['sha256'])
                require_hash(record['label'], saved['source_label']['sha256'])
                require_hash(label_path, saved['target']['sha256'])
                if not image_path.is_symlink() or image_path.resolve() != record['image'].resolve():
                    raise ValueError(f'Raw E6 CT link changed: {image_path}')
                continue
            started = time.monotonic()
            image_fp, label_fp = fingerprint(record['image']), fingerprint(record['label'])
            image, label = sitk.ReadImage(str(record['image'])), sitk.ReadImage(str(record['label']))
            require_geometry(geometry(image), geometry(label))
            target = binary_target(sitk.GetArrayFromImage(label), record['label_codes'])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.exists() or image_path.is_symlink():
                if not image_path.is_symlink() or image_path.resolve() != record['image'].resolve():
                    raise ValueError(f'Refuse to replace existing E6 image: {image_path}')
            else:
                image_path.symlink_to(record['image'].resolve())
            saved_target = write_volume(label_path, target, label)
            require_hash(record['image'], image_fp['sha256'])
            require_hash(record['label'], label_fp['sha256'])
            manifest['records'][case_id] = {'source_image': image_fp, 'source_label': label_fp,
                                           'target': saved_target, 'seconds': time.monotonic()-started}
            save_json(manifest_path, manifest)
        if was_complete:
            require_hash(raw / 'dataset.json', manifest['dataset_json_sha256'])
            print('E6 PREPARATION ALREADY COMPLETE', flush=True)
            return
        dataset = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'fracture': 1},
                   'numTraining': 420, 'file_ending': '.nii.gz',
                   'description': config['target'] + ' Official validation excluded.'}
        save_json(raw / 'dataset.json', dataset)
        manifest.update(status='complete', completed_at_unix=time.time(), storage_estimate=estimate,
                        dataset_json_sha256=sha256(raw / 'dataset.json'))
        save_json(manifest_path, manifest)
    print('E6 PREPARATION COMPLETE: 420 training cases; 80 external excluded', flush=True)


def preprocessing_identity(config: dict, root: Path) -> dict:
    prep = read_json(root / 'preparation_manifest.json')
    if prep['status'] != 'complete' or len(prep['records']) != 420:
        raise ValueError('E6 requires complete 420-case preparation')
    expected_prep = {**identity(config, root), 'metadata_sha256': {str(p): sha256(p) for p in sorted({r['metadata'] for r in inventory(Path(config['dataset_root'])).values()})}}
    if prep['provenance'] != expected_prep or prep['signature'] != signature(expected_prep):
        raise ValueError('Preparation config/code/software identity differs')
    raw = root / 'backend/nnUNet_raw' / config['dataset_name']
    require_hash(raw / 'dataset.json', prep['dataset_json_sha256'])
    for cid, record in prep['records'].items():
        require_hash(raw / 'imagesTr' / f'{cid}_0000.nii.gz', record['source_image']['sha256'])
        require_hash(raw / 'labelsTr' / f'{cid}.nii.gz', record['target']['sha256'])
    return {**identity(config, root), 'preparation_manifest_sha256': sha256(root / 'preparation_manifest.json')}


def run_preprocessing(config: dict, root: Path, ids: list[str], processes: int) -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    ds = config['dataset_id']
    with planning_threads(processes):
        extract_fingerprints([ds], num_processes=processes, check_dataset_integrity=True, clean=True)
        plan_experiments([ds], gpu_memory_target_in_gb=config['backend']['gpu_memory_target_gb'],
                         overwrite_plans_name=config['backend']['plans_name'])
        preprocess([ds], plans_identifier=config['backend']['plans_name'],
                   configurations=[config['backend']['configuration']], num_processes=[processes])
    folder = root / 'backend/nnUNet_preprocessed' / config['dataset_name']
    split = internal_split(ids, config['internal_split_seed'], config['internal_validation_fraction'])
    save_json(folder / 'splits_final.json', [split])


def preprocess(config: dict, root: Path, processes: int) -> None:
    configure_paths(root)
    provenance = preprocessing_identity(config, root)
    manifest_path = root / 'preprocessing_manifest.json'
    with lock(root / '.preprocess.lock'):
        manifest = stage_state(manifest_path, provenance)
        if manifest['status'] == 'complete':
            verify_outputs(root, manifest['output_sha256'])
            print('E6 PREPROCESSING ALREADY COMPLETE', flush=True)
            return
        save_json(manifest_path, manifest)
        ids = sorted(read_json(root / 'preparation_manifest.json')['records'])
        run_preprocessing(config, root, ids, processes)
        folder = root / 'backend/nnUNet_preprocessed' / config['dataset_name']
        outputs = {str(p.relative_to(root)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        manifest.update(status='complete', completed_at_unix=time.time(), output_sha256=outputs,
                        train_cases=420, internal_split=read_json(folder / 'splits_final.json')[0])
        save_json(manifest_path, manifest)
    print('E6 PREPROCESSING COMPLETE', flush=True)


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


def smoke(config: dict, root: Path, *, device: str = 'cuda') -> None:
    """Actual tiny synthetic nnU-Net train/backward/checkpoint/predict on one GPU.

    This output cannot be consumed as full E6 inference or external evidence.
    """
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU-only implementation smoke, never production.
    configure_paths(root, 'smoke')
    if device == 'cuda':
        require_cuda()
    raw = root / 'backend/nnUNet_raw' / config['dataset_name']
    if (root / 'preparation_manifest.json').exists():
        raise ValueError('Synthetic smoke root must be separate from full E6')
    from copy import deepcopy
    cfg = deepcopy(config)
    cfg['backend'].update(epochs=1, iterations_per_epoch=2, validation_iterations_per_epoch=1,
                          gpu_memory_target_gb=2, mirroring=False)
    started = time.monotonic()
    with lock(root / '.smoke.lock'):
        provenance = {**identity(cfg, root, smoke=True), 'smoke_device': device}
        path = root / 'smoke_manifest.json'
        result = stage_state(path, provenance)
        if result['status'] == 'complete':
            verify_outputs(root, result['output_sha256'])
            print(f'E6 SYNTHETIC {device.upper()} SMOKE ALREADY COMPLETE', flush=True)
            return
        save_json(path, result)
        ids = [f'E6Smoke{i:03d}' for i in range(4)]
        rng = np.random.default_rng(8123)
        for i, cid in enumerate(ids):
            z, y, x = np.ogrid[:32, :32, :32]
            target = (((z-12-i)**2+(y-12)**2+(x-12)**2) < 16).astype(np.uint8)
            image = sitk.GetImageFromArray((rng.normal(0, 10, target.shape) + 250*target).astype(np.float32))
            write_volume(raw / 'imagesTr' / f'{cid}_0000.nii.gz', sitk.GetArrayFromImage(image), image)
            write_volume(raw / 'labelsTr' / f'{cid}.nii.gz', target, image)
        save_json(raw / 'dataset.json', {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'fracture': 1},
                                        'numTraining': 4, 'file_ending': '.nii.gz'})
        run_preprocessing(cfg, root, ids, 1)
        seed_everything(32001)
        trainer = make_trainer(root, cfg, 'smoke', device=device)
        trainer.e6_identity = signature(provenance)
        trainer.run_training()
        checkpoint = model_folder(cfg, root, 'smoke') / 'checkpoint_final.pth'
        predictor = make_predictor(root, cfg, 'smoke', checkpoint, device=device)
        image = sitk.ReadImage(str(raw / 'imagesTr' / f'{ids[0]}_0000.nii.gz'))
        _, probs = predictor.predict_single_npy_array(sitk.GetArrayFromImage(image)[None],
            {'spacing': tuple(image.GetSpacing()[::-1])}, save_or_return_probabilities=True)
        probability = np.asarray(probs[1], np.float32)
        if not ((probability > 0) & (probability < 1)).any():
            raise ValueError('Smoke did not produce true soft probabilities')
        labels = reconstruct_instances(probability)
        out = write_volume(root / 'smoke_probability.mha', probability, image)
        pred = write_volume(root / 'smoke_prediction.mha', labels, image)
        from exchange_ml.e6_evaluation import score_case, evaluate_predictions
        score, pair = score_case([labels]*3, [probability]*3)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(raw / 'labelsTr' / f'{ids[0]}.nii.gz')))
        evaluated = evaluate_predictions('synthetic_smoke', [labels]*3, gt, (1., 1., 1.))
        import torch
        payload = torch.load(str(checkpoint), map_location='cpu')
        if 'e6_rng' not in payload or payload['current_epoch'] != 1:
            raise ValueError('Smoke checkpoint missing final epoch or RNG state')
        result.update(status='complete', completed_at_unix=time.time(),
                      summary={'role': 'synthetic_pipeline_smoke_only_not_external_evidence',
                               'seconds': time.monotonic()-started, 'scores': score,
                               'model_risks': evaluated['model_risks'],
                               'probability_shape': list(probability.shape),
                               'min_probability': float(probability.min()), 'max_probability': float(probability.max()),
                               'smoke_device': device,
                               'cuda_peak_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                               'one_network_repeated_for_evaluation_api_test': True},
                      output_sha256={'smoke_probability.mha': out['sha256'], 'smoke_prediction.mha': pred['sha256'],
                                     str(checkpoint.relative_to(root)): sha256(checkpoint)})
        save_json(path, result)
    print(f'E6 SYNTHETIC {device.upper()} SMOKE COMPLETE; no external result or full-runtime estimate', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'preprocess', 'train', 'infer', 'evaluate', 'smoke'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/e6.json')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--model', choices=['member_01', 'member_02', 'member_03'])
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='smoke only; production always Slurm CUDA')
    args = parser.parse_args()
    if args.processes < 1:
        parser.error('--processes must be positive')
    if args.device != 'cuda' and args.stage != 'smoke':
        parser.error('--device cpu is allowed only for synthetic smoke')
    config = load_config(args.config)
    if args.stage == 'smoke':
        smoke_root = args.output_root or Path(os.environ.get('E6_SMOKE_ROOT', str(ROOT / 'outputs/smoke_e6')))
        root = output_path(config, smoke_root)
    else:
        root = output_path(config, args.output_root)
    if args.stage in ('train', 'infer') and not args.model:
        parser.error('--model is required for train/infer')
    if args.stage == 'check': check(config, root)
    elif args.stage == 'prepare': prepare(config, root)
    elif args.stage == 'preprocess': preprocess(config, root, args.processes)
    elif args.stage == 'train': train(config, root, args.model)
    elif args.stage == 'infer': infer(config, root, args.model)
    elif args.stage == 'evaluate': evaluate(config, root)
    else: smoke(config, root, device=args.device)


if __name__ == '__main__':
    main()
