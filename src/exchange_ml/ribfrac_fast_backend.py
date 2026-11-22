"""Explicit performance fork of E6; original E6 files and artifacts stay frozen."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import tempfile
import time

from .e6_backend import *  # Frozen public backend helpers; inference semantics unchanged.
from . import e6_backend as baseline


def configure_paths(root: Path, member: str | None = None) -> None:
    baseline.configure_paths(root, member)
    # Single augmentation process preserves the existing checkpoint RNG contract.
    # Do not retain hundreds of mmap views in the training process. The kernel
    # still caches file pages, while per-sample mappings can be released.
    os.environ['nnUNet_keep_files_open'] = 'false'


def make_trainer(root: Path, config: dict, member: str, *, device: str = 'cuda'):
    import torch
    folder = root / 'backend/nnUNet_preprocessed' / config['dataset_name']
    trainer = trainer_class()(read_json(folder / 'nnUNetPlans.json'), '3d_fullres', 0,
                             read_json(folder / 'dataset.json'), unpack_dataset=False,
                             device=torch.device(device))
    trainer.num_epochs = config['backend']['epochs']
    trainer.num_iterations_per_epoch = config['backend']['iterations_per_epoch']
    trainer.num_val_iterations_per_epoch = config['backend']['validation_iterations_per_epoch']
    trainer.save_every = 5
    return trainer


def load_parent_checkpoint(trainer, path: Path, parent_signature: str) -> None:
    """Authorized fork: restore weights/optimizer/scaler/RNG, never edit source."""
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    if payload.get('e6_identity') != parent_signature or 'e6_rng' not in payload:
        raise ValueError('Parent checkpoint identity/RNG mismatch')
    current_args = trainer.my_init_kwargs
    # nnU-Net 2.5.1's loader only initializes its local checkpoint variable
    # for filename inputs. Materialize the authorized bridge in the NEW run.
    bridge = Path(trainer.output_folder) / 'warmstart_bridge.pth'
    bridge.parent.mkdir(parents=True, exist_ok=True)
    payload['e6_identity'] = trainer.e6_identity
    payload['init_args'] = current_args
    torch.save(payload, str(bridge))
    nnUNetTrainer.load_checkpoint(trainer, str(bridge))
    trainer.my_init_kwargs = current_args  # Future checkpoint records the new batch plan.
    trainer._e6_resume_rng = payload['e6_rng']


def batch_budget(batch: int, train_patches: int = 500, val_patches: int = 100) -> dict:
    if batch not in (5, 6):
        raise ValueError('Requested batch must be 5 or 6 (baseline is 2)')
    train_steps = max(1, int(np.floor(train_patches / batch + .5)))
    val_steps = max(1, int(np.floor(val_patches / batch + .5)))
    return {'batch_size': batch, 'iterations_per_epoch': train_steps,
            'validation_iterations_per_epoch': val_steps,
            'train_patches_per_epoch': batch * train_steps,
            'validation_patches_per_epoch': batch * val_steps}


def parent_metadata(settings: dict) -> tuple[Path, dict, dict]:
    parent = Path(settings['parent_root']).resolve()
    prep = read_json(parent / 'preprocessing_manifest.json')
    if prep['status'] != 'complete' or prep['signature'] != signature(prep['provenance']):
        raise ValueError('Complete hash-bound parent preprocessing required')
    source_config = prep['provenance']['config']
    for name, digest in prep['provenance']['code_sha256'].items():
        # Parent code paths are relative to the research repository.
        require_hash(Path(__file__).resolve().parents[2] / name, digest)
    folder = parent / 'backend/nnUNet_preprocessed' / source_config['dataset_name']
    for name in ('dataset.json', 'nnUNetPlans.json', 'splits_final.json'):
        require_hash(folder / name, prep['output_sha256'][str((folder / name).relative_to(parent))])
    return parent, prep, source_config


def unpack_one(source: Path, destination: Path, expected: dict) -> dict:
    """Bit-preserving .npy cache sidecars; no spatial/intensity transformation."""
    result = {}
    for suffix in ('.npz', '.pkl'):
        path = source.with_suffix(suffix)
        require_hash(path, expected[suffix])
        target = destination.with_suffix(suffix)
        if not target.exists():
            target.symlink_to(path)
        elif target.resolve() != path.resolve():
            raise ValueError(f'Cache source alias drift: {target}')
        result[str(target)] = expected[suffix]
    with np.load(source.with_suffix('.npz'), allow_pickle=False) as archive:
        for key, suffix in (('data', '.npy'), ('seg', '_seg.npy')):
            value = archive[key]
            target = destination.parent / (destination.name + suffix)
            temporary = target.with_name(target.name + '.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, value, allow_pickle=False)
            temporary.replace(target)
            observed = np.load(target, mmap_mode='r', allow_pickle=False)
            if observed.shape != value.shape or observed.dtype != value.dtype:
                raise ValueError('Cache shape/dtype drift')
            for start in range(value.shape[1]):
                if not np.array_equal(value[:, start], observed[:, start], equal_nan=True):
                    raise ValueError('Cache content drift')
            result[str(target)] = sha256(target)
            del observed, value
    return result


def profile_case(settings: dict, output: Path) -> dict:
    """One training case only: compare compressed and mmap patch access."""
    parent, prep, config = parent_metadata(settings)
    folder = parent / 'backend/nnUNet_preprocessed' / config['dataset_name']
    plans = read_json(folder / 'nnUNetPlans.json')['configurations']['3d_fullres']
    data = folder / plans['data_identifier']
    case = sorted(prep['internal_split']['train'])[0]
    source = data / f'{case}.npz'
    timings = []
    with tempfile.TemporaryDirectory(prefix='ribfrac-io-', dir='/tmp') as temporary:
        target = Path(temporary)
        for repeat in range(3):
            started = time.perf_counter()
            with np.load(source, allow_pickle=False) as archive:
                image, segmentation = archive['data'], archive['seg']
            timings.append(time.perf_counter() - started)
        np.save(target / 'data.npy', image); np.save(target / 'seg.npy', segmentation)
        patch = tuple(plans['patch_size'])
        started = time.perf_counter()
        for _ in range(10):
            a = np.load(target / 'data.npy', mmap_mode='r')
            b = np.load(target / 'seg.npy', mmap_mode='r')
            slices = (slice(None), *(slice(0, n) for n in patch))
            assert np.array_equal(np.array(a[slices]), image[slices])
            assert np.array_equal(np.array(b[slices]), segmentation[slices])
        patch_seconds = (time.perf_counter() - started) / 10
        result = {'role': 'one_training_case_IO_smoke_not_epoch_throughput', 'case_id': case,
                  'compressed_seconds': timings, 'mmap_patch_seconds': patch_seconds,
                  'shape': list(image.shape), 'dtype': str(image.dtype),
                  'source_sha256': sha256(source), 'same_patch_values': True}
    save_json(output, result)
    return result


def benchmark(settings: dict, output: Path) -> dict:
    """Short synthetic full-patch GPU probe; no real training checkpoint writes."""
    require_cuda()
    import torch
    seed_everything(32001)
    torch.backends.cudnn.benchmark = settings['cudnn_benchmark']
    parent, _, config = parent_metadata(settings)
    source = parent / 'backend/nnUNet_preprocessed' / config['dataset_name']
    original_plan = read_json(source / 'nnUNetPlans.json')
    dataset = read_json(source / 'dataset.json')
    patch = original_plan['configurations']['3d_fullres']['patch_size']
    results = []
    with tempfile.TemporaryDirectory(prefix='ribfrac-gpu-', dir='/tmp') as temporary:
        configure_paths(Path(temporary), 'benchmark')
        for batch in settings['batch_candidates']:
            trainer = data = targets = None
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            external = total - free - torch.cuda.memory_reserved()
            try:
                plans = copy.deepcopy(original_plan)
                plans['configurations']['3d_fullres']['batch_size'] = batch
                trainer = trainer_class()(plans, '3d_fullres', 0, dataset,
                                          unpack_dataset=False, device=torch.device('cuda'))
                trainer.initialize()
                # Normal on_train_start enables the decoder's deep supervision;
                # the bounded probe skips dataloaders, so do that explicitly.
                trainer.set_deep_supervision_enabled(True)
                trainer.network.train()
                data = torch.randn((batch, 1, *patch), dtype=torch.float32)
                targets = [torch.randint(0, 2, (batch, 1, *(int(round(p*s)) for p, s in zip(patch, scales))),
                                         dtype=torch.int64) for scales in trainer._get_deep_supervision_scales()]
                payload = {'data': data, 'target': targets}
                for _ in range(settings['benchmark_warmup_steps']): trainer.train_step(payload)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                losses = [float(trainer.train_step(payload)['loss']) for _ in range(settings['benchmark_steps'])]
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - started) / settings['benchmark_steps']
                peak = torch.cuda.max_memory_reserved()
                headroom = total - external - peak
                results.append({'batch': batch, 'status': 'ok', 'seconds_per_step': seconds,
                                'patches_per_second': batch / seconds, 'peak_reserved_bytes': peak,
                                'headroom_bytes': headroom, 'finite_loss': bool(np.isfinite(losses).all()),
                                'eligible': bool(np.isfinite(losses).all() and headroom >= settings['memory_headroom_gib'] * 2**30)})
            except torch.cuda.OutOfMemoryError:
                results.append({'batch': batch, 'status': 'OOM', 'eligible': False})
            finally:
                trainer = data = targets = payload = None
                gc.collect(); torch.cuda.empty_cache()
            save_json(output, {'status': 'running', 'results': results})
    candidates = [r for r in results if r['batch'] in (5, 6) and r['eligible']]
    if not candidates:
        save_json(output, {'status': 'no_safe_requested_batch', 'results': results})
        raise RuntimeError('Neither requested larger batch retained measured memory headroom')
    chosen = max(candidates, key=lambda r: r['patches_per_second'])
    result = {'status': 'complete', 'results': results, 'selected_batch': chosen['batch'],
              'gpu': torch.cuda.get_device_name(0), 'role': 'synthetic_compute_probe_not_epoch_ETA',
              'budget': batch_budget(chosen['batch']), 'settings': settings,
              'backend_sha256': sha256(Path(__file__)),
              'parent_plans_sha256': sha256(source / 'nnUNetPlans.json')}
    save_json(output, result)
    return result
