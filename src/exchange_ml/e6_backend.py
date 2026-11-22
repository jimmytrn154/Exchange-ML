"""RibFrac binary nnU-Net backend and explicit connected-component instances.

This is a new, fixed E6 backend, not a change to PENGWIN. A connected component
is a fracture-lesion prediction, not a whole rib or bone fragment. Touching
lesions may merge and disconnected lesion voxels may split; E6 measures these
limitations rather than using GT to repair predictions.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import platform
from typing import Iterator

import numpy as np
from scipy import ndimage
import SimpleITK as sitk


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


@contextmanager
def lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another E6 writer owns {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')


def fingerprint(path: Path) -> dict:
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed while hashing: {path}')
    return {'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'sha256': digest}


def software() -> dict:
    return {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in
            ('nnunetv2', 'torch', 'numpy', 'scipy', 'SimpleITK', 'dynamic_network_architectures',
             'batchgenerators', 'batchgeneratorsv2', 'acvl_utils')}}


def internal_split(ids: list[str], seed: int, validation_fraction: float) -> dict:
    """Fixed internal holdout; official validation IDs may never enter this API."""
    if not 0 < validation_fraction < 1 or len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('Require unique IDs and a nonempty train/internal-validation split')
    if any(x.startswith('RibFrac') and not 1 <= int(x[7:]) <= 420 for x in ids):
        raise ValueError('External RibFrac case entered training split')
    ordered = np.asarray(sorted(ids))
    shuffled = np.random.default_rng(seed).permutation(ordered)
    n_val = max(1, min(len(ids)-1, int(round(len(ids) * validation_fraction))))
    return {'train': sorted(shuffled[n_val:].tolist()), 'val': sorted(shuffled[:n_val].tolist())}


def reconstruct_instances(probability: np.ndarray, *, threshold: float = .5,
                          connectivity: int = 26, minimum_component_mm3: float = 0,
                          spacing_zyx: tuple[float, ...] = (1., 1., 1.)) -> np.ndarray:
    """Soft foreground [Z,Y,X] -> nonnegative int32 components; no GT input."""
    probability = np.asarray(probability)
    if probability.ndim != 3 or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Expected finite soft foreground [Z,Y,X] in [0,1]')
    if not 0 < threshold < 1 or connectivity != 26 or minimum_component_mm3 < 0:
        raise ValueError('Invalid reconstruction parameters')
    if len(spacing_zyx) != 3 or not np.isfinite(spacing_zyx).all() or min(spacing_zyx) <= 0:
        raise ValueError('Expected positive voxel spacing')
    labels, _ = ndimage.label(probability >= threshold, structure=np.ones((3, 3, 3), bool))
    if minimum_component_mm3 > 0:
        counts = np.bincount(labels.ravel())
        keep = counts * float(np.prod(spacing_zyx)) >= minimum_component_mm3
        keep[0] = False
        labels, _ = ndimage.label(keep[labels], structure=np.ones((3, 3, 3), bool))
    return labels.astype(np.int32, copy=False)


def write_volume(path: Path, array: np.ndarray, reference: sitk.Image) -> dict:
    if tuple(array.shape[::-1]) != tuple(reference.GetSize()):
        raise ValueError('Output volume shape differs from reference')
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    temporary = path.with_name('.tmp-' + path.name)
    sitk.WriteImage(image, str(temporary), useCompression=True)
    reread = sitk.ReadImage(str(temporary))
    if not np.array_equal(sitk.GetArrayViewFromImage(reread), array):
        raise ValueError(f'Volume roundtrip changed values: {path}')
    os.replace(temporary, path)
    return fingerprint(path)


def header_voxels(path: Path) -> int:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return int(np.prod(reader.GetSize(), dtype=np.int64))


def disk_estimate(records: list[dict], output_root: Path, storage: dict) -> dict:
    """Conservative planning estimate, not a promise of compressed disk usage."""
    train = sum(header_voxels(r['image']) for r in records if r['split'] == 'train')
    external = sum(header_voxels(r['image']) for r in records if r['split'] != 'train')
    estimated = (train * storage['estimated_bytes_per_training_voxel']
                 + external * storage['estimated_bytes_per_external_voxel']
                 + int(storage['reserve_gib'] * 2**30))
    parent = output_root.resolve()
    while not parent.exists():
        parent = parent.parent
    available = shutil.disk_usage(parent).free
    return {'estimated_required_bytes': estimated, 'available_bytes': available,
            'training_voxels': train, 'external_voxels': external,
            'sufficient': available >= estimated,
            'note': 'Header-based conservative estimate; actual resampled/compressed bytes may differ.'}


def require_disk(estimate: dict) -> None:
    if not estimate['sufficient']:
        raise RuntimeError('Insufficient E6 storage: estimated '
                           f"{estimate['estimated_required_bytes']/2**30:.1f} GiB, "
                           f"available {estimate['available_bytes']/2**30:.1f} GiB. "
                           'Choose E6_OUTPUT_ROOT on a larger filesystem; no input was modified.')


def configure_paths(root: Path, member: str | None = None) -> None:
    """Must run before importing nnunetv2.paths in a fresh worker process."""
    os.environ['nnUNet_raw'] = str(root / 'backend/nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(root / 'backend/nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(root / 'backend/results' / (member or 'unused'))
    os.environ['nnUNet_n_proc_DA'] = '0'  # deterministic RNG in one augmentation process
    os.environ['nnUNet_compile'] = 'false'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/exchange-e6-mpl')


@contextmanager
def planning_threads(processes: int) -> Iterator[None]:
    """nnU-Net planner uses DA count for torch threads, which must be positive.

    Training deliberately uses zero DA workers for SingleThreadedAugmenter;
    planning temporarily needs a positive value and restores the training one.
    """
    previous = os.environ.get('nnUNet_n_proc_DA')
    os.environ['nnUNet_n_proc_DA'] = str(max(1, processes))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('nnUNet_n_proc_DA', None)
        else:
            os.environ['nnUNet_n_proc_DA'] = previous


def require_cuda() -> None:
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('E6 training/inference requires Slurm with exactly one visible CUDA device')


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def trainer_class():
    """Local trainer: nnU-Net behavior with atomic, RNG-preserving checkpoints.

    Kept local to the repo; no writes to the installed nnU-Net package. Single
    process augmentation allows Python/NumPy/Torch RNG restoration on resume.
    Hardware/library differences can still prevent bitwise reproducibility.
    """
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class E6Trainer(nnUNetTrainer):
        def save_checkpoint(self, filename: str) -> None:
            if self.disable_checkpointing or self.local_rank != 0:
                return
            network = self.network
            if hasattr(network, '_orig_mod'):
                network = network._orig_mod
            payload = {'network_weights': network.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler else None,
                       'logging': self.logger.get_checkpoint(), '_best_ema': self._best_ema,
                       'current_epoch': self.current_epoch + 1, 'init_args': self.my_init_kwargs,
                       'trainer_name': 'E6Trainer', 'e6_identity': self.e6_identity,
                       'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                       'e6_rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                  'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
            temporary = str(filename) + '.tmp'
            torch.save(payload, temporary)
            os.replace(temporary, filename)

        def load_checkpoint(self, filename_or_checkpoint):
            if not isinstance(filename_or_checkpoint, str):
                raise TypeError('E6 checkpoint must be a recorded local file')
            payload = torch.load(filename_or_checkpoint, map_location='cpu')
            if 'e6_rng' not in payload:
                raise ValueError('E6 resume requires a checkpoint with recorded RNG state')
            if payload.get('e6_identity') != self.e6_identity:
                raise ValueError('E6 checkpoint seed/config/code identity differs from training manifest')
            super().load_checkpoint(filename_or_checkpoint)
            self._e6_resume_rng = payload['e6_rng']

        def get_dataloaders(self):
            loaders = super().get_dataloaders()
            state = getattr(self, '_e6_resume_rng', None)
            if state is not None:
                random.setstate(state['python'])
                np.random.set_state(state['numpy'])
                torch.set_rng_state(state['torch'].cpu())
                torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
                self._e6_resume_rng = None
            return loaders

    return E6Trainer


def make_trainer(root: Path, config: dict, member: str, *, device: str = 'cuda'):
    import torch
    folder = root / 'backend/nnUNet_preprocessed' / config['dataset_name']
    plans = read_json(folder / (config['backend']['plans_name'] + '.json'))
    dataset = read_json(folder / 'dataset.json')
    trainer = trainer_class()(plans, config['backend']['configuration'], 0, dataset,
                              unpack_dataset=False, device=torch.device(device))
    trainer.num_epochs = config['backend']['epochs']
    trainer.num_iterations_per_epoch = config['backend']['iterations_per_epoch']
    trainer.num_val_iterations_per_epoch = config['backend']['validation_iterations_per_epoch']
    trainer.save_every = 10
    return trainer


def make_predictor(root: Path, config: dict, member: str, checkpoint: Path, *, device: str = 'cuda'):
    """Build architecture from recorded plans without custom-class discovery."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    folder = root / 'backend/nnUNet_preprocessed' / config['dataset_name']
    plans = PlansManager(read_json(folder / (config['backend']['plans_name'] + '.json')))
    configuration = plans.get_configuration(config['backend']['configuration'])
    dataset = read_json(folder / 'dataset.json')
    state = torch.load(str(checkpoint), map_location='cpu')
    network = nnUNetTrainer.build_network_architecture(
        configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        determine_num_input_channels(plans, configuration, dataset),
        plans.get_label_manager(dataset).num_segmentation_heads, enable_deep_supervision=False)
    network.load_state_dict(state['network_weights'])
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True,
                                use_mirroring=config['backend']['mirroring'],
                                perform_everything_on_device=device == 'cuda', device=torch.device(device),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans, configuration, [state['network_weights']], dataset, 'E6Trainer',
                                    state['inference_allowed_mirroring_axes'])
    predictor.network = predictor.network.to(torch.device(device)).eval()
    return predictor
