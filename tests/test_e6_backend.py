from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import os
from unittest.mock import patch
import numpy as np
import SimpleITK as sitk
from exchange_ml.e6_backend import (internal_split, reconstruct_instances, write_volume,
                                    fingerprint, require_hash, lock, disk_estimate, require_disk, planning_threads)


class E6BackendTests(unittest.TestCase):
    def test_split_is_fixed_and_excludes_external(self):
        ids = [f'RibFrac{i}' for i in range(1,421)]
        split = internal_split(ids, 20261121, .2)
        self.assertEqual((len(split['train']),len(split['val'])),(336,84))
        self.assertFalse(set(split['train']) & set(split['val']))
        self.assertEqual(set(ids),set(split['train']+split['val']))
        self.assertEqual(split,internal_split(ids[::-1],20261121,.2))
        with self.assertRaisesRegex(ValueError,'External'):
            internal_split(['RibFrac1','RibFrac421'],1,.2)

    def test_component_policy_preserves_small_and_empty_and_soft_range(self):
        probability=np.zeros((5,5,5),np.float32)
        probability[0,0,0]=.5
        probability[1,1,1]=.8
        probability[4,4,4]=.9
        labels=reconstruct_instances(probability)
        self.assertEqual(int(labels.max()),2)
        self.assertEqual(labels[0,0,0],labels[1,1,1])
        self.assertEqual(labels.dtype,np.int32)
        self.assertFalse(reconstruct_instances(probability*0).any())
        self.assertFalse(reconstruct_instances(probability,minimum_component_mm3=10).any())
        with self.assertRaises(ValueError): reconstruct_instances(probability+1)
        with self.assertRaises(ValueError): reconstruct_instances(probability,spacing_zyx=(0,1,1))

    def test_volume_formats_preserve_values_geometry_and_hash_integrity(self):
        with TemporaryDirectory() as directory:
            reference=sitk.Image((5,4,3),sitk.sitkFloat32)
            reference.SetSpacing((.7,.8,1.1));reference.SetDirection((-1.,0,0,0,-1.,0,0,0,1.))
            array=np.random.default_rng(1).random((3,4,5),dtype=np.float32)
            for name in ('probability.mha','probability.nii.gz'):
                path=Path(directory)/name
                fp=write_volume(path,array,reference)
                require_hash(path,fp['sha256'])
                output=sitk.ReadImage(str(path))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(output),array)
                np.testing.assert_allclose(output.GetSpacing(),reference.GetSpacing(),rtol=1e-6)
                data=path.read_bytes();path.write_bytes(data[:-1]+bytes([data[-1]^1]))
                with self.assertRaisesRegex(ValueError,'SHA-256'): require_hash(path,fp['sha256'])

    def test_locks_exclude_duplicate_writers(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'.lock'
            with lock(path):
                with self.assertRaisesRegex(RuntimeError,'writer'):
                    with lock(path): pass

    def test_planner_thread_count_positive_and_training_da_restored(self):
        with patch.dict(os.environ, {'nnUNet_n_proc_DA': '0'}):
            with planning_threads(4):
                self.assertEqual(os.environ['nnUNet_n_proc_DA'], '4')
                import torch
                before = torch.get_num_threads()
                torch.set_num_threads(int(os.environ['nnUNet_n_proc_DA']))
                torch.set_num_threads(before)
            self.assertEqual(os.environ['nnUNet_n_proc_DA'], '0')
            with self.assertRaisesRegex(RuntimeError, 'planner failure'):
                with planning_threads(1):
                    raise RuntimeError('planner failure')
            self.assertEqual(os.environ['nnUNet_n_proc_DA'], '0')

    def test_insufficient_disk_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError,'Insufficient E6 storage'):
            require_disk({'sufficient':False,'estimated_required_bytes':2**40,'available_bytes':2**20})

if __name__=='__main__':unittest.main()
