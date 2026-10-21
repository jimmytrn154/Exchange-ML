"""Stage 3 backend fixes F1 (Anatomy L/R mirror), F2 (DDP-global Fragment validation),
D3 (Anatomy partial-label case map), the per-bone Fragment check (D6), the
Fragment inference checkpoint (F8), and the paint-time block mask (F9)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "baselines" / "pengwin2026-task1-abbc" / "code_task1"
sys.path.insert(0, str(CODE_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import core  # type: ignore[import-not-found]  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import stage2_backend as backend_cli  # type: ignore[import-not-found]  # noqa: E402
from exchange_ml.stage2_backend import case_labeled_class_map, marginal_case_key  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.utilities.collate_outputs import collate_outputs  # noqa: E402

ANATOMY = "Dataset701_PENGWINStage2Anatomy"
FRAGMENT = "Dataset702_PENGWINStage2Fragment"


def mirror_config(trainer_cls: type, dataset_name: str) -> tuple[tuple[int, ...], object]:
    trainer = object.__new__(trainer_cls)
    trainer.plans_manager = SimpleNamespace(dataset_name=dataset_name)
    trainer.print_to_log_file = lambda *args, **kwargs: None
    trainer.inference_allowed_mirroring_axes = (0, 1, 2)
    stock = (None, False, (128, 288, 192), (0, 1, 2))
    with mock.patch.object(
        nnUNetTrainer, "configure_rotation_dummyDA_mirroring_and_inital_patch_size", return_value=stock
    ):
        axes = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()[3]
    return axes, trainer.inference_allowed_mirroring_axes


def val_step(loss: float, iouf: float, recall: float, precision: float) -> dict:
    a = lambda v: np.asarray([v], dtype=np.float64)  # noqa: E731
    return {"loss": np.asarray(loss), "iouf": a(iouf), "recall": a(recall),
            "precision": a(precision), "npred": a(4.0), "ngt": a(5.0)}


class AnatomyMirrorFixTests(unittest.TestCase):
    def test_anatomy_dataset_drops_lr_axis_for_training_and_inference(self) -> None:
        axes, inference_axes = mirror_config(core.PengwinTrainerSTUNetBaseAnatomyV301, ANATOMY)
        self.assertEqual(axes, (0, 1))
        self.assertEqual(inference_axes, (0, 1))

    def test_fragment_dataset_keeps_all_mirror_axes(self) -> None:
        axes, inference_axes = mirror_config(core.PengwinTrainerSTUNetBaseAffinityV308, FRAGMENT)
        self.assertEqual(axes, (0, 1, 2))
        self.assertEqual(inference_axes, (0, 1, 2))
        self.assertNotIn(FRAGMENT, core.PengwinTrainer.DISABLE_X_MIRROR_DATASETS)

    def test_upstream_entry_is_kept(self) -> None:
        self.assertIn("Dataset539_PelvicFemurAnatomyV3", core.PengwinTrainer.DISABLE_X_MIRROR_DATASETS)


class FragmentDdpValidationFixTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two ranks whose local F1 would differ sharply.
        self.rank0 = collate_outputs([val_step(0.4, 0.8, 0.9, 0.9), val_step(0.5, 0.7, 0.8, 0.9)])
        self.rank1 = collate_outputs([val_step(0.9, 0.2, 0.1, 0.3), val_step(0.8, 0.3, 0.2, 0.2)])

    def test_merge_concatenates_every_step_in_rank_order(self) -> None:
        merged = core._merge_rank_val_outputs([self.rank0, self.rank1])
        np.testing.assert_allclose(merged["recall"], [0.9, 0.8, 0.1, 0.2])
        np.testing.assert_allclose(merged["loss"], [0.4, 0.5, 0.9, 0.8])
        self.assertEqual(set(merged), set(self.rank0))

    def test_every_rank_reaches_identical_f1_and_early_stop(self) -> None:
        decisions = []
        for _rank in range(2):
            trainer = object.__new__(core.PengwinTrainerSTUNetBaseAffinityV308)
            logged: dict[str, list] = {}
            trainer.logger = SimpleNamespace(
                log=lambda key, value, epoch, store=logged: store.setdefault(key, []).append(value),
                my_fantastic_logging={"ema_fg_dice": [0.5]},
            )
            trainer.print_to_log_file = lambda *args, **kwargs: None
            trainer.is_ddp = True
            trainer.current_epoch = 40
            trainer.num_epochs = 1000
            trainer.ES_MIN_EPOCHS, trainer.ES_PATIENCE, trainer.ES_MIN_DELTA = 30, 1, 5e-3
            trainer._es_best_dice, trainer._es_no_improve, trainer._es_triggered = 0.6, 0, False
            fake_dist = SimpleNamespace(
                get_world_size=lambda: 2,
                all_gather_object=lambda out, obj: out.__setitem__(slice(None), [self.rank0, self.rank1]),
            )
            local = [val_step(0.4, 0.8, 0.9, 0.9)] if _rank == 0 else [val_step(0.9, 0.2, 0.1, 0.3)]
            with mock.patch.object(core, "dist", fake_dist):
                trainer.on_validation_epoch_end(local)
            decisions.append((logged["mean_fg_dice"][0], trainer.num_epochs, trainer._es_triggered))
        self.assertEqual(decisions[0], decisions[1])
        recall, precision = np.mean([0.9, 0.8, 0.1, 0.2]), np.mean([0.9, 0.9, 0.3, 0.2])
        self.assertAlmostEqual(decisions[0][0], 2 * precision * recall / (precision + recall))
        self.assertEqual(decisions[0][1:], (41, True))


def anatomy_trainer_with_map(path: str | None):
    trainer = object.__new__(core.PengwinTrainerSTUNetBaseAnatomyV301)
    trainer.print_to_log_file = lambda *args, **kwargs: None
    env = {"PENGWIN_CASE_LABELED_MAP": path} if path else {}
    with mock.patch.dict(os.environ, env, clear=False):
        if not path:
            os.environ.pop("PENGWIN_CASE_LABELED_MAP", None)
        trainer._load_case_labeled_map()
    return trainer


class AnatomyCaseMapFixTests(unittest.TestCase):
    def test_map_records_classes_present_per_case(self) -> None:
        pelvic = np.zeros((2, 2, 2), dtype=np.uint8)
        pelvic[0, 0, 0], pelvic[0, 0, 1], pelvic[0, 1, 0] = 1, 2, 3
        femur = np.zeros((2, 2, 2), dtype=np.uint8)
        femur[1, 1, 1] = 4
        mapping = case_labeled_class_map({"PENGWIN_002": pelvic, "PENGWIN_254": femur})
        self.assertEqual(mapping, {"002": [1, 2, 3], "254": [4]})

    def test_invalid_targets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            case_labeled_class_map({"PENGWIN_002": np.zeros((2, 2), dtype=np.uint8)})
        with self.assertRaises(ValueError):
            case_labeled_class_map({"PENGWIN_002": np.full((2, 2), 7, dtype=np.uint8)})

    def test_key_matches_trainer_parsing_and_masks_unlabelled_bones(self) -> None:
        self.assertEqual(marginal_case_key("PENGWIN_002"), "002")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(json.dumps({"002": [1, 2, 3], "254": [4]}), encoding="utf-8")
            trainer = anatomy_trainer_with_map(str(path))
        trainer.device = "cpu"
        trainer.label_manager = SimpleNamespace(num_segmentation_heads=5)
        trainer._marginal_loss = SimpleNamespace(labeled_mask=None)
        trainer._set_marginal_mask(["PENGWIN_002", "PENGWIN_254"])
        mask = trainer._marginal_loss.labeled_mask.tolist()
        self.assertEqual(mask[0], [True, True, True, True, False])   # pelvic: Femur unlabelled
        self.assertEqual(mask[1], [True, False, False, False, True])  # femur: pelvis unlabelled

    def test_explicit_map_that_cannot_load_fails_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            anatomy_trainer_with_map(str(Path(tmp) / "missing.json"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                anatomy_trainer_with_map(str(path))

    def test_wrapper_writes_map_covering_the_frozen_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "raw" / ANATOMY / "labelsTr"
            labels.mkdir(parents=True)
            for identifier, value in (("PENGWIN_002", 2), ("PENGWIN_254", 4)):
                array = np.zeros((2, 2, 2), dtype=np.uint8)
                array[0, 0, 0] = value
                sitk.WriteImage(sitk.GetImageFromArray(array), str(labels / f"{identifier}.mha"))
            config = {
                "paths": {"workspace_root": str(root), "raw_root": str(root / "raw"),
                          "results_root": str(root / "results")},
                "anatomy_stage": {"dataset_name": ANATOMY},
            }
            manifest = {"split_samples": {"anatomy": {"train": ["PENGWIN_002"],
                                                       "validation": ["PENGWIN_254"]}}}
            (root / "stage2_preparation_manifest.json").write_text(json.dumps(manifest))
            with mock.patch("builtins.print"):
                path = backend_cli.write_case_labeled_map(config)
            self.assertEqual(path, root / "results" / ANATOMY / "case_labeled_classes.json")
            self.assertEqual(json.loads(path.read_text()), {"002": [2], "254": [4]})
            manifest["split_samples"]["anatomy"]["validation"].append("PENGWIN_999")
            (root / "stage2_preparation_manifest.json").write_text(json.dumps(manifest))
            with mock.patch("builtins.print"), self.assertRaises(RuntimeError):
                backend_cli.write_case_labeled_map(config)


INFERENCE_DIR = CODE_DIR.parent / "inference"
BACKEND_CONFIG = {"paths": {"baseline_root": str(CODE_DIR.parent)},
                  "anatomy_stage": {"trainer": "PengwinTrainerSTUNetBaseAnatomyV301"},
                  "fragment_stage": {"trainer": "PengwinTrainerSTUNetBaseAffinityV308",
                                     "inference_checkpoint": "checkpoint_final.pth"}}


def fresh_inference_module(disable_budget: bool):
    import importlib

    sys.modules.pop("inference", None)
    if str(INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(INFERENCE_DIR))
    with mock.patch.dict(os.environ, {"PENGWIN_DISABLE_TIME_BUDGET": "1" if disable_budget else "0"}):
        return importlib.import_module("inference")


class InferenceTimeBudgetFixTests(unittest.TestCase):
    def tearDown(self) -> None:
        sys.modules.pop("inference", None)
        os.environ.pop("PENGWIN_DISABLE_TIME_BUDGET", None)
        os.environ.pop("PENGWIN_DS538_CHECKPOINT", None)

    def test_switch_disables_budget_and_default_is_upstream(self) -> None:
        self.assertFalse(fresh_inference_module(disable_budget=True).V03_USE_TIME_BUDGET)
        self.assertTrue(fresh_inference_module(disable_budget=False).V03_USE_TIME_BUDGET)

    def test_loader_refuses_cpu_inference(self) -> None:
        sys.modules.pop("inference", None)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                backend_cli.import_inference_backend(BACKEND_CONFIG)

    def test_loader_disables_budget(self) -> None:
        sys.modules.pop("inference", None)
        with mock.patch("torch.cuda.is_available", return_value=True):
            backend = backend_cli.import_inference_backend(BACKEND_CONFIG)
        self.assertFalse(backend.V03_USE_TIME_BUDGET)

    def test_loader_refuses_a_cached_module_with_budget_on(self) -> None:
        fresh_inference_module(disable_budget=False)  # already imported with the budget active
        with mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "time budget is still active"):
                backend_cli.import_inference_backend(BACKEND_CONFIG)


class FragmentCheckpointFixTests(unittest.TestCase):
    """F8: Fragment inference uses the configured checkpoint (checkpoint_final)."""

    def tearDown(self) -> None:
        sys.modules.pop("inference", None)
        os.environ.pop("PENGWIN_DISABLE_TIME_BUDGET", None)
        os.environ.pop("PENGWIN_DS538_CHECKPOINT", None)

    def test_switch_selects_checkpoint_and_default_is_upstream(self) -> None:
        os.environ.pop("PENGWIN_DS538_CHECKPOINT", None)
        self.assertEqual(fresh_inference_module(True).DS538_CHECKPOINT_NAME, "checkpoint_best.pth")
        with mock.patch.dict(os.environ, {"PENGWIN_DS538_CHECKPOINT": "checkpoint_final.pth"}):
            module = fresh_inference_module(True)
        self.assertEqual(module.DS538_CHECKPOINT_NAME, "checkpoint_final.pth")
        self.assertEqual(module.CHECKPOINT_NAME, "checkpoint_best.pth")   # Stage A unchanged

    def test_fragment_predictor_is_built_with_the_switch(self) -> None:
        source = (INFERENCE_DIR / "inference.py").read_text(encoding="utf-8")
        self.assertIn("DS538_FOLD,\n                checkpoint_name=DS538_CHECKPOINT_NAME,", source)

    def test_loader_sets_and_verifies_the_configured_checkpoint(self) -> None:
        sys.modules.pop("inference", None)
        with mock.patch("torch.cuda.is_available", return_value=True):
            backend = backend_cli.import_inference_backend(BACKEND_CONFIG)
        self.assertEqual(backend.DS538_CHECKPOINT_NAME, "checkpoint_final.pth")

    def test_loader_refuses_a_cached_module_with_another_checkpoint(self) -> None:
        with mock.patch.dict(os.environ, {"PENGWIN_DS538_CHECKPOINT": "checkpoint_best.pth"}):
            fresh_inference_module(True)
        with mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "config requires checkpoint_final.pth"):
                backend_cli.import_inference_backend(BACKEND_CONFIG)

    def test_unsupported_checkpoint_name_is_rejected(self) -> None:
        bad = {"fragment_stage": {"inference_checkpoint": "checkpoint_latest.pth"}}
        with self.assertRaises(ValueError):
            backend_cli.fragment_inference_checkpoint(bad)

    def test_project_and_member_inference_use_checkpoint_final(self) -> None:
        import stage3_h1  # type: ignore[import-not-found]

        stage2 = backend_cli.load_config(Path("configs/stage2_backend.json"))
        self.assertEqual(backend_cli.fragment_inference_checkpoint(stage2), "checkpoint_final.pth")
        config = stage3_h1.load_config(Path("configs/stage3_h1.json"))
        member = stage3_h1.member_stage2_config(config, "member_02")
        self.assertEqual(stage3_h1.inference_checkpoint_path(member, "fragment").name, "checkpoint_final.pth")
        self.assertEqual(stage3_h1.inference_checkpoint_path(member, "anatomy").name, "checkpoint_best.pth")


class PaintBlockMaskFixTests(unittest.TestCase):
    """F9 (D8): a bone-skeleton fallback mask must not block other anatomies at paint time."""

    def tearDown(self) -> None:
        sys.modules.pop("inference", None)
        for key in ("PENGWIN_DISABLE_TIME_BUDGET", "PENGWIN_DS538_CHECKPOINT", "PENGWIN_CONFINE_DS539_ONLY"):
            os.environ.pop(key, None)

    @staticmethod
    def femur_scan_masks():
        # 1-D toy volume: Stage A femur at 4..7; RightHip fallback covers all bone 2..9.
        femur = np.zeros(10, dtype=bool)
        femur[4:8] = True
        sacrum = np.zeros(10, dtype=bool)
        sacrum[0:2] = True
        fallback = np.zeros(10, dtype=bool)
        fallback[2:10] = True
        return {"Sacrum": {"mask": sacrum, "source": "ds539"},
                "RightHip": {"mask": fallback, "source": "bone_fallback"},
                "Femur": {"mask": femur, "source": "ds539"},
                "LeftHip": {"mask": None, "source": "none"}}

    def test_fallback_mask_no_longer_blocks_the_femur(self) -> None:
        module = fresh_inference_module(True)
        merged = self.femur_scan_masks()
        upstream = module._paint_block_mask(merged, "Femur", (10,), ds539_only=False)
        fixed = module._paint_block_mask(merged, "Femur", (10,), ds539_only=True)
        self.assertTrue(upstream[4:8].all())        # D8: femur fully blocked
        self.assertFalse(fixed[4:8].any())          # F9: femur free to paint
        self.assertTrue(fixed[0:2].all())           # a Ds539 neighbour still blocks

    def test_ds539_masks_block_exactly_as_upstream(self) -> None:
        module = fresh_inference_module(True)
        merged = self.femur_scan_masks()
        femur, sacrum = merged["Femur"]["mask"], merged["Sacrum"]["mask"]
        np.testing.assert_array_equal(
            module._paint_block_mask(merged, "Sacrum", (10,), ds539_only=True), femur)
        np.testing.assert_array_equal(
            module._paint_block_mask(merged, "RightHip", (10,), ds539_only=True), femur | sacrum)

    def test_switch_and_default_is_upstream(self) -> None:
        os.environ.pop("PENGWIN_CONFINE_DS539_ONLY", None)
        self.assertFalse(fresh_inference_module(True).V03_CONFINE_DS539_ONLY)
        with mock.patch.dict(os.environ, {"PENGWIN_CONFINE_DS539_ONLY": "1"}):
            self.assertTrue(fresh_inference_module(True).V03_CONFINE_DS539_ONLY)

    def test_paint_step_uses_the_helper(self) -> None:
        source = (INFERENCE_DIR / "inference.py").read_text(encoding="utf-8")
        self.assertIn("other = _paint_block_mask(merged, anatomy, img_shape, V03_CONFINE_DS539_ONLY)", source)

    def test_loader_sets_the_switch(self) -> None:
        sys.modules.pop("inference", None)
        with mock.patch("torch.cuda.is_available", return_value=True):
            backend = backend_cli.import_inference_backend(BACKEND_CONFIG)
        self.assertTrue(backend.V03_CONFINE_DS539_ONLY)

    def test_loader_refuses_a_cached_module_without_the_switch(self) -> None:
        with mock.patch.dict(os.environ, {"PENGWIN_DS538_CHECKPOINT": "checkpoint_final.pth",
                                          "PENGWIN_CONFINE_DS539_ONLY": "0"}):
            fresh_inference_module(True)
        with mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "bone-fallback masks still block"):
                backend_cli.import_inference_backend(BACKEND_CONFIG)


class D6DiagnosisRecorderTests(unittest.TestCase):
    def test_recorder_attributes_each_step_to_its_anatomy(self) -> None:
        import diagnose_d6_sacrum as d6  # type: ignore[import-not-found]

        logged: list[str] = []
        fake = SimpleNamespace(
            log=logged.append,
            bbox_from_mask=lambda mask: (slice(1, 3), slice(0, 2), slice(0, 2)),
            resample_label_map_to_original=lambda label_pp, props, predictor: np.repeat(label_pp, 2, axis=0),
        )
        recorder = d6.Recorder(fake)
        recorder.install()
        pp = np.zeros((2, 2, 2), dtype=np.int32)
        pp[0, 0, 0], pp[1, 1, 1] = 1, 2
        fake.bbox_from_mask(None)
        fake.log("[Sacrum] bbox crop=(2, 2, 2) fraction=1% source=ds539 ETA=1s elapsed=1s")
        fake.resample_label_map_to_original(pp, {}, None)
        fake.log("[Sacrum] painted 3 voxels, 2 fragments in range [1,50]")
        fake.log("[LeftHip] no usable mask, emit zero")
        row = recorder.rows["Sacrum"]
        self.assertEqual(row["bbox"], [[1, 3], [0, 2], [0, 2]])
        self.assertEqual(row["after_decode_pp"], {"voxels": 2, "instances": 2, "largest": [1, 1]})
        self.assertEqual(row["after_resample_crop"]["voxels"], 4)
        self.assertEqual(row["pp_to_original_scale"], 2.0)
        self.assertEqual((row["painted_voxels"], row["painted_fragments"]), (3, 2))
        self.assertEqual(recorder.rows["LeftHip"]["note"], "no usable mask, emit zero")
        self.assertEqual(len(logged), 3)  # original log still receives every line

    def test_model_foreground_counts(self) -> None:
        import diagnose_d6_sacrum as d6  # type: ignore[import-not-found]

        probs = np.zeros((13, 2, 2, 2), dtype=np.float16)
        probs[0] = 1.0                      # background everywhere ...
        probs[0, 0, 0, 0], probs[3, 0, 0, 0] = 0.0, 1.0   # ... except one core voxel
        probs[4:7, 0, 0, 0] = 0.8
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probs13_Sacrum.npy"
            np.save(path, probs)
            fg = d6.model_foreground(path)
        self.assertEqual(fg["fg_prob_gt_0.5_voxels_pp"], 1)
        self.assertEqual(fg["argmax_class_voxels_pp"], {"bg": 7, "border": 0, "boundary": 0, "core": 1})
        self.assertAlmostEqual(fg["mean_short_affinity_in_fg"], 0.8, places=2)


class FragmentGateTests(unittest.TestCase):
    def test_foreground_metrics(self) -> None:
        import check_fragment_gate as fg  # type: ignore[import-not-found]

        label = np.zeros((2, 2, 2), dtype=np.uint8)
        label[0] = 3                                    # 4 GT voxels, any fragment id
        pred = np.zeros((2, 2, 2), dtype=bool)
        pred[0, 0], pred[1, 0, 0] = True, True          # 2 hit + 1 false positive
        m = fg.foreground_metrics(pred, label)
        self.assertAlmostEqual(m["dice"], 4 / 7)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        with self.assertRaisesRegex(RuntimeError, "differ"):
            fg.foreground_metrics(pred[:1], label)

    def test_gate_fails_a_single_bad_bone(self) -> None:
        import check_fragment_gate as fg  # type: ignore[import-not-found]

        rows = [{"bone": "LeftHip", "dice": 0.97}] * 5 + [{"bone": "RightHip", "dice": 0.96}] * 5
        rows += [{"bone": "Sacrum", "dice": d} for d in (0.95, 0.94, 0.2, 0.3, 0.1)]
        result = fg.gate(rows, 0.90, 2)
        self.assertTrue(result["LeftHip"]["pass"] and result["RightHip"]["pass"])
        self.assertFalse(result["Sacrum"]["pass"])      # 3 crops < 0.5
        self.assertFalse(result["PASS"])
        self.assertNotIn("Femur", result)
        self.assertTrue(fg.gate(rows[:10], 0.90, 2)["PASS"])

    def test_bone_from_crop_name_and_split(self) -> None:
        import check_fragment_gate as fg  # type: ignore[import-not-found]

        self.assertEqual(fg.crop_bone("PENGWIN_030_Sacrum"), "Sacrum")
        with self.assertRaises(ValueError):
            fg.crop_bone("PENGWIN_030_Pelvis")
        crops = fg.validation_crops(Path("outputs/stage2_backend/nnUNet_preprocessed"), FRAGMENT)
        self.assertIn("PENGWIN_030_Sacrum", crops)
        self.assertEqual(len(crops), 80)


class Stage3MemberWiringTests(unittest.TestCase):
    def test_member_runs_read_the_shared_split_manifest(self) -> None:
        import stage3_h1  # type: ignore[import-not-found]

        config = stage3_h1.load_config(Path("configs/stage3_h1.json"))
        shared = backend_cli.stage2_paths(backend_cli.load_config(Path(config["stage2_config"])))
        for member in ("member_01", "member_02", "member_03"):
            paths = backend_cli.stage2_paths(stage3_h1.member_stage2_config(config, member))
            self.assertEqual(paths["manifest"], shared["manifest"])
            self.assertNotEqual(paths["results"], shared["results"])


if __name__ == "__main__":
    unittest.main()
