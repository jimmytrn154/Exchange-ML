"""Stage 3 backend fixes F1 (Anatomy L/R mirror), F2 (DDP-global Fragment validation),
and D3 (Anatomy partial-label case map)."""

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
