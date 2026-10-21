"""Exercise output isolation and portability without loading GPU models."""

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_batch import evaluate_cases, main as evaluate_main
from run_batch import run_batch


class RunOutputTests(unittest.TestCase):
    def test_isolated_runs_survive_zip_download_and_rename(self):
        cases = [{"folder_id": "000048", "image_id": "000001", "object_id": 1}]
        cuda = SimpleNamespace(
            is_available=lambda: True, synchronize=lambda: None,
            get_device_name=lambda: "test GPU",
        )
        backend = MagicMock()
        segmenter = MagicMock()
        segmenter.predict_mask.return_value = np.ones((2, 2), dtype=bool)
        modules = {
            "google.colab": SimpleNamespace(files=SimpleNamespace(download=MagicMock())),
            "torch": SimpleNamespace(cuda=cuda),
            "nvdiffrast": ModuleType("nvdiffrast"),
            "nvdiffrast.torch": SimpleNamespace(RasterizeCudaContext=object),
            "foundation_pose_backend": SimpleNamespace(
                FoundationPoseBackend=backend, ScorePredictor=object,
                PoseRefinePredictor=object,
            ),
            "ycb_video": SimpleNamespace(
                load_frame=lambda *args: SimpleNamespace(
                    rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth=np.ones((2, 2))
                ),
                load_mesh=lambda *args: None,
            ),
            "segmentation": SimpleNamespace(
                EfficientTAMSegmenter=lambda **kwargs: segmenter
            ),
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, modules):
            # Load the real mask-writing implementation with lightweight model stubs.
            sys.modules.pop("run_segmentation", None)
            importlib.import_module("run_segmentation")
            self.addCleanup(sys.modules.pop, "run_segmentation", None)
            root = Path(temporary)
            scene = root / "data/ycbv/test/000048"
            scene.mkdir(parents=True)
            (scene / "scene_gt.json").write_text(json.dumps({"1": [{
                "obj_id": 1, "cam_R_m2c": np.eye(3).ravel().tolist(),
                "cam_t_m2c": [0, 0, 0],
            }]}))
            (scene / "scene_gt_info.json").write_text(
                json.dumps({"1": [{"bbox_visib": [0, 0, 2, 2]}]})
            )
            for index in range(2):
                run_dir = root / "outputs" / f"run-{index}"
                pose = np.eye(4)
                pose[0, 3] = index * 0.01
                backend.return_value.estimate.return_value = pose
                with redirect_stdout(io.StringIO()):
                    results = run_batch(cases, root, run_dir / "results.json")
                self.assertEqual(results[0]["status"], "ok", results)
                self.assertEqual(json.loads((run_dir / "cases.json").read_text()), cases)
                summary = json.loads((run_dir / "timings.json").read_text())
                self.assertEqual(summary["format_version"], 1)
                for field in ("pose_path", "mask_path"):
                    path = Path(results[0][field])
                    self.assertFalse(path.is_absolute())
                    self.assertTrue((run_dir / path).is_file())
                with self.assertRaises(FileExistsError):
                    run_batch(cases, root, run_dir / "results.json")

                notebook_path = Path(__file__).resolve().parents[1] / "notebooks/run_colab.ipynb"
                notebook = json.loads(notebook_path.read_text())
                download_cell = next(
                    cell for cell in notebook["cells"] if cell.get("id") == "download-cases"
                )
                with redirect_stdout(io.StringIO()):
                    exec("".join(download_cell["source"]), {
                        "RESULTS": results, "BATCH_DIR": run_dir,
                    })
                archive_path = run_dir / "predictions.zip"
                downloaded = root / f"renamed-download-{index}"
                with ZipFile(archive_path) as archive:
                    archive.extractall(downloaded)
                rows = evaluate_cases(root, cases, pose_root=downloaded)
                self.assertAlmostEqual(rows[0]["translation_error_mm"], index * 10)
                # Standard runs need only --run-dir, even after extraction elsewhere.
                with patch.object(sys, "argv", [
                    "evaluate_batch.py", "--project-root", str(root),
                    "--run-dir", str(downloaded),
                ]), redirect_stdout(io.StringIO()):
                    evaluate_main()
                self.assertTrue((downloaded / "evaluation.csv").is_file())
                saved = json.loads((downloaded / "results.json").read_text())[0]
                self.assertTrue((downloaded / saved["mask_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
