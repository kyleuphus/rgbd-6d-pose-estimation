from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray
import trimesh

from data_types import RGBDFrame

foundationpose_dir = Path(__file__).resolve().parents[2] / "FoundationPose"
sys.path.insert(0, str(foundationpose_dir))

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor


class FoundationPoseBackend:
    def __init__(self, mesh: trimesh.Trimesh, *, scorer=None, refiner=None, glctx=None):
        self.mesh = mesh
        self.scorer = scorer if scorer is not None else ScorePredictor()
        self.refiner = refiner if refiner is not None else PoseRefinePredictor()

        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=glctx,
            debug=0,
            debug_dir=str(
                Path(__file__).resolve().parents[1] / "outputs" / "foundationspose"
            ),
        )

    def estimate(
        self, frame: RGBDFrame, mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        if mask.shape != frame.depth.shape:
            raise ValueError("Mask must match the depth image dimensions")

        pose = self.estimator.register(
            K=frame.intrinsics.K,
            rgb=frame.rgb,
            depth=frame.depth,
            ob_mask=mask,
            iteration=5,
        )

        return np.asarray(pose, dtype=np.float32)

    def track(self, frame):
        pose = self.estimator.track_one(
            rgb=frame.rgb,
            depth=frame.depth,
            K=frame.intrinsics.K,
            iteration=2,
        )

        return np.asarray(pose, dtype=np.float32)
