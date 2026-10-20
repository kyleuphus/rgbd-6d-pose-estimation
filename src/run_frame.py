from pathlib import Path

import numpy as np

from ycb_video import load_frame, load_target
from foundation_pose_backend import FoundationPoseBackend


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "ycbv"

    folder_id = "000048"
    image_id = "000001"
    object_id = 1

    frame = load_frame(dataset_root, folder_id, image_id)
    mesh, mask = load_target(dataset_root, folder_id, image_id, object_id)

    backend = FoundationPoseBackend(mesh)

    pose = backend.estimate(frame, mask)

    print("Object-to-camera transform:")
    print(pose)
    print("Position in camera coordinates, meters:")
    print(pose[:3, 3])

    output_dir = (
        project_root / "outputs" / f"{folder_id}_{image_id}_obj_{object_id:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (f"pose_{folder_id}_{image_id}_obj_{object_id:06d}.npy")
    np.save(output_path, pose)
    print(f"Saved pose to {output_path}")


if __name__ == "__main__":
    main()
