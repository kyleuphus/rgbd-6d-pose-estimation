import json
import numpy as np
from pathlib import Path


def load_predicted_pose(project_root, folder_id, image_id, object_id, pose_root=None):
    pose_root = (
        Path(pose_root) if pose_root is not None else Path(project_root) / "outputs"
    )
    pose_path = (
        pose_root
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}.npy"
    )

    pose = np.load(pose_path)

    R = pose[:3, :3]
    t = pose[:3, 3]
    return (R, t)


def load_reference_pose(project_root, folder_id, image_id, object_id):
    dataset_root = project_root / "data" / "ycbv"
    scene_dir = Path(dataset_root) / "test" / f"{int(folder_id):06d}"
    with (scene_dir / "scene_gt.json").open(encoding="utf-8") as file:
        scene_gt = json.load(file)

    annotations = scene_gt[str(int(image_id))]

    R = None
    t = None
    for annotation in annotations:
        if annotation["obj_id"] == object_id:
            R = np.array(annotation["cam_R_m2c"], dtype=float).reshape(3, 3)
            t = np.array(annotation["cam_t_m2c"], dtype=float) / 1000

            break
    else:
        raise ValueError(f"Object {object_id} is missing from image {image_id}")

    return (R, t)
