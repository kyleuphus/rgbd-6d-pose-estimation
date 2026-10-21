from pathlib import Path

import numpy as np
import argparse

from ycb_video import load_frame, load_mesh
import nvdiffrast.torch as dr
from foundation_pose_backend import (
    FoundationPoseBackend,
    ScorePredictor,
    PoseRefinePredictor,
)
from run_segmentation import construct_model, segment_images
from yolo import load_yolo, segment_image, pair_masks
from collections import defaultdict


def run_frame(folder_id, image_id, object_id, backend):

    folder_id = f"{int(folder_id):06d}"
    image_id = f"{int(image_id):06d}"

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "ycbv"

    frame = load_frame(dataset_root, folder_id, image_id)

    segmenter = construct_model(project_root)
    masks = segment_images(
        segmenter, folder_id, image_id, [object_id], project_root, save_mask=True
    )
    mask = masks[object_id]

    pose = backend.estimate(frame, mask)

    # print("Object-to-camera transform:")
    # print(pose)
    # print("Position in camera coordinates, meters:")
    # print(pose[:3, 3])
    #
    # output_dir = (
    #     project_root
    #     / "outputs"
    #     / "efficienttam"
    #     / f"{folder_id}_{image_id}_obj_{object_id:06d}"
    # )
    # output_dir.mkdir(parents=True, exist_ok=True)
    #
    # output_path = output_dir / (f"pose_{folder_id}_{image_id}_obj_{object_id:06d}.npy")
    # np.save(output_path, pose)
    # print(f"Saved pose to {output_path}")

    return pose


def run_frame_yolo(
    folder_id, image_id, meshes, backends, object_id_to_class_id
):  # meshes = {ycb object_id: mesh}
    folder_id = f"{int(folder_id):06d}"
    image_id = f"{int(image_id):06d}"

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "ycbv"

    ycb_objects = list(meshes)
    yolo_classes = [object_id_to_class_id[object_id] for object_id in ycb_objects]

    if len(yolo_classes) != len(set(yolo_classes)):
        raise ValueError("Configure only one YCB object per YOLO class")

    frame = load_frame(dataset_root, folder_id, image_id)
    model = load_yolo()

    results = segment_image(frame.rgb, model, yolo_classes)
    mask_pairs = pair_masks(results)

    poses = defaultdict(list)

    for object_id in ycb_objects:
        class_id = object_id_to_class_id[object_id]
        masks = mask_pairs.get(class_id, [])

        if len(masks) != 1:
            raise ValueError(
                f"YCB object {object_id}, YOLO class {class_id}: "
                f"expected one mask, found {len(masks)}"
            )

        backend = backends[object_id]
        pose = backend.estimate(frame, masks[0])
        poses[object_id].append(pose)

    return poses


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "ycbv"

    object_id_to_class_id = {14: 41}

    meshes = {
        object_id: load_mesh(dataset_root, object_id)
        for object_id in object_id_to_class_id
    }

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    backends = {
        object_id: FoundationPoseBackend(
            mesh, scorer=scorer, refiner=refiner, glctx=glctx
        )
        for object_id, mesh in meshes.items()
    }

    poses = run_frame_yolo(
        folder_id="000048",
        image_id="000001",
        meshes=meshes,
        backends=backends,
        object_id_to_class_id=object_id_to_class_id,
    )

    for object_id, object_poses in poses.items():
        print(f"YCB object {object_id}:")
        print(object_poses[0])


if __name__ == "__main__":
    main()
