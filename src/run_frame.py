from pathlib import Path

import numpy as np
import argparse

from ycb_video import load_frame, load_mesh
from foundation_pose_backend import FoundationPoseBackend
from run_segmentation import construct_model, segment_images


def run_frame(folder_id, image_id, object_id):

    folder_id = f"{int(folder_id):06d}"
    image_id = f"{int(image_id):06d}"

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "ycbv"

    frame = load_frame(dataset_root, folder_id, image_id)
    mesh = load_mesh(dataset_root, object_id)

    segmenter = construct_model(project_root)
    mask = segment_images(
        segmenter, folder_id, image_id, [object_id], project_root, save_mask=True
    )[object_id,]

    backend = FoundationPoseBackend(mesh)
    pose = backend.estimate(frame, mask)

    print("Object-to-camera transform:")
    print(pose)
    print("Position in camera coordinates, meters:")
    print(pose[:3, 3])

    output_dir = (
        project_root
        / "outputs"
        / "efficienttam"
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (f"pose_{folder_id}_{image_id}_obj_{object_id:06d}.npy")
    np.save(output_path, pose)
    print(f"Saved pose to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--object-id", type=int, required=True)
    args = parser.parse_args()

    run_frame(args.folder_id, args.image_id, args.object_id)


if __name__ == "__main__":
    main()
