"""RGB pose overlays, with a command for visualizing saved YCB batch runs."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def overlay_pose_axes(rgb, intrinsics, pose, axis_length=0.05):
    """Return an RGB copy with XYZ axes for a 4x4 object-to-camera pose.

    intrinsics must expose a 3x3 K matrix. Axis length uses the pose's units
    (meters for this project's poses). Axes behind the camera are skipped.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be a finite 4x4 object-to-camera matrix")
    if not np.isfinite(axis_length) or axis_length <= 0:
        raise ValueError("Axis length must be positive and finite")

    points_object = np.array(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]
    )
    R, t = pose[:3, :3], pose[:3, 3]
    pixels = []
    for point_object in points_object:
        point_camera = R @ point_object + t
        if point_camera[2] <= 0:
            pixels.append(None)
            continue
        projected = intrinsics.K @ point_camera
        pixels.append(tuple(np.rint(projected[:2] / projected[2]).astype(int)))

    image = rgb.copy()
    if pixels[0] is not None:
        origin = pixels[0]
        cv2.circle(image, origin, radius=4, color=(0, 0, 0), thickness=-1)
        for endpoint, color in zip(pixels[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            if endpoint is not None:
                cv2.line(image, origin, endpoint, color=color)
    return image


def _label_image(rgb, label):
    image = rgb.copy()
    cv2.rectangle(image, (8, 8), (140, 45), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_reference(rgb, intrinsics, pose):
    """Return an RGB reference image with pose axes and a Reference label."""
    return _label_image(overlay_pose_axes(rgb, intrinsics, pose), "Reference")


def create_comparison(annotated, reference):
    """Return the annotated and reference images side by side, in that order."""
    return np.hstack((annotated, reference))


def main():
    # Dataset-specific loading and saving belong to this command, not the helpers.
    from ycb_video import load_frame
    from pose_io import load_predicted_pose, load_reference_pose

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "ycbv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    pose_root = args.pose_root if args.pose_root is not None else run_dir
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cases = [result for result in results if result["status"] == "ok"]
    print(
        f"Generating {len(cases)} comparisons; "
        f"skipping {len(results) - len(cases)} failed cases."
    )

    def make_pose(rotation, translation):
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        return pose

    for index, case in enumerate(cases, start=1):
        folder_id = f"{int(case['folder_id']):06d}"
        image_id = f"{int(case['image_id']):06d}"
        object_id = int(case["object_id"])
        frame = load_frame(dataset_root, folder_id, image_id)

        predicted_pose = make_pose(
            *load_predicted_pose(
                project_root, folder_id, image_id, object_id, pose_root=pose_root
            )
        )
        reference_pose = make_pose(
            *load_reference_pose(project_root, folder_id, image_id, object_id)
        )
        annotated = _label_image(
            overlay_pose_axes(frame.rgb, frame.intrinsics, predicted_pose), "Predicted"
        )
        reference = draw_reference(frame.rgb, frame.intrinsics, reference_pose)
        combined = create_comparison(annotated, reference)

        name = f"{folder_id}_{image_id}_obj_{object_id:06d}"
        output_dir = pose_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, image in (
            ("annotated", annotated),
            ("reference", reference),
            ("comparison", combined),
        ):
            path = output_dir / f"pose_{name}_{suffix}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not save visualization: {path}")

        print(f"[{index}/{len(cases)}] {name}")


if __name__ == "__main__":
    main()
