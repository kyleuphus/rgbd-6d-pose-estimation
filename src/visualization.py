import numpy as np
import cv2
import json
import argparse

from ycb_video import load_intrinsics, load_frame
from pathlib import Path
from pose_io import load_predicted_pose, load_reference_pose


def draw_pose_axes(
    rgb,
    intrinsics,
    project_root,
    folder_id,
    image_id,
    object_id,
    axis_length=0.05,
    *,
    pose_root,
):

    rgb = rgb.copy()

    points_object = np.array(
        [
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # +X endpoint
            [0, axis_length, 0],  # +Y endpoint
            [0, 0, axis_length],  # +Z endpoint
        ]
    )

    R, t = load_predicted_pose(
        project_root, folder_id, image_id, object_id, pose_root=pose_root
    )

    pixels = []

    for point_object in points_object:
        point_camera = R @ point_object + t

        if point_camera[2] <= 0:
            pixels.append(None)
            continue

        point_projected = intrinsics.K @ point_camera
        coordinate = (point_projected[:2] / point_projected[2]).astype(int)
        pixels.append(coordinate)

    cv2.circle(img=rgb, center=pixels[0], radius=0, color=(0, 0, 0), thickness=-1)
    cv2.line(img=rgb, pt1=pixels[0], pt2=pixels[1], color=(255, 0, 0))
    cv2.line(img=rgb, pt1=pixels[0], pt2=pixels[2], color=(0, 255, 0))
    cv2.line(img=rgb, pt1=pixels[0], pt2=pixels[3], color=(0, 0, 255))

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.rectangle(img_bgr, (8, 8), (130, 45), (0, 0, 0), -1)

    cv2.putText(
        img_bgr,
        "Predicted",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    write_path = (
        pose_root
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}_annotated.png"
    )
    cv2.imwrite(write_path, img_bgr)

    return img_bgr


def draw_reference(
    rgb,
    intrinsics,
    project_root,
    folder_id,
    image_id,
    object_id,
    axis_length=0.05,
    *,
    pose_root,
):

    R, t = load_reference_pose(project_root, folder_id, image_id, object_id)

    points_object = np.array(
        [
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # +X endpoint
            [0, axis_length, 0],  # +Y endpoint
            [0, 0, axis_length],  # +Z endpoint
        ]
    )

    pixels = []

    for point_object in points_object:
        point_camera = R @ point_object + t

        if point_camera[2] <= 0:
            pixels.append(None)
            continue

        point_projected = intrinsics.K @ point_camera
        coordinate = (point_projected[:2] / point_projected[2]).astype(int)
        pixels.append(coordinate)

    img = rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    if pixels[0] is not None:
        origin = tuple(pixels[0])
        cv2.circle(img, origin, radius=4, color=(0, 0, 0), thickness=-1)

        for endpoint, color in zip(pixels[1:], colors):
            if endpoint is not None:
                cv2.line(img=img, pt1=origin, pt2=endpoint, color=color)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.rectangle(img_bgr, (8, 8), (140, 45), (0, 0, 0), -1)

    cv2.putText(
        img_bgr,
        "Reference",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    write_path = (
        pose_root
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}_reference.png"
    )
    cv2.imwrite(write_path, img_bgr)

    return img_bgr


def create_comparison(
    annotated, reference, project_root, folder_id, image_id, object_id, *, pose_root
):
    combined = np.hstack((annotated, reference))

    write_path = (
        pose_root
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}_comparison.png"
    )

    cv2.imwrite(write_path, combined)


def main():
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

    for index, case in enumerate(cases, start=1):
        folder_id = f"{int(case['folder_id']):06d}"
        image_id = f"{int(case['image_id']):06d}"
        object_id = int(case["object_id"])

        frame = load_frame(dataset_root, folder_id, image_id)

        annotated = draw_pose_axes(
            frame.rgb,
            frame.intrinsics,
            project_root,
            folder_id,
            image_id,
            object_id,
            pose_root=pose_root,
        )

        reference = draw_reference(
            frame.rgb,
            frame.intrinsics,
            project_root,
            folder_id,
            image_id,
            object_id,
            pose_root=pose_root,
        )

        create_comparison(
            annotated,
            reference,
            project_root,
            folder_id,
            image_id,
            object_id,
            pose_root=pose_root,
        )

        print(f"[{index}/{len(cases)}] {folder_id}_{image_id}_obj_{object_id:06d}")


if __name__ == "__main__":
    main()
