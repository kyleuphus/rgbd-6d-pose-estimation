import numpy as np
import cv2

from ycb_video import load_intrinsics, load_frame
from pathlib import Path
from pose_io import load_predicted_pose, load_reference_pose


project_root = Path(__file__).resolve().parents[1]
dataset_root = project_root / "data" / "ycbv"
folder_id = "000048"
image_id = "000001"
object_id = 1

pose_path = (
    project_root
    / "outputs"
    / f"{folder_id}_{image_id}_obj_{object_id:06d}"
    / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}.npy"
)
pose = np.load(pose_path)

intrinsics = load_intrinsics(dataset_root, folder_id, image_id)

frame = load_frame(dataset_root, folder_id, image_id)
rgb = frame.rgb.copy()


def draw_pose_axes(
    rgb,
    intrinsics,
    project_root,
    folder_id,
    image_id,
    object_id,
    axis_length=0.05,
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

    R, t = load_predicted_pose(project_root, folder_id, image_id, object_id)

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

    write_path = (
        project_root
        / "outputs"
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}_annotated.png"
    )
    cv2.imwrite(write_path, img_bgr)

    return img_bgr


def draw_reference(
    rgb, intrinsics, project_root, folder_id, image_id, object_id, axis_length=0.05
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
                cv2.line(img=rgb, pt1=origin, pt2=endpoint, color=color)

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    write_path = (
        project_root
        / "outputs"
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}_reference.png"
    )
    cv2.imwrite(write_path, img_bgr)

    return img_bgr


def create_comparison(
    annotated, reference, project_root, folder_id, image_id, object_id
):
    combined = np.hstack((annotated, reference))

    write_path = (
        project_root
        / "outputs"
        / f"{folder_id}_{image_id}_obj_{object_id:06d}"
        / f"pose_{folder_id}_{image_id}_obj_{object_id:06d}_comparison.png"
    )

    cv2.imwrite(write_path, combined)


annotated = draw_pose_axes(
    rgb,
    intrinsics,
    project_root,
    folder_id,
    image_id,
    object_id,
    axis_length=0.05,
)

reference = draw_reference(
    rgb, intrinsics, project_root, folder_id, image_id, object_id, axis_length=0.05
)

create_comparison(annotated, reference, project_root, folder_id, image_id, object_id)
