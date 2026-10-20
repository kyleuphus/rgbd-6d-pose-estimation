import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
import trimesh

from data_types import CameraIntrinsics, RGBDFrame


def load_frame(dataset_root: str | Path, folder_id: str, image_id: str) -> RGBDFrame:
    """Load a BOP YCB-Video frame with RGB color and depth in meters.

    dataset_root points to the ycbv directory containing test/.
    Scene and image IDs can be supplied with or without leading zeros.
    """
    scene_dir = Path(dataset_root) / "test" / f"{int(folder_id):06d}"

    image_number = int(image_id)
    image_filename = f"{image_number:06d}.png"
    camera_key = str(image_number)

    rgb_path = scene_dir / "rgb" / image_filename
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise FileNotFoundError(f"Could not load RGB: {rgb_path}")

    rgb_matrix = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    depth_path = scene_dir / "depth" / image_filename

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        raise FileNotFoundError(f"Could not load depth: {depth_path}")

    with (scene_dir / "scene_camera.json").open(encoding="utf-8") as file:
        camera_entry = json.load(file)[camera_key]

    # BOP depth_scale converts stored values to millimeters.
    meters_per_stored_unit = np.float32(camera_entry["depth_scale"] / 1000.0)
    depth_matrix = depth_raw.astype(np.float32) * meters_per_stored_unit

    camera_intrinsics = load_intrinsics(dataset_root, folder_id, image_id)

    frame = RGBDFrame(rgb=rgb_matrix, depth=depth_matrix, intrinsics=camera_intrinsics)

    return frame


def load_intrinsics(
    dataset_root: str | Path, folder_id: str, image_id: str
) -> CameraIntrinsics:
    """Read the selected frame's calibration at its original image size."""
    scene_dir = Path(dataset_root) / "test" / f"{int(folder_id):06d}"

    image_number = int(image_id)
    image_filename = f"{image_number:06d}.png"
    camera_key = str(image_number)

    with (scene_dir / "scene_camera.json").open(encoding="utf-8") as file:
        camera_data = json.load(file)

    rgb_path = scene_dir / "rgb" / image_filename
    img = cv2.imread(str(rgb_path))

    if img is None:
        raise FileNotFoundError(f"Could not load RGB: {rgb_path}")

    img_height, img_width = img.shape[:2]

    K = camera_data[camera_key]["cam_K"]
    intrinsics = CameraIntrinsics(
        width=img_width, height=img_height, fx=K[0], fy=K[4], cx=K[2], cy=K[5]
    )

    return intrinsics


def load_target(
    dataset_root: str | Path,
    folder_id: str,
    image_id: str,
    object_id: int,
) -> tuple[trimesh.Trimesh, NDArray[np.bool_]]:
    """Load an object's mesh in meters and its visible mask for one frame.

    The mask suffix is the object's annotation index, not its object ID.
    A frame must contain exactly one instance of the requested object.
    """
    mask = load_visible_mask(dataset_root, folder_id, image_id, object_id)
    return load_mesh(dataset_root, object_id), mask


def load_visible_mask(
    dataset_root: str | Path,
    folder_id: str,
    image_id: str,
    object_id: int,
) -> NDArray[np.bool_]:
    """Load the selected instance's visible mask without reloading its mesh."""
    dataset_root = Path(dataset_root)
    scene_dir = dataset_root / "test" / f"{int(folder_id):06d}"
    image_number = int(image_id)

    with (scene_dir / "scene_gt.json").open(encoding="utf-8") as file:
        scene_annotations = json.load(file)

    camera_key = str(image_number)
    if camera_key not in scene_annotations:
        raise ValueError(f"Image {image_number} is not annotated in {scene_dir}")

    annotations = scene_annotations[camera_key]
    matches = [
        index
        for index, annotation in enumerate(annotations)
        if annotation["obj_id"] == object_id
    ]

    if not matches:
        raise ValueError(f"Object {object_id} is not present in image {image_number}")
    if len(matches) > 1:
        raise ValueError(
            f"Object {object_id} has multiple instances in image {image_number}; "
            "selecting an instance is not yet supported"
        )

    annotation_index = matches[0]
    mask_path = (
        scene_dir / "mask_visib" / f"{image_number:06d}_{annotation_index:06d}.png"
    )
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Could not load target mask: {mask_path}")
    mask = mask_raw > 0
    if not mask.any():
        raise ValueError(f"Object {object_id} has no visible pixels in {mask_path}")

    return mask


def load_mesh(dataset_root: str | Path, object_id: int) -> trimesh.Trimesh:
    """Load a BOP object mesh in meters."""
    dataset_root = Path(dataset_root)
    # Support the standard BOP layout and a separately extracted models archive.
    mesh_filename = f"obj_{object_id:06d}.ply"
    mesh_paths = [
        dataset_root / "models" / mesh_filename,
        dataset_root / "models" / "models" / mesh_filename,
    ]
    mesh_path = next((path for path in mesh_paths if path.is_file()), None)
    if mesh_path is None:
        searched_paths = ", ".join(str(path) for path in mesh_paths)
        raise FileNotFoundError(
            f"Could not find mesh for object {object_id}; searched: {searched_paths}"
        )

    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh in {mesh_path}")
    mesh.apply_scale(0.001)  # BOP model coordinates are in millimeters.

    return mesh
