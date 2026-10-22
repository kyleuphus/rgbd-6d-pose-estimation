from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np

from data_types import CameraIntrinsics, RGBDFrame


def load_mustard_frame(frame_id: str, intrinsics: CameraIntrinsics) -> RGBDFrame:
    project_root = Path(__file__).resolve().parents[1]
    sequence_dir = project_root / "data" / "mustard0"

    filename = f"{frame_id}.png"
    rgb_path = sequence_dir / "rgb" / filename
    depth_path = sequence_dir / "depth" / filename

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")

    rgb = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.uint8)
    depth = depth_raw.astype(np.float32) * np.float32(0.001)

    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
    )


def iter_mustard_frames() -> Iterator[tuple[str, RGBDFrame]]:
    sequence_dir = Path(__file__).resolve().parents[1] / "data" / "mustard0"
    rgb_dir = sequence_dir / "rgb"
    depth_dir = sequence_dir / "depth"

    rgb_paths = sorted(
        rgb_dir.glob("*.png"),
        key=lambda path: int(path.stem),
    )

    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")

    rgb_names = {path.name for path in rgb_paths}
    depth_names = {path.name for path in depth_dir.glob("*.png")}

    missing_depth = rgb_names - depth_names
    if missing_depth:
        raise FileNotFoundError(
            f"Missing matching depth files: {sorted(missing_depth)[:5]}"
        )

    print(
        f"Found {len(rgb_paths)} RGB/depth pairs; "
        f"{len(depth_names - rgb_names)} depth files have no matching RGB."
    )

    K = np.loadtxt(sequence_dir / "cam_K.txt")
    if K.shape != (3, 3):
        raise ValueError("Expected a 3×3 camera intrinsic matrix")

    first_image = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_paths[0]}")

    height, width = first_image.shape[:2]
    del first_image

    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )

    for rgb_path in rgb_paths:
        frame_id = rgb_path.stem
        frame = load_mustard_frame(frame_id, intrinsics)
        yield frame_id, frame
