from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CameraIntrinsics:
    """Calibration for processed image being used by pipeline"""

    width: int
    height: int

    fx: float  # horizontal focal length
    fy: float  # vertical focal length
    cx: float  # principal point's horizontal coordinate
    cy: float  # principal point's vertical coordinate

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")

        values = [self.fx, self.fy, self.cx, self.cy]
        if not np.isfinite(values).all():
            raise ValueError("Intrinsics must contain finite values")

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")

    @property
    def K(self) -> NDArray[np.float64]:
        """Return the 3×3 intrinsic matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class RGBDFrame:
    """Aligned RGB and depth image"""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    intrinsics: CameraIntrinsics

    def __post_init__(self):
        height = self.intrinsics.height
        width = self.intrinsics.width

        if self.rgb.shape != (height, width, 3):
            raise ValueError("RGB shape does not match the intrinsics")

        if self.depth.shape != (height, width):
            raise ValueError("Depth shape does not match the intrinsics")

        if self.rgb.dtype != np.uint8:
            raise ValueError("RGB must use uint8")

        if self.depth.dtype != np.float32:
            raise ValueError("Depth must use float32")
