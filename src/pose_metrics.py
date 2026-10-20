import math
import numpy as np

from pathlib import Path
from pose_io import load_predicted_pose, load_reference_pose

project_root = Path(__file__).resolve().parents[1]
dataset_root = project_root / "data" / "ycbv"
folder_id = "000048"
image_id = "000001"
object_id = 1


def rotation_error_degrees(R_predicted, R_reference):
    R_difference = R_reference.T @ R_predicted
    trace = np.trace(R_difference)
    cos_error = np.clip(((trace - 1) / 2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_error))
    return angle


def translation_error_mm(t_predicted, t_reference):
    delta_t = t_predicted - t_reference

    error = math.sqrt(delta_t[0] ** 2 + delta_t[1] ** 2 + delta_t[2] ** 2) * 1000
    return error


def calculate_error(project_root, folder_id, image_id, object_id):
    R_predicted, t_predicted = load_predicted_pose(
        project_root, folder_id, image_id, object_id
    )
    R_reference, t_reference = load_reference_pose(
        project_root, folder_id, image_id, object_id
    )

    rotation_error = rotation_error_degrees(R_predicted, R_reference)
    translation_error = translation_error_mm(t_predicted, t_reference)
    return (rotation_error, translation_error)


print(calculate_error(project_root, folder_id, image_id, object_id))
