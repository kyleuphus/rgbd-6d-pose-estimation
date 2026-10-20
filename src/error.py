import math

from pathlib import Path
from pose_io import load_predicted_pose, load_reference_pose

project_root = Path(__file__).resolve().parents[1]
dataset_root = project_root / "data" / "ycbv"
folder_id = "000048"
image_id = "000001"
object_id = 1


def calculate_error(project_root, folder_id, image_id, object_id):
    t_predicted = load_predicted_pose(project_root, folder_id, image_id, object_id)[1]
    t_reference = load_reference_pose(project_root, folder_id, image_id, object_id)[1]

    delta_t = t_predicted - t_reference

    error = math.sqrt(delta_t[0] ** 2 + delta_t[1] ** 2 + delta_t[2] ** 2) * 1000
    return error


print(calculate_error(project_root, folder_id, image_id, object_id), "mm")
