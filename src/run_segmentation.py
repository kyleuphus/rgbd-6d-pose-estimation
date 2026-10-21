import json
import numpy as np
from pathlib import Path

from ycb_video import load_frame
from segmentation import EfficientTAMSegmenter


def construct_model(project_root):

    eta_dir = project_root.parent / "EfficientTAM"
    checkpoint = eta_dir / "checkpoints" / "efficienttam_s.pt"
    model_cfg = "configs/efficienttam/efficienttam_s.yaml"

    segmenter = EfficientTAMSegmenter(
        checkpoint=checkpoint, model_cfg=model_cfg, device="cpu"
    )

    return segmenter


def segment_images(
    segmenter, folder_id, image_id, object_ids, project_root, save_mask=False,
    output_dir=None,
):

    project_root = Path(project_root)
    dataset_root = Path(project_root / "data" / "ycbv")
    scene_dir = dataset_root / "test" / f"{int(folder_id):06d}"
    image_number = int(image_id)

    rgb = load_frame(dataset_root, folder_id, image_id).rgb

    with (scene_dir / "scene_gt_info.json").open(encoding="utf-8") as file:
        scene_gt_info = json.load(file)

    with (scene_dir / "scene_gt.json").open(encoding="utf-8") as file:
        scene_gt = json.load(file)

    image_key = str(image_number)

    requested_ids = set(object_ids)
    boxes = {}

    for annotation_index, annotation in enumerate(scene_gt[image_key]):
        object_id = annotation["obj_id"]

        if object_id in requested_ids:
            x, y, width, height = scene_gt_info[image_key][annotation_index][
                "bbox_visib"
            ]

            boxes[object_id] = np.array([x, y, x + width, y + height])

    missing_ids = requested_ids - boxes.keys()

    if missing_ids:
        raise ValueError(f"Objects missing from image: {sorted(missing_ids)}")

    segmenter.set_image(rgb)

    masks = {}

    for object_id, box in boxes.items():
        masks[object_id] = segmenter.predict_mask(box)

    if save_mask:
        mask_dir = (
            Path(output_dir) if output_dir is not None
            else project_root / "outputs" / "segmentation" / "efficienttam"
        )
        mask_dir.mkdir(parents=True, exist_ok=True)

        for object_id, mask in masks.items():
            filename = (
                f"{int(folder_id):06d}_{int(image_id):06d}_obj_{object_id:06d}_mask.npy"
            )
            np.save(mask_dir / filename, mask)

    return masks
