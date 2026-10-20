import numpy as np

from efficient_track_anything.build_efficienttam import build_efficienttam
from efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)


class EfficientTAMSegmenter:
    def __init__(self, checkpoint, model_cfg, device="cpu"):

        model = build_efficienttam(
            model_cfg,
            str(checkpoint),
            device=device,
            hydra_overrides_extra=["++model.compile_image_encoder=False"],
        )

        self.predictor = EfficientTAMImagePredictor(model)

    def set_image(self, rgb):
        self.predictor.set_image(rgb)

    def predict_mask(self, box_xyxy):

        x, y, width, height = box_xyxy
        input_box = np.array([x, y, x + width, y + height])

        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0].astype(bool)

        if mask is not None:
            return mask
