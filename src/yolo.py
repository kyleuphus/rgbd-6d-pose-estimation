import cv2
from ultralytics import YOLO
from pathlib import Path
from ycb_video import load_frame
from collections import defaultdict


def load_yolo():
    model = YOLO("yolo26s-seg.pt", task="segment")
    return model


def segment_image(rgb, model, classes, conf=0.5):  # classes = [yolo class ids]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr, conf=conf, classes=classes, retina_masks=True)

    return results


def pair_masks(results):
    result = results[0]
    pairs = defaultdict(list)  # yolo class_id: [masks]

    if result.boxes is not None and result.masks is not None:
        for i, class_tensor in enumerate(result.boxes.cls):
            class_id = int(class_tensor.item())
            mask = result.masks.data[i].cpu().numpy().astype(bool)
            pairs[class_id].append(mask)

    return pairs
