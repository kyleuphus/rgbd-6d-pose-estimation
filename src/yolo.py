import cv2
from ultralytics import YOLO
from pathlib import Path
from ycb_video import load_frame
from collections import defaultdict


def load_yolo():
    model = YOLO("yolo26s-seg.pt", task="segment")
    return model


def segment_image(rgb, model, classes):  # classes = [yolo class ids]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr, conf=0.05, classes=classes, retina_masks=True)

    return results


def pair_masks(results, classes):  # classes = [yolo class ids]
    result = results[0]
    pairs = defaultdict(list)  # yolo class_id: [masks]

    if result.boxes is not None and result.masks is not None:
        for i, class_id in enumerate(result.boxes.cls):
            if int(class_id.item()) in classes.keys():
                mask = result.masks.data[i].cpu().numpy().astype(bool)
                pairs[classes[int(class_id.item())]].append(mask)

    return pairs


# project_root = Path(__file__).resolve().parents[1]
# dataset_root = Path(project_root / "data" / "ycbv")
# folder_id = "000048"
# image_id = "000001"
#
# rgb = load_frame(dataset_root, folder_id, image_id).rgb
# model = load_yolo()
#
# results = segment_image(rgb, model, [41])
# classes = {41: "obj_000014"}
#
# pairs = pair_masks(results, classes)
# results[0].show()
#
# print(len(pairs["obj_000014"]))
