"""Register independent RGB-D cases with YOLO or dataset-box-prompted EfficientTAM.

For the banana/mustard baseline, use --yolo-baseline --run-dir <new folder>.
Use --preview to inspect the selection without loading models or using CUDA.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np


YOLO_OBJECT_CLASSES = {5: 39, 10: 46}
YOLO_BASELINE_SCENES = {50: (5, 10), 52: (5,), 56: (10,)}


class CaseFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_cases(cases):
    normalized = []
    for case in cases:
        folder = int(case["folder_id"])
        image = int(case["image_id"])
        obj = int(case["object_id"])
        if folder < 0 or image < 0 or obj <= 0:
            raise ValueError(
                "Folder/image IDs must be nonnegative; object IDs must be positive"
            )
        case = {
            "folder_id": f"{folder:06d}",
            "image_id": f"{image:06d}",
            "object_id": obj,
        }
        if case not in normalized:
            normalized.append(case)
    if not normalized:
        raise ValueError("Select at least one case")
    return normalized


def baseline_cases(project_root):
    """Select every available RGB-D pair, with scene-specific mesh identities.

    Scene IDs identify the controlled experimental targets. YOLO never receives
    reference boxes, masks, or poses, including while choosing its detections.
    """
    cases = []
    for folder_id, object_ids in YOLO_BASELINE_SCENES.items():
        scene = Path(project_root) / "data/ycbv/test" / f"{folder_id:06d}"
        images = sorted((scene / "rgb").glob("*.png"), key=lambda p: int(p.stem))
        if not images:
            raise FileNotFoundError(f"No RGB images in {scene / 'rgb'}")
        for image in images:
            if not (scene / "depth" / image.name).is_file():
                raise FileNotFoundError(
                    f"Missing paired depth: {scene / 'depth' / image.name}"
                )
            cases.extend(
                {
                    "folder_id": folder_id,
                    "image_id": int(image.stem),
                    "object_id": object_id,
                }
                for object_id in object_ids
            )
    return normalize_cases(cases)


def validate_yolo_mapping(cases, mapping):
    mapping = {int(k): int(v) for k, v in mapping.items()}
    if any(k <= 0 or v < 0 for k, v in mapping.items()):
        raise ValueError("Use positive YCB IDs and nonnegative YOLO class IDs")
    missing = {case["object_id"] for case in cases} - mapping.keys()
    if missing:
        raise ValueError(
            f"Missing explicit YOLO class mapping for YCB objects {sorted(missing)}"
        )
    by_image = defaultdict(list)
    for case in cases:
        by_image[(case["folder_id"], case["image_id"])].append(
            mapping[case["object_id"]]
        )
    for image, classes in by_image.items():
        if len(classes) != len(set(classes)):
            raise ValueError(
                f"{image}: configure only one mesh per YOLO class per image"
            )
    return mapping


def yolo_detections(model, frame, classes, confidence, device, imgsz):
    """One inference for the entire frame; preserve each candidate's metadata."""
    import cv2

    result = model.predict(
        cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
        classes=classes,
        conf=confidence,
        retina_masks=True,
        device=device,
        imgsz=imgsz,
        verbose=False,
    )[0]
    candidates = defaultdict(list)
    if result.boxes is None or len(result.boxes) == 0:
        return candidates
    if result.masks is None:
        raise ValueError("Segmentation model returned boxes without masks")
    masks = result.masks.data.cpu().numpy().astype(bool)
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(masks) != len(class_ids) or masks.shape[1:] != frame.depth.shape:
        raise ValueError("YOLO masks do not match detections or RGB-D dimensions")
    for index, class_id in enumerate(class_ids):
        candidates[int(class_id)].append(
            {
                "mask": masks[index],
                "detection_index": index,
                "confidence": float(confidences[index]),
                "bbox_xyxy": boxes[index].tolist(),
            }
        )
    return candidates


def run_batch(
    cases,
    project_root,
    results_path,
    output_root=None,
    *,
    mask_source="efficienttam",
    object_id_to_class_id=None,
    yolo_weights="yolo26s-seg.pt",
    confidence=0.5,
    segmentation_device=None,
    imgsz=640,
    threshold_study_summary=None,
):
    """Initialize once, then load/segment each image and register its targets.

    Every requested object/frame case receives a result, including detection
    failures. Backends are reused for independent registration, not tracking.
    """
    started = perf_counter()
    cases = normalize_cases(cases)
    if mask_source not in ("yolo", "efficienttam"):
        raise ValueError("mask_source must be yolo or efficienttam")
    if not 0 < confidence <= 1 or imgsz <= 0:
        raise ValueError("Use 0 < confidence <= 1 and a positive image size")
    mapping = (
        validate_yolo_mapping(
            cases,
            YOLO_OBJECT_CLASSES
            if object_id_to_class_id is None
            else object_id_to_class_id,
        )
        if mask_source == "yolo"
        else {}
    )
    study_frames = {}
    if threshold_study_summary is not None:
        study = json.loads(Path(threshold_study_summary).read_text(encoding="utf-8"))
        study_frames = {
            (f"{int(f['scene']):06d}", f"{int(f['image']):06d}"): f["split"]
            for f in study["frames"]
        }
    project_root, results_path = Path(project_root), Path(results_path).resolve()
    if results_path.exists():
        raise FileExistsError(
            f"Run already exists: {results_path}. Choose a new run folder."
        )
    run_dir = results_path.parent
    output_root = Path(output_root).resolve() if output_root is not None else run_dir
    mask_root = run_dir / "masks"

    def saved_path(path):
        # Paths in a run remain usable after downloading or renaming its folder.
        path = Path(path)
        return (
            str(path.relative_to(run_dir))
            if path.is_relative_to(run_dir)
            else str(path)
        )

    write_json(run_dir / "cases.json", cases)
    summary_path = results_path.with_name("timings.json")
    results = []
    by_image = defaultdict(list)
    for case in cases:
        by_image[(case["folder_id"], case["image_id"])].append(case)
    device = segmentation_device or ("cuda:0" if mask_source == "yolo" else "cpu")
    box_source = "yolo" if mask_source == "yolo" else "scene_gt_info.bbox_visib"
    summary = {
        "format_version": 1,
        "selected": len(cases),
        "object_setup_seconds": {},
        "selected_images": len(by_image),
        "mask_source": mask_source,
        "box_source": box_source,
        "segmentation_device": device,
        "pose_mode": "independent_registration",
        "pose_output_root": saved_path(output_root),
        "image_segmentation_seconds": {},
        "image_load_seconds": {},
        "image_seconds": {},
    }
    if mask_source == "yolo":
        summary.update(
            yolo_weights=str(yolo_weights),
            confidence_threshold=confidence,
            imgsz=imgsz,
            object_id_to_class_id=mapping,
            instance_policy="exactly_one_detection_per_requested_class",
        )
    if threshold_study_summary is not None:
        summary["threshold_study_summary"] = str(threshold_study_summary)

    def checkpoint():
        summary["successful"] = sum(r["status"] == "ok" for r in results)
        summary["failed"] = sum(r["status"] == "failed" for r in results)
        summary["coverage"] = summary["successful"] / len(cases)
        summary["completed"] = len(results)
        summary["remaining"] = len(cases) - len(results)
        summary["failure_counts"] = dict(
            Counter(r["failure_reason"] for r in results if r["status"] == "failed")
        )
        summary["successful_by_object"] = dict(
            Counter(str(r["object_id"]) for r in results if r["status"] == "ok")
        )
        summary["selected_by_object"] = dict(
            Counter(str(c["object_id"]) for c in cases)
        )
        if threshold_study_summary is not None:
            summary["threshold_study_groups"] = {
                group: {
                    "selected": sum(
                        r["threshold_study_split"] == group for r in results
                    ),
                    "successful": sum(
                        r["threshold_study_split"] == group and r["status"] == "ok"
                        for r in results
                    ),
                }
                for group in ("calibration", "check", "not_sampled")
            }
        summary["elapsed_seconds"] = perf_counter() - started
        times = [r["registration_seconds"] for r in results if r["status"] == "ok"]
        warmed = [
            r["registration_seconds"]
            for r in results
            if r["status"] == "ok" and not r["first_for_object"]
        ]
        summary["mean_registration_seconds"] = (
            sum(times) / len(times) if times else None
        )
        summary["mean_registration_after_first_per_object_seconds"] = (
            sum(warmed) / len(warmed) if warmed else None
        )
        write_json(results_path, results)
        write_json(summary_path, summary)

    checkpoint()
    print("Loading GPU libraries (once for this batch)...", flush=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab")
    import nvdiffrast.torch as dr
    from foundation_pose_backend import (
        FoundationPoseBackend,
        ScorePredictor,
        PoseRefinePredictor,
    )
    from ycb_video import load_frame, load_mesh

    torch.cuda.synchronize()
    summary["import_seconds"] = perf_counter() - started
    summary["gpu"] = torch.cuda.get_device_name()
    summary["torch_version"] = getattr(torch, "__version__", None)
    summary["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    print(
        f"GPU: {summary['gpu']}. Loading scorer/refiner and renderer once...",
        flush=True,
    )
    setup_started = perf_counter()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    torch.cuda.synchronize()
    summary["model_setup_seconds"] = perf_counter() - setup_started
    print(
        f"Shared model/renderer setup: {summary['model_setup_seconds']:.2f}s",
        flush=True,
    )

    setup_started = perf_counter()
    if mask_source == "yolo":
        from ultralytics import YOLO, __version__ as ultralytics_version

        print(
            f"Loading YOLO once: {yolo_weights}, confidence {confidence}...", flush=True
        )
        segmenter = YOLO(str(yolo_weights), task="segment")
        for class_id in set(mapping.values()):
            if class_id not in segmenter.names:
                raise ValueError(f"YOLO checkpoint does not contain class {class_id}")
        summary["ultralytics_version"] = ultralytics_version
        summary["yolo_class_names"] = {
            str(c): segmenter.names[c] for c in set(mapping.values())
        }
    else:
        if device != "cpu":
            raise ValueError("The existing EfficientTAM constructor uses CPU")
        from run_segmentation import construct_model, segment_images

        print("Loading EfficientTAM once (CPU, dataset box prompts)...", flush=True)
        segmenter = construct_model(project_root)
    torch.cuda.synchronize()
    summary["segmentation_setup_seconds"] = perf_counter() - setup_started

    dataset_root = project_root / "data" / "ycbv"

    backends, setup_errors = {}, {}
    for object_id in sorted({c["object_id"] for c in cases}):
        setup_started = perf_counter()
        print(
            f"Preparing object {object_id} (once)...",
            flush=True,
        )
        try:
            mesh = load_mesh(dataset_root, object_id)
            backends[object_id] = FoundationPoseBackend(
                mesh, scorer=scorer, refiner=refiner, glctx=glctx
            )
            torch.cuda.synchronize()
        except Exception as error:
            setup_errors[object_id] = (
                f"Object setup failed: {type(error).__name__}: {error}"
            )
            traceback.print_exc()
        summary["object_setup_seconds"][str(object_id)] = perf_counter() - setup_started
        print(
            f"Object setup: {summary['object_setup_seconds'][str(object_id)]:.2f}s",
            flush=True,
        )
    checkpoint()
    registrations = Counter()
    completed = 0
    for image_index, ((folder_id, image_id), image_cases) in enumerate(
        sorted(by_image.items()), 1
    ):
        image_started = perf_counter()
        image_name = f"{folder_id}_{image_id}"
        object_ids = [c["object_id"] for c in image_cases]
        print(
            f"Image [{image_index}/{len(by_image)}] {image_name}, objects {object_ids}",
            flush=True,
        )
        frame = None
        candidates, masks = {}, {}
        image_error = None
        load_started = perf_counter()
        try:
            frame = load_frame(dataset_root, folder_id, image_id)
        except Exception as error:
            image_error = ("frame_load_failed", f"{type(error).__name__}: {error}")
            traceback.print_exc()
        load_seconds = perf_counter() - load_started
        summary["image_load_seconds"][image_name] = load_seconds
        segmentation_seconds = 0.0
        if frame is not None:
            torch.cuda.synchronize()
            segmentation_started = perf_counter()
            try:
                if mask_source == "yolo":
                    classes = sorted({mapping[o] for o in object_ids})
                    candidates = yolo_detections(
                        segmenter, frame, classes, confidence, device, imgsz
                    )
                else:
                    masks = segment_images(
                        segmenter,
                        folder_id,
                        image_id,
                        object_ids,
                        project_root,
                        save_mask=False,
                    )
                torch.cuda.synchronize()
            except Exception as error:
                image_error = (
                    "segmentation_failed",
                    f"{type(error).__name__}: {error}",
                )
                traceback.print_exc()
            finally:
                segmentation_seconds = perf_counter() - segmentation_started
        summary["image_segmentation_seconds"][image_name] = segmentation_seconds

        for case in image_cases:
            object_id = case["object_id"]
            completed += 1
            case_started = perf_counter()
            name = f"{case['folder_id']}_{case['image_id']}_obj_{object_id:06d}"
            print(f"[{completed}/{len(cases)}] {name}", flush=True)
            result = {
                **case,
                "status": "failed",
                "pose_path": None,
                "mask_path": None,
                "mask_source": mask_source,
                "box_source": box_source,
                "load_seconds": load_seconds,
                "image_segmentation_seconds": segmentation_seconds,
            }
            if threshold_study_summary is not None:
                result["threshold_study_split"] = study_frames.get(
                    (folder_id, image_id), "not_sampled"
                )
            failure_reason = "registration_failed"
            try:
                if mask_source == "yolo":
                    detections = candidates.get(mapping[object_id], [])
                    result.update(
                        yolo_class_id=mapping[object_id],
                        confidence_threshold=confidence,
                        detection_count=len(detections)
                        if image_error is None
                        else None,
                        candidate_detections=[
                            {k: v for k, v in d.items() if k != "mask"}
                            for d in detections
                        ],
                    )
                if object_id in setup_errors:
                    raise CaseFailure("object_setup_failed", setup_errors[object_id])
                if image_error is not None:
                    raise CaseFailure(*image_error)
                if mask_source == "yolo":
                    if not detections:
                        raise CaseFailure(
                            "not_detected",
                            f"No YOLO class {mapping[object_id]} detection",
                        )
                    if len(detections) != 1:
                        raise CaseFailure(
                            "ambiguous_detection",
                            f"Expected one detection, found {len(detections)}",
                        )
                    mask = detections[0]["mask"]
                    result["confidence"] = detections[0]["confidence"]
                    result["bbox_xyxy"] = detections[0]["bbox_xyxy"]
                else:
                    if object_id not in masks:
                        raise CaseFailure(
                            "not_detected", "No segmentation mask for requested object"
                        )
                    mask = masks[object_id]
                if (
                    mask.shape != frame.depth.shape
                    or mask.dtype != np.bool_
                    or not mask.any()
                ):
                    raise CaseFailure(
                        "invalid_mask",
                        "Expected a nonempty boolean mask matching depth",
                    )
                failure_reason = "save_failed"
                mask_root.mkdir(parents=True, exist_ok=True)
                mask_path = mask_root / f"{name}_mask.npy"
                np.save(mask_path, mask)
                result["mask_path"] = saved_path(mask_path)
                valid_depth = mask & np.isfinite(frame.depth) & (frame.depth >= 0.001)
                result["valid_depth_pixels"] = int(valid_depth.sum())
                result["mask_pixels"] = int(mask.sum())
                if valid_depth.sum() < 4:
                    raise CaseFailure(
                        "insufficient_depth",
                        "Fewer than four valid depth pixels in mask",
                    )

                # CUDA work is asynchronous: synchronize around the timed registration.
                failure_reason = "registration_failed"
                torch.cuda.synchronize()
                registration_started = perf_counter()
                result["first_for_object"] = registrations[object_id] == 0
                registrations[object_id] += 1
                try:
                    pose = backends[object_id].estimate(frame, mask)
                finally:
                    torch.cuda.synchronize()
                    result["registration_seconds"] = (
                        perf_counter() - registration_started
                    )
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise CaseFailure(
                        "invalid_pose", "Estimator returned an invalid pose matrix"
                    )

                failure_reason = "save_failed"
                save_started = perf_counter()
                pose_path = output_root / name / f"pose_{name}.npy"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(pose_path, pose)
                result.update(status="ok", pose_path=saved_path(pose_path))
                result["save_seconds"] = perf_counter() - save_started
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                result["failure_reason"] = (
                    error.code if isinstance(error, CaseFailure) else failure_reason
                )
                if not isinstance(error, CaseFailure):
                    traceback.print_exc()

            result["case_seconds"] = perf_counter() - case_started
            results.append(result)
            checkpoint()
            if result["status"] == "ok":
                note = (
                    " (first registration for object; may include warm-up)"
                    if result["first_for_object"]
                    else ""
                )
                print(
                    f"  ok: load {result['load_seconds']:.2f}s, "
                    f"register {result['registration_seconds']:.2f}s, "
                    f"save {result['save_seconds']:.2f}s{note}",
                    flush=True,
                )
            else:
                print(f"  failed: {result['error']}", flush=True)

        summary["image_seconds"][image_name] = perf_counter() - image_started
        checkpoint()

    checkpoint()
    print(f"Finished: {summary['successful']}/{len(cases)} succeeded.", flush=True)
    if summary["mean_registration_seconds"] is not None:
        print(
            f"Mean registration: {summary['mean_registration_seconds']:.3f}s",
            flush=True,
        )
    if summary["mean_registration_after_first_per_object_seconds"] is not None:
        print(
            "Mean registration after first per object: "
            f"{summary['mean_registration_after_first_per_object_seconds']:.3f}s",
            flush=True,
        )
    print(f"Results: {results_path}\nTimings: {summary_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cases-file", type=Path)
    selection.add_argument(
        "--yolo-baseline",
        action="store_true",
        help="All RGB-D frames in scenes 50 (objects 5,10), 52 (5), 56 (10)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Print selection without GPU/model setup"
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--run-dir", type=Path, help="New folder for this run's outputs"
    )
    destination.add_argument(
        "--results-file", type=Path, help="Legacy alternative to --run-dir"
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the pose directory (default: run directory)",
    )
    parser.add_argument(
        "--mask-source",
        choices=("yolo", "efficienttam"),
        default=None,
        help="Defaults to YOLO for --yolo-baseline; EfficientTAM for --cases-file",
    )
    parser.add_argument(
        "--object-class-map",
        type=Path,
        help='JSON mapping YCB IDs to YOLO IDs; default {"5":39,"10":46}',
    )
    parser.add_argument("--yolo-weights", default="yolo26s-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--segmentation-device",
        help="YOLO device (default cuda:0); EfficientTAM uses CPU",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--threshold-study-summary",
        type=Path,
        help="Optional scratch summary.json for tagging previously sampled frames",
    )
    args = parser.parse_args()
    cases = (
        baseline_cases(args.project_root)
        if args.yolo_baseline
        else normalize_cases(json.loads(args.cases_file.read_text(encoding="utf-8")))
    )
    mask_source = args.mask_source or ("yolo" if args.yolo_baseline else "efficienttam")
    mapping = (
        json.loads(args.object_class_map.read_text(encoding="utf-8"))
        if args.object_class_map
        else YOLO_OBJECT_CLASSES
    )
    if mask_source == "yolo":
        validate_yolo_mapping(cases, mapping)
    print(
        f"Selected {len(cases)} object/frame cases across "
        f"{len({(c['folder_id'], c['image_id']) for c in cases})} images; {mask_source}.",
        flush=True,
    )
    for folder in sorted({c["folder_id"] for c in cases}):
        selected = [c for c in cases if c["folder_id"] == folder]
        print(
            f"  Scene {folder}: {len({c['image_id'] for c in selected})} images, "
            f"{len(selected)} cases, objects {sorted({c['object_id'] for c in selected})}",
            flush=True,
        )
    if args.preview:
        return
    if args.run_dir is None and args.results_file is None:
        parser.error("Provide --run-dir or --results-file for inference")
    results_path = (
        args.run_dir / "results.json" if args.run_dir is not None else args.results_file
    )
    results = run_batch(
        cases,
        args.project_root,
        results_path,
        args.output_root,
        mask_source=mask_source,
        object_id_to_class_id=mapping,
        yolo_weights=args.yolo_weights,
        confidence=args.confidence,
        segmentation_device=args.segmentation_device,
        imgsz=args.imgsz,
        threshold_study_summary=args.threshold_study_summary,
    )
    if not any(result["status"] == "ok" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
