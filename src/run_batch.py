"""Register RGB-D cases with EfficientTAM masks prompted by dataset boxes."""

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np


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


def run_batch(cases, project_root, results_path, output_root=None):
    started = perf_counter()
    cases = normalize_cases(cases)
    project_root, results_path = Path(project_root), Path(results_path).resolve()
    if results_path.exists():
        raise FileExistsError(f"Run already exists: {results_path}. Choose a new run folder.")
    run_dir = results_path.parent
    output_root = (
        Path(output_root).resolve()
        if output_root is not None
        else run_dir
    )
    mask_root = run_dir / "masks"

    def saved_path(path):
        # Paths in a run remain usable after downloading or renaming its folder.
        path = Path(path)
        return str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path)

    write_json(run_dir / "cases.json", cases)
    summary_path = results_path.with_name("timings.json")
    results = []
    summary = {
        "format_version": 1,
        "selected": len(cases),
        "object_setup_seconds": {},
        "mask_source": "efficienttam",
        "box_source": "scene_gt_info.bbox_visib",
        "segmentation_device": "cpu",
        "pose_output_root": saved_path(output_root),
        "image_segmentation_seconds": {},
    }

    def checkpoint():
        summary["successful"] = sum(r["status"] == "ok" for r in results)
        summary["failed"] = sum(r["status"] == "failed" for r in results)
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
    from run_segmentation import construct_model, segment_images

    torch.cuda.synchronize()
    summary["import_seconds"] = perf_counter() - started
    summary["gpu"] = torch.cuda.get_device_name()
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

    print("Loading EfficientTAM once (CPU, dataset box prompts)...", flush=True)
    setup_started = perf_counter()
    segmenter = construct_model(project_root)
    summary["segmentation_setup_seconds"] = perf_counter() - setup_started

    dataset_root = project_root / "data" / "ycbv"

    @lru_cache(maxsize=16)
    def cached_frame(folder_id, image_id):
        return load_frame(dataset_root, folder_id, image_id)

    by_object = defaultdict(list)
    by_image = defaultdict(list)
    for case in cases:
        by_object[case["object_id"]].append(case)
        by_image[(case["folder_id"], case["image_id"])].append(case["object_id"])

    @lru_cache(maxsize=None)
    def cached_masks(folder_id, image_id):
        # Segment all requested objects together so the image encoder runs once.
        segmentation_started = perf_counter()
        try:
            print(f"Segmenting image {folder_id}_{image_id}...", flush=True)
            masks = segment_images(
                segmenter,
                folder_id,
                image_id,
                by_image[(folder_id, image_id)],
                project_root,
                save_mask=True,
                output_dir=mask_root,
            )
            return masks, None
        except Exception as error:
            traceback.print_exc()
            # Cache failures too, rather than retrying the same image per object.
            return {}, f"Image segmentation failed: {type(error).__name__}: {error}"
        finally:
            # Includes image/annotation loading, all prompts, and saving masks.
            summary["image_segmentation_seconds"][f"{folder_id}_{image_id}"] = (
                perf_counter() - segmentation_started
            )

    completed = 0
    for object_id, object_cases in by_object.items():
        backend = None
        setup_error = None
        setup_started = perf_counter()
        print(
            f"Preparing object {object_id} for {len(object_cases)} case(s)...",
            flush=True,
        )
        try:
            mesh = load_mesh(dataset_root, object_id)
            backend = FoundationPoseBackend(
                mesh, scorer=scorer, refiner=refiner, glctx=glctx
            )
            torch.cuda.synchronize()
        except Exception as error:
            setup_error = f"Object setup failed: {type(error).__name__}: {error}"
            traceback.print_exc()
        summary["object_setup_seconds"][str(object_id)] = perf_counter() - setup_started
        print(
            f"Object setup: {summary['object_setup_seconds'][str(object_id)]:.2f}s",
            flush=True,
        )
        registrations = 0

        for case in object_cases:
            completed += 1
            case_started = perf_counter()
            name = f"{case['folder_id']}_{case['image_id']}_obj_{object_id:06d}"
            print(f"[{completed}/{len(cases)}] {name}", flush=True)
            result = {
                **case,
                "status": "failed",
                "pose_path": None,
                "mask_path": None,
                "mask_source": "efficienttam",
                "box_source": "scene_gt_info.bbox_visib",
            }
            try:
                if setup_error is not None:
                    raise RuntimeError(setup_error)
                load_started = perf_counter()
                frame = cached_frame(case["folder_id"], case["image_id"])
                result["load_seconds"] = perf_counter() - load_started
                masks, segmentation_error = cached_masks(
                    case["folder_id"], case["image_id"]
                )
                result["image_segmentation_seconds"] = summary[
                    "image_segmentation_seconds"
                ][f"{case['folder_id']}_{case['image_id']}"]
                if segmentation_error is not None:
                    raise RuntimeError(segmentation_error)
                mask = masks[object_id]
                if (
                    mask.shape != frame.depth.shape
                    or mask.dtype != np.bool_
                    or not mask.any()
                ):
                    raise ValueError("Expected a nonempty boolean mask matching depth")
                mask_path = mask_root / f"{name}_mask.npy"
                result["mask_path"] = saved_path(mask_path)

                # CUDA work is asynchronous: synchronize around the timed registration.
                torch.cuda.synchronize()
                registration_started = perf_counter()
                pose = backend.estimate(frame, mask)
                torch.cuda.synchronize()
                result["registration_seconds"] = perf_counter() - registration_started
                result["first_for_object"] = registrations == 0
                registrations += 1
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError("Estimator returned an invalid pose matrix")

                save_started = perf_counter()
                pose_path = output_root / name / f"pose_{name}.npy"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(pose_path, pose)
                result.update(status="ok", pose_path=saved_path(pose_path))
                result["save_seconds"] = perf_counter() - save_started
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
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

        # Retain the shared networks/context, but release this object's estimator.
        backend = None

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
    parser.add_argument("--cases-file", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--run-dir", type=Path, help="New folder for this run's outputs")
    destination.add_argument("--results-file", type=Path, help="Legacy alternative to --run-dir")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the pose directory (default: run directory)",
    )
    args = parser.parse_args()
    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))
    results_path = args.run_dir / "results.json" if args.run_dir is not None else args.results_file
    results = run_batch(cases, args.project_root, results_path, args.output_root)
    if not any(result["status"] == "ok" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
