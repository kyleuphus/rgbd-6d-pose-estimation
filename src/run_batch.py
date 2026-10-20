"""Register independent RGB-D cases while sharing models and GPU resources."""

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
            raise ValueError("Folder/image IDs must be nonnegative; object IDs must be positive")
        case = {"folder_id": f"{folder:06d}", "image_id": f"{image:06d}", "object_id": obj}
        if case not in normalized:
            normalized.append(case)
    if not normalized:
        raise ValueError("Select at least one case")
    return normalized


def run_batch(cases, project_root, results_path):
    started = perf_counter()
    cases = normalize_cases(cases)
    project_root, results_path = Path(project_root), Path(results_path)
    summary_path = results_path.with_name("timings.json")
    results = []
    summary = {"selected": len(cases), "object_setup_seconds": {}}

    def checkpoint():
        summary["successful"] = sum(r["status"] == "ok" for r in results)
        summary["failed"] = sum(r["status"] == "failed" for r in results)
        summary["elapsed_seconds"] = perf_counter() - started
        times = [r["registration_seconds"] for r in results if r["status"] == "ok"]
        warmed = [r["registration_seconds"] for r in results
                  if r["status"] == "ok" and not r["first_for_object"]]
        summary["mean_registration_seconds"] = sum(times) / len(times) if times else None
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
    from foundation_pose_backend import FoundationPoseBackend, ScorePredictor, PoseRefinePredictor
    from ycb_video import load_frame, load_mesh, load_visible_mask

    torch.cuda.synchronize()
    summary["import_seconds"] = perf_counter() - started
    summary["gpu"] = torch.cuda.get_device_name()
    print(f"GPU: {summary['gpu']}. Loading scorer/refiner and renderer once...", flush=True)
    setup_started = perf_counter()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    torch.cuda.synchronize()
    summary["model_setup_seconds"] = perf_counter() - setup_started
    print(f"Shared model/renderer setup: {summary['model_setup_seconds']:.2f}s", flush=True)

    dataset_root = project_root / "data" / "ycbv"

    @lru_cache(maxsize=16)
    def cached_frame(folder_id, image_id):
        return load_frame(dataset_root, folder_id, image_id)

    by_object = defaultdict(list)
    for case in cases:
        by_object[case["object_id"]].append(case)

    completed = 0
    for object_id, object_cases in by_object.items():
        backend = None
        setup_error = None
        setup_started = perf_counter()
        print(f"Preparing object {object_id} for {len(object_cases)} case(s)...", flush=True)
        try:
            mesh = load_mesh(dataset_root, object_id)
            backend = FoundationPoseBackend(mesh, scorer=scorer, refiner=refiner, glctx=glctx)
            torch.cuda.synchronize()
        except Exception as error:
            setup_error = f"Object setup failed: {type(error).__name__}: {error}"
            traceback.print_exc()
        summary["object_setup_seconds"][str(object_id)] = perf_counter() - setup_started
        print(f"Object setup: {summary['object_setup_seconds'][str(object_id)]:.2f}s", flush=True)
        registrations = 0

        for case in object_cases:
            completed += 1
            case_started = perf_counter()
            name = f"{case['folder_id']}_{case['image_id']}_obj_{object_id:06d}"
            print(f"[{completed}/{len(cases)}] {name}", flush=True)
            result = {**case, "status": "failed", "pose_path": None}
            try:
                if setup_error is not None:
                    raise RuntimeError(setup_error)
                load_started = perf_counter()
                frame = cached_frame(case["folder_id"], case["image_id"])
                mask = load_visible_mask(dataset_root, case["folder_id"], case["image_id"], object_id)
                result["load_seconds"] = perf_counter() - load_started

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
                pose_path = project_root / "outputs" / name / f"pose_{name}.npy"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(pose_path, pose)
                result.update(status="ok", pose_path=str(pose_path))
                result["save_seconds"] = perf_counter() - save_started
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()

            result["case_seconds"] = perf_counter() - case_started
            results.append(result)
            checkpoint()
            if result["status"] == "ok":
                note = " (first registration for object; may include warm-up)" if result["first_for_object"] else ""
                print(f"  ok: load {result['load_seconds']:.2f}s, "
                      f"register {result['registration_seconds']:.2f}s, "
                      f"save {result['save_seconds']:.2f}s{note}", flush=True)
            else:
                print(f"  failed: {result['error']}", flush=True)

        # Retain the shared networks/context, but release this object's estimator.
        backend = None

    checkpoint()
    print(f"Finished: {summary['successful']}/{len(cases)} succeeded.", flush=True)
    if summary["mean_registration_seconds"] is not None:
        print(f"Mean registration: {summary['mean_registration_seconds']:.3f}s", flush=True)
    if summary["mean_registration_after_first_per_object_seconds"] is not None:
        print("Mean registration after first per object: "
              f"{summary['mean_registration_after_first_per_object_seconds']:.3f}s", flush=True)
    print(f"Results: {results_path}\nTimings: {summary_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))
    results = run_batch(cases, args.project_root, args.results_file)
    if not any(result["status"] == "ok" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
