import nvdiffrast.torch as dr
import numpy as np
import json
import cv2
import argparse
from itertools import islice
import trimesh

from visualization import overlay_pose_axes
from yolo import load_yolo, segment_image, pair_masks
from foundation_pose_backend import FoundationPoseBackend
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from pathlib import Path
from mustard_reader import iter_mustard_frames


def run_sequence(meshes, object_id_to_class_id, conf, run_dir, *, max_frames=None):
    project_root = Path(__file__).resolve().parents[1]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    mask_dir = run_dir / "masks"
    mask_dir.mkdir()

    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir()

    results_path = run_dir / "results.json"
    results_path.write_text("[]\n", encoding="utf-8")

    segmenter = load_yolo()

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    backends = {
        instance_id: FoundationPoseBackend(
            mesh, scorer=scorer, refiner=refiner, glctx=glctx
        )
        for instance_id, mesh in meshes.items()
    }

    records = []
    i = 0

    frames = iter_mustard_frames()
    if max_frames is not None:
        frames = islice(frames, max_frames)

    for frame_id, frame in frames:
        vis = frame.rgb.copy()

        searching_ids = [
            instance_id
            for instance_id, backend in backends.items()
            if not backend.is_registered()
        ]

        mask_pairs = {}

        if searching_ids:
            classes = sorted(
                {object_id_to_class_id[instance_id] for instance_id in searching_ids}
            )

            results = segment_image(frame.rgb, segmenter, classes, conf)
            mask_pairs = pair_masks(results)

        for instance_id, backend in backends.items():
            record = {
                "frame_id": frame_id,
                "frame_index": i,
                "instance_id": instance_id,
                "action": "search",
                "status": None,
                "detection_count": None,
                "pose": None,
                "error": None,
                "mask_path": None,
            }
            try:
                if backend.is_registered():
                    record["action"] = "tracking"
                    pose = backend.track(frame)
                    if pose.shape != (4, 4) or not np.isfinite(pose).all():
                        raise ValueError("Estimator returned an invalid pose matrix")

                else:
                    class_id = object_id_to_class_id[instance_id]
                    masks = mask_pairs.get(class_id, [])
                    record["detection_count"] = len(masks)

                    if len(masks) != 1:
                        record["status"] = (
                            "not detected" if len(masks) == 0 else "ambiguous detection"
                        )
                        records.append(record)

                        print(
                            f"{frame_id}, instance {instance_id}: "
                            f"expected one mask, found {len(masks)}"
                        )
                        continue

                    record["action"] = "registration"

                    mask_path = mask_dir / f"{frame_id}_instance_{instance_id}.npy"
                    np.save(mask_path, masks[0])
                    record["mask_path"] = str(mask_path.relative_to(run_dir))

                    pose = backend.estimate(frame, masks[0])
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError("Estimator returned an invalid pose matrix")

                record["status"] = "pose_returned"
                record["pose"] = pose.tolist()
            except (RuntimeError, ValueError) as error:
                backend.reset()
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"

                print(
                    f"{frame_id}, instance {instance_id}",
                    f"{record['action']} failed: {error}",
                )

            records.append(record)

        for record in records[-len(backends) :]:
            if record["pose"] is not None:
                vis = overlay_pose_axes(
                    vis, frame.intrinsics, np.asarray(record["pose"]), axis_length=0.05
                )

        temporary_path = results_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(results_path)

        vis_path = vis_dir / f"{i:06d}_{frame_id}.png"
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(vis_path), vis_bgr):
            raise OSError(f"Could not save visualization: {vis_path}")

        i += 1

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")

    project_root = Path(__file__).resolve().parents[1]
    mesh_path = project_root / "data" / "mustard0" / "mesh" / "textured_simple.obj"
    mesh = trimesh.load(str(mesh_path), force="mesh")

    records = run_sequence(
        meshes={5: mesh},
        object_id_to_class_id={5: 39},
        conf=args.confidence,
        run_dir=args.run_dir,
        max_frames=args.max_frames,
    )

    successful = sum(r["status"] == "pose_returned" for r in records)
    print(f"Finished: {successful}/{len(records)} records returned poses.")
    print(f"Outputs: {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()
