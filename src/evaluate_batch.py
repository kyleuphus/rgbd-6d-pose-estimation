import csv
import json
from pathlib import Path
from statistics import median

from pose_metrics import calculate_error


FIELDS = [
    "folder_id",
    "image_id",
    "object_id",
    "rotation_error_deg",
    "translation_error_mm",
]


def evaluate_cases(project_root, cases):
    rows = []

    for case in cases:
        folder_id = f"{int(case['folder_id']):06d}"
        image_id = f"{int(case['image_id']):06d}"
        object_id = int(case["object_id"])

        rotation_deg, translation_mm = calculate_error(
            project_root, folder_id, image_id, object_id
        )

        rows.append(
            {
                "folder_id": folder_id,
                "image_id": image_id,
                "object_id": object_id,
                "rotation_error_deg": rotation_deg,
                "translation_error_mm": translation_mm,
            }
        )

    return rows


def write_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_object(rows):
    object_ids = sorted({row["object_id"] for row in rows})

    for object_id in object_ids:
        object_rows = [row for row in rows if row["object_id"] == object_id]

        translation_errors = [row["translation_error_mm"] for row in object_rows]
        rotation_errors = [row["rotation_error_deg"] for row in object_rows]

        print(f"Object {object_id}: {len(object_rows)} cases")
        print(
            f"  Translation: median {median(translation_errors):.2f} mm, "
            f"worst {max(translation_errors):.2f} mm"
        )
        print(
            f"  Rotation: median {median(rotation_errors):.2f}°, "
            f"worst {max(rotation_errors):.2f}°"
        )


def select_inspection_cases(rows):
    object_ids = sorted({row["object_id"] for row in rows})
    selected = []
    seen = set()

    for object_id in object_ids:
        object_rows = [row for row in rows if row["object_id"] == object_id]

        worst_translation = max(
            object_rows,
            key=lambda row: row["translation_error_mm"],
        )
        worst_rotation = max(
            object_rows,
            key=lambda row: row["rotation_error_deg"],
        )

        for case in (worst_translation, worst_rotation):
            case_id = (
                case["folder_id"],
                case["image_id"],
                case["object_id"],
            )

            if case_id not in seen:
                selected.append(case)
                seen.add(case_id)

    return selected


def main():
    project_root = Path(__file__).resolve().parents[1]
    outputs = project_root / "outputs"
    cases = json.loads((outputs / "cases.json").read_text(encoding="utf-8"))

    rows = evaluate_cases(project_root, cases)
    assert len(rows) == len(cases), "Expected one row per case"

    write_csv(rows, outputs / "evaluation.csv")
    summarize_by_object(rows)
    inspection_cases = select_inspection_cases(rows)
    for case in inspection_cases:
        print(
            f"{case['folder_id']}_{case['image_id']} "
            f"object {case['object_id']}: "
            f"{case['translation_error_mm']:.2f} mm, "
            f"{case['rotation_error_deg']:.2f}°"
        )


if __name__ == "__main__":
    main()
