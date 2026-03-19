import json
import csv
import shutil
from pathlib import Path


images_dir = Path("chartinfo_train/images/line")
ann_dir = Path("chartinfo_train/annotations_JSON/line")

out_dir = Path("chartinfo_lineex_new")
(out_dir / "images").mkdir(parents=True, exist_ok=True)
(out_dir / "labels").mkdir(exist_ok=True)

manifest_path = out_dir / "manifest.jsonl"

converted = 0
removed = 0
skipped = 0


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"⚠ could not delete {path}: {e}")


def get_nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


with open(manifest_path, "w", encoding="utf-8") as manifest:
    for ann_file in ann_dir.glob("*.json"):
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            print(f"⚠ skipping invalid json {ann_file}: {e}")
            skipped += 1
            continue

        chart_type = get_nested(ann, "task1", "output", "chart_type", default=None)

        stem = ann_file.stem

        candidate_images = [
            images_dir / f"{stem}.png",
            images_dir / f"{stem}.jpg",
            images_dir / f"{stem}.jpeg",
            images_dir / f"{stem}.webp",
        ]

        image_path = None
        for p in candidate_images:
            if p.exists():
                image_path = p
                break

        if image_path is None:
            print(f"⚠ skipping {ann_file}, no matching image found for stem '{stem}'")
            skipped += 1
            continue

        image_name = image_path.name

        # remove non-line charts
        if chart_type != "line":
            safe_unlink(ann_file)
            safe_unlink(image_path)
            removed += 1
            continue

        data_series = get_nested(ann, "task6", "output", "data series", default=[])

        if not data_series or not isinstance(data_series, list):
            print(f"⚠ skipping {ann_file}, no task6/output/data series found")
            skipped += 1
            continue

        # keep only SINGLE-line charts
        if len(data_series) != 1:
            print(f"⚠ removing {ann_file}, expected exactly 1 line, got {len(data_series)}")
            safe_unlink(ann_file)
            safe_unlink(image_path)
            removed += 1
            continue

        plot_bb = get_nested(ann, "task4", "output", "_plot_bb", default=None)

        image_dst = out_dir / "images" / image_name
        csv_path = out_dir / "labels" / f"{stem}.csv"

        try:
            shutil.copy(image_path, image_dst)
        except Exception as e:
            print(f"⚠ could not copy image {image_path}: {e}")
            skipped += 1
            continue

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "line_id"])

                for line_id, series in enumerate(data_series):
                    if not isinstance(series, dict):
                        continue
                    pts = series.get("data", [])
                    if not isinstance(pts, list):
                        continue

                    for pt in pts:
                        if not isinstance(pt, dict):
                            continue
                        x = pt.get("x", None)
                        y = pt.get("y", None)
                        if x is None or y is None:
                            continue
                        writer.writerow([x, y, line_id])
        except Exception as e:
            print(f"⚠ could not write csv for {ann_file}: {e}")
            skipped += 1
            continue

        manifest_record = {
            "id": stem,
            "full": f"images/{image_name}",
            "masked": None,
            "csv": f"labels/{stem}.csv",
            "n_lines": len(data_series),
        }

        if isinstance(plot_bb, dict):
            manifest_record["plot_bb"] = plot_bb

        manifest.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
        converted += 1


print("--------------------------------------------------")
print("Conversion finished")
print("Single-line charts kept:", converted)
print("Charts removed:", removed)
print("Skipped:", skipped)
print("Manifest:", manifest_path)