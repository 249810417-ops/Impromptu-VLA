import argparse
import io
import json
import tarfile
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract only the images referenced by nuscenes_test.json from the WebDataset shards."
    )
    parser.add_argument(
        "--dataset_json",
        type=Path,
        default=Path("nuscenes_test.json"),
        help="Path to the nuscenes test json file.",
    )
    parser.add_argument(
        "--tar_root",
        type=Path,
        required=True,
        help="Directory containing the downloaded unstructed_nuScenes *.tar shards.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Directory to reconstruct the nuscenes image tree under.",
    )
    return parser.parse_args()


def load_required_images(dataset_json: Path) -> set[str]:
    with open(dataset_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    required_images = set()
    for sample in dataset:
        for image_path in sample.get("images", []):
            required_images.add(image_path)

    return required_images


def parse_original_paths(text: str) -> dict[str, str]:
    mapping = {}
    for line in text.strip().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def reconstruct_relative_path(original_abs_path: str) -> str | None:
    marker = "/nuscenes/"
    pos = original_abs_path.find(marker)
    if pos < 0:
        return None

    return original_abs_path[pos + 1 :]


def save_as_jpeg(image_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(output_path, format="JPEG", quality=95)


def main():
    args = parse_args()
    required_images = load_required_images(args.dataset_json)
    remaining_images = set(required_images)

    print(f"Need to extract {len(required_images)} images from {args.dataset_json}.")

    written = 0
    scanned_samples = 0

    for tar_path in sorted(args.tar_root.glob("*.tar")):
        if not remaining_images:
            break

        print(f"Processing {tar_path.name}")
        with tarfile.open(tar_path, "r") as tar:
            members = {member.name: member for member in tar.getmembers() if member.isfile()}
            prefixes = sorted(
                {
                    name[: -len(".original_image_paths.txt")]
                    for name in members
                    if name.endswith(".original_image_paths.txt")
                }
            )

            for prefix in prefixes:
                if not remaining_images:
                    break

                scanned_samples += 1
                original_paths_name = f"{prefix}.original_image_paths.txt"
                cam_front_name = f"{prefix}.CAM_FRONT.png"

                if original_paths_name not in members or cam_front_name not in members:
                    continue

                original_paths_text = tar.extractfile(members[original_paths_name]).read().decode(
                    "utf-8", errors="ignore"
                )
                path_mapping = parse_original_paths(original_paths_text)
                original_abs_path = path_mapping.get(cam_front_name)
                if not original_abs_path:
                    continue

                relative_path = reconstruct_relative_path(original_abs_path)
                if not relative_path or relative_path not in remaining_images:
                    continue

                image_bytes = tar.extractfile(members[cam_front_name]).read()
                output_path = args.output_root / relative_path
                save_as_jpeg(image_bytes, output_path)

                remaining_images.remove(relative_path)
                written += 1

        print(
            f"Scanned {scanned_samples} samples so far, wrote {written} images, remaining {len(remaining_images)}."
        )

    print(f"Finished. Wrote {written} images to {args.output_root}.")
    if remaining_images:
        print(f"Missing {len(remaining_images)} images.")
        missing_preview = sorted(remaining_images)[:20]
        for image_path in missing_preview:
            print(f"MISSING {image_path}")


if __name__ == "__main__":
    main()
