import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter nuscenes_test.json to samples whose referenced images exist under image_root."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path("nuscenes_test.json"),
        help="Path to the original nuscenes test json.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        required=True,
        help="Root directory containing the reconstructed nuscenes image tree.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("nuscenes_test_available.json"),
        help="Path to write the filtered dataset json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    available = []
    missing = []

    for sample in dataset:
        image_paths = sample.get("images", [])
        resolved_paths = [args.image_root / image_path for image_path in image_paths]
        if all(path.exists() for path in resolved_paths):
            available.append(sample)
        else:
            missing.append(sample)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(available, f, ensure_ascii=False, indent=2)

    print(f"Total samples: {len(dataset)}")
    print(f"Available samples: {len(available)}")
    print(f"Missing samples: {len(missing)}")
    print(f"Saved filtered dataset to: {args.output_json}")


if __name__ == "__main__":
    main()
