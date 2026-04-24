import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, env):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Prepare nuScenes mini caches and JSONs for Impromptu-VLA.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("v1.0-mini"),
        help="Path to the nuScenes mini root directory.",
    )
    parser.add_argument(
        "--canbus",
        type=Path,
        default=None,
        help="Path to the can_bus directory. Defaults to <root>/can_bus.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data_qa_generate/data_engine/data_storage/cached_responses"),
        help="Directory where nuscenes_infos_*_mini.pkl will be written.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()
    canbus = args.canbus
    if canbus is None:
        canbus = root / "can_bus"
    else:
        canbus = (repo_root / canbus).resolve() if not canbus.is_absolute() else canbus.resolve()

    cache_dir = (repo_root / args.cache_dir).resolve() if not args.cache_dir.is_absolute() else args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NUSCENES_DATAROOT"] = str(root)
    env["NUSCENES_VERSION"] = "v1.0-mini"
    pythonpath_entries = [str((repo_root / "data_qa_generate").resolve())]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    run_command(
        [
            sys.executable,
            str((repo_root / "data_qa_generate" / "nuScenes_qa.py").resolve()),
            "--dataset",
            "nuscenes",
            "--root-path",
            str(root),
            "--canbus",
            str(canbus),
            "--version",
            "v1.0-mini",
            "--out-dir",
            str(cache_dir),
            "--extra-tag",
            "nuscenes",
        ],
        env,
    )

    run_command(
        [
            sys.executable,
            str((repo_root / "data_qa_generate" / "data_engine" / "datasets" / "nuscenes" / "scripts" / "dataset_nuscenes.py").resolve()),
        ],
        env,
    )

    print("Done.")
    print("Cache files:")
    print(cache_dir / "nuscenes_infos_train_mini.pkl")
    print(cache_dir / "nuscenes_infos_val_mini.pkl")
    print("JSON files:")
    print(repo_root / "data" / "nuscenes" / "nuscenes_test_mini.json")
    print(repo_root / "data" / "nuscenes" / "nuscenes_train_b2_mini.json")


if __name__ == "__main__":
    main()
