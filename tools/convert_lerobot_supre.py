"""Convert LeRobot Supre episodes to OpenDM JSONL records.

The converter keeps the source MP4 files in place and writes video references
with absolute paths, allowing two source dataset roots to be merged safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_JOINTS = [
    *(f"left_arm_joint_{i}.pos" for i in range(1, 8)),
    *(f"right_arm_joint_{i}.pos" for i in range(1, 7)),
    "trunk_joint_1.pos",
    "trunk_joint_2.pos",
]
EXPECTED_FORCE_JOINTS = [name.replace(".pos", ".force") for name in EXPECTED_JOINTS]
EXPECTED_CAMERAS = ["head_cam", "left_wrist_cam", "right_wrist_cam"]


def expand_supre_vector(values: list[float], *, field: str) -> list[float]:
    """Insert the fixed right gripper after the six right-arm joints."""
    if len(values) != 15:
        raise ValueError(f"{field} must contain 15 values, got {len(values)}")
    return [*values[:13], 0.0, *values[13:]]


def load_info(root: Path) -> dict[str, Any]:
    with (root / "meta/info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    features = info.get("features", {})
    for field in ("action", "observation.state", "observation.force"):
        feature = features.get(field, {})
        expected_names = (
            EXPECTED_FORCE_JOINTS if field == "observation.force" else EXPECTED_JOINTS
        )
        if feature.get("shape") != [15] or feature.get("names") != expected_names:
            raise ValueError(f"{root}: unexpected {field} schema")
    image_features = [f"observation.images.{camera}" for camera in EXPECTED_CAMERAS]
    if any(features.get(field, {}).get("dtype") != "video" for field in image_features):
        raise ValueError(f"{root}: expected all three camera features to be videos")
    if info.get("fps") != 30 or info.get("robot_type") != "supre_robot_follower":
        raise ValueError(f"{root}: unsupported FPS or robot type")
    return info


def _read_tasks(root: Path) -> dict[int, str]:
    tasks = {}
    with (root / "meta/tasks.jsonl").open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def convert_dataset(source: Path, output_jsonl: Path, *, source_tag: str) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot conversion requires pyarrow; install it with `pip install pyarrow`."
        ) from exc

    info = load_info(source)
    tasks = _read_tasks(source)
    output_jsonl.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    video_pattern = info["video_path"]
    for parquet_path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        if not rows:
            continue
        episode_index = int(rows[0]["episode_index"])
        out_path = output_jsonl / f"{source_tag}_episode_{episode_index:06d}.jsonl"
        with out_path.open("w", encoding="utf-8") as file:
            for row in rows:
                frame_index = int(row["frame_index"])
                record = {
                    "images_1": {
                        "type": "video",
                        "url": str(
                            (
                                source
                                / video_pattern.format(
                                    episode_chunk=episode_index // info["chunks_size"],
                                    video_key="observation.images.head_cam",
                                    episode_index=episode_index,
                                )
                            ).resolve()
                        ).lstrip("/"),
                        "frame_idx": frame_index,
                    },
                    "images_2": {
                        "type": "video",
                        "url": str(
                            (
                                source
                                / video_pattern.format(
                                    episode_chunk=episode_index // info["chunks_size"],
                                    video_key="observation.images.left_wrist_cam",
                                    episode_index=episode_index,
                                )
                            ).resolve()
                        ).lstrip("/"),
                        "frame_idx": frame_index,
                    },
                    "images_3": {
                        "type": "video",
                        "url": str(
                            (
                                source
                                / video_pattern.format(
                                    episode_chunk=episode_index // info["chunks_size"],
                                    video_key="observation.images.right_wrist_cam",
                                    episode_index=episode_index,
                                )
                            ).resolve()
                        ).lstrip("/"),
                        "frame_idx": frame_index,
                    },
                    "state": expand_supre_vector(
                        row["observation.state"], field="state"
                    ),
                    "action": expand_supre_vector(row["action"], field="action"),
                    "force": expand_supre_vector(
                        row["observation.force"], field="force"
                    ),
                    "prompt": tasks[int(row["task_index"])],
                    "is_robot": True,
                }
                file.write(json.dumps(record, separators=(",", ":")) + "\n")
                total_frames += 1
        if episode_index != int(rows[-1]["episode_index"]):
            raise ValueError(f"{parquet_path}: contains multiple episodes")
    return total_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", dest="inputs", action="append", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = args.output / "jsonl"
    if (
        output_jsonl.exists()
        and any(output_jsonl.glob("*.jsonl"))
        and not args.overwrite
    ):
        raise SystemExit(f"{output_jsonl} is non-empty; pass --overwrite to replace it")
    output_jsonl.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, source in enumerate(args.inputs):
        total += convert_dataset(
            source, output_jsonl, source_tag=f"dataset_{index:02d}"
        )
    print(
        f"Converted {len(args.inputs)} datasets and {total} frames into {output_jsonl}"
    )


if __name__ == "__main__":
    main()
