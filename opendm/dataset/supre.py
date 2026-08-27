"""Supre dual-arm and trunk dataset registration."""

import os

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

register_dataset(
    {
        "pickup_long": {
            "jsonl_dir": os.getenv(
                "OPENDM_SUPRE_JSONL_DIR", "./data/supre_pickup_long/jsonl"
            ),
            # Converted records contain root-relative video URLs because the
            # two source datasets live under different roots.
            "image_dir": os.getenv("OPENDM_SUPRE_IMAGE_DIR", "/"),
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.SUPRE,
            "state_desc": ROBOT_STATE_DESCS[RobotType.SUPRE],
        },
    },
    prefix="supre",
)
