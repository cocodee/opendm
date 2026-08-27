"""DM05 SFT and inference entry point for the Supre 16D robot."""

import os
from dataclasses import dataclass, field

import tyro

from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.exp.dm05_exp import (
    DM05DataConfig as _DM05DataConfig,
)
from opendm.exp.dm05_exp import (
    DM05Exp as _DM05Exp,
)
from opendm.exp.dm05_exp import (
    DM05InferenceConfig as _DM05InferenceConfig,
)
from opendm.exp.dm05_exp import (
    DM05OptimizerConfig as _DM05OptimizerConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="supre_pickup_long")
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=2.5e-5)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=10000)
    num_train_steps: int = field(default=50000)
    save_only_model: bool = field(default=False)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=16)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )

    def _request_default_overrides(self) -> dict:
        return {
            "default_robot_type": RobotType.SUPRE.value,
            "default_state_desc": list(ROBOT_STATE_DESCS[RobotType.SUPRE]),
        }


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
