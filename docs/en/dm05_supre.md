# DM05 Supre Dataset and Inference

OpenDM supports the Supre follower data in
`dataset_0729_pickup_long` and `dataset_0813_pickup_long`. The converter keeps
the original LeRobot MP4 files and creates OpenDM JSONL records with three
video views: `Head`, `Left wrist`, and `Right wrist`.

## Convert the datasets

Install the conversion dependency and run from the OpenDM repository root:

```bash
pip install -e ".[data-convert]"
python tools/convert_lerobot_supre.py \
  --input /root/data2/dc_dir/datasets/dataset_0729_pickup_long \
  --input /root/data2/dc_dir/datasets/dataset_0813_pickup_long \
  --output /root/data2/dc_dir/datasets/supre_pickup_long_opendm
```

The output contains one JSONL file per source episode under `jsonl/`. Video
paths remain root-relative references to the source datasets, so no video copy
is required. Use `--overwrite` only when intentionally regenerating the output.

The converted state, action, and force vectors have 16 dimensions in this
order:

```text
left_arm_joint_1..6, left_arm_joint_7,
right_arm_joint_1..6, right_arm_joint_7,
trunk_joint_1, trunk_joint_2
```

`right_arm_joint_7` is inserted as a fixed `0.0` placeholder. The two trunk
joint values are preserved from the original 15-dimensional LeRobot records.

## Train and serve

Point the registry and launch the Supre entry point:

```bash
export OPENDM_SUPRE_JSONL_DIR=/root/data2/dc_dir/datasets/supre_pickup_long_opendm/jsonl
torchrun --nproc_per_node 1 playground/dm05_supre.py \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000
```

For inference, use the trained Supre checkpoint with
`--exp playground/dm05_supre.py`, `--task inference`, and
`--inference-config.output-action-dim 16`. Requests must send 16 state values
and three images in the configured order. The SDK controller must use the same
joint order and must support `trunk_joint_1/2`.
