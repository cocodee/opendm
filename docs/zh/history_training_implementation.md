# 历史视觉输入：文档描述与当前实现对照

本文核对以下训练设计与当前 OpenDM 代码的对应关系：从当前时刻向前取多个历史 slot，对每个 slot 做时间与空间抽样，压缩为固定数量视觉 token，并以随机历史长度和历史增强训练模型，使其适应长、短和缺失历史。

## 结论

当前仓库**实现了历史图像在模型/推理阶段的表示、压缩和缺失 slot 屏蔽**，但**尚未实现该描述中的训练数据管线**。特别是，训练样本不会从同一 episode 的早期帧构造 `history_images`；训练 collator 也不会把历史张量组成 batch。因此目前不能认为训练已使用随机历史长度、时间抽样、历史增强，或因这些训练策略而获得“无历史退化为当前观测策略”的鲁棒性。

下表中的“实现”仅表示代码存在相应能力，不表示该能力已被训练入口使用。

| 文档中的设计 | 当前状态 | 代码证据与说明 |
| --- | --- | --- |
| 从当前时刻向前采样多个历史 slot | 未实现（训练） | `JsonlDataset.__getitem__` 只读取当前 `frame_index` 对应的一行 JSONL，并把完整 `raw_lines` 原样传给后续 transform；没有读取先前帧、slot 数、间隔或时间戳的逻辑。|
| 每个 slot 的时间采样 | 未实现（训练） | 数据集索引只保存 `(file_id, frame_index)`；全仓库训练数据路径中没有按 offset/随机间隔选择历史帧的 transform。RoboChallenge 的 `LogicalStepHistoryStore` 是在线推理客户端的逻辑步采样，不能替代离线训练采样。|
| 每个 slot 的空间抽样 | 部分实现，但不是 history 专用随机空间抽样 | `TrainingTransformPipeline` 对图像提供随机裁剪、旋转、颜色抖动；`PixelTransform` 在 `history_images` 已由调用方提供时也会逐张应用同一 pipeline。但训练数据目前不产生 `history_images`，且每张图独立随机变换，没有 history slot 级的空间采样策略。|
| 每个历史 slot 合并为固定数量视觉 token | 已实现（模型/推理路径） | 每帧历史图对应 16 个 `<unused0>` 占位符。视觉塔输出先重排为二维网格，再经自适应平均池化压缩为 `4 × 4 = 16` 个 token，随后按 `history_mask` 写入这些占位符。|
| 随机历史长度 | 未实现（训练） | 没有历史长度分布或随机截断配置。推理接口允许请求提供 0–5 张历史图，但这是调用方决定的可变输入，而非训练期随机化。|
| 历史增强/历史缺失增强 | 未实现（训练） | 没有针对 slot 丢失、历史帧失效、置空或替换的训练 transform。`<unused1>` 仅在推理 logical-step history 中表示无效 slot，并在模型中置零、从 attention 中屏蔽。|
| 无有效历史时退化为当前观测 | 已支持（推理执行），训练鲁棒性未验证 | 无 `history_images` 时，`ChatTokenization` 不添加历史占位符，当前相机图仍按标准多模态路径编码。若使用 `<unused1>`，模型将其 embedding 置零并从 attention mask 中排除。|

## 当前实际数据流

```text
JSONL 当前帧
  └─ JsonlDataset.__getitem__(frame_index)
      └─ BuildAction / LoadImages（仅当前相机）
          └─ PixelTransform（训练时的通用图像增强）
              └─ ChatTokenization（训练入口未开启 is_history）
                  └─ TrainingCollator（不收集 history_* 字段）
                      └─ 模型仅接收当前相机视觉 token

推理请求或 RoboChallenge 在线缓存提供 history_images
  └─ ChatTokenization(is_history=True)
      ├─ 每张历史图预留 16 个 <unused0>
      └─ 输出 history_pixel_values + history_mask
          └─ 视觉塔 → 2D adaptive average pooling (4×4) → 16 token/帧
              └─ masked_scatter 写入 <unused0> 位置
```

## 已实现部分的细节

### 固定 token 预算

`opendm/constants/robot.py` 定义：

```python
HISTORY_TOKENS_PER_IMAGE = 16
HISTORY_POOL_SIZE = 4
```

当 `ChatTokenization(is_history=True)` 收到 `N` 张 `history_images` 时，会在 prompt 中附加 `N × 16` 个 `<unused0>`。同时会对这些历史图单独调用 image processor，输出 `history_pixel_values`，并从 token ID 得到 `history_mask`。

模型侧（默认路径见 `DM05ForConditionalGeneration._compute_prefix_cache`）将视觉塔输出 `(N, T, H)` 视作 `sqrt(T) × sqrt(T)` 的二维网格，以 `adaptive_avg_pool2d(..., output_size=(4, 4))` 压缩为 `(N, 16, H)`；随后 `masked_scatter` 将其写入恰好对应的占位符。因此 token 数按“每个有效历史图 16 个”固定，而不是按原视觉 patch 数增长。

### 缺失历史的运行时处理

`opendm/model/dm05/dm05_utils.py` 定义 `HISTORY_PAD_TOKEN_ID = 7`（`<unused1>`）。模型在 prefix 阶段将这类位置的 embedding 清零，并通过 `mask_history_pad_tokens_in_attention` 取消其 attention；所以在线 slot 不足不会让无效 token 参与上下文。

RoboChallenge 客户端的 `LogicalStepHistoryStore` 维护每个 session 的主视角历史，并以逻辑动作步构建最多 5 个 slot；无效 slot 用 `<unused1>`，有效 slot 按从旧到新的顺序作为 `history_images` 提交。这是为部署期输入准备的实现，位于 `third_party/robochallenge_inference/`，不在 OpenDM SFT `Dataset`/`DataLoader` 中执行。

### 当前图像增强

训练配置使用 `TrainingTransformPipeline(p=0.5)`：先补边、缩放到 448×448，再以概率 0.5 做固定比例随机裁剪、轻微旋转和颜色抖动。`PixelTransform` 会在 `history_images` 已存在时对它们逐张执行相同 pipeline；这提供了可复用的图像增强基础，但不能自行生成历史帧，也没有保证同一时刻各 slot 的几何变换一致。

## 训练接线为何尚未完成

即使 `DM05DataConfig` 声明了 `is_history: bool = False`，其 `build_dataset()` 构造 `ChatTokenization(...)` 时没有传入 `is_history=self.is_history`。因此即使训练命令设置 `--data-config.is-history true`，默认训练数据流仍不会生成历史占位符或 `history_pixel_values`。

而且，即便补上该参数，仍有两处必要缺口：

1. `JsonlDataset`/transform 没有把过去帧加载并写入 `data["history_images"]`；
2. `TrainingCollator` 只拼接 `input_ids`、当前 `pixel_values`、action 等字段，没有整理变长的 `history_pixel_values` 和 `history_mask`，也不会把它们传给训练模型。

所以当前训练中 `DM05ForConditionalGeneration.forward(..., history_pixel_values, history_mask)` 的可选参数没有来自标准训练 batch 的供给。它们主要由服务推理路径使用。

## 若要完整实现该训练设计，最小改动范围

1. 在数据集/transform 层按当前 `frame_index` 与 episode 边界生成候选过去帧；为每个历史 slot 定义时间采样规则（例如随机 offset 或随机间隔），并加载指定历史相机。
2. 增加历史策略配置：最大 slot 数、历史长度分布、slot drop/失效概率、时间抖动，以及 history 专用或时序一致的图像增强策略。
3. 将采得的帧以从旧到新顺序写入 `history_images`，在训练构造 `ChatTokenization` 时显式传递 `is_history=self.is_history`。
4. 扩展 `TrainingCollator`：处理 batch 内不同历史长度，拼接 history pixel tensor、保留每个样本对应的 mask/占位符，并把两个字段传给 trainer/model。
5. 为 0、1、最大 slot 数、部分 slot 无效及 episode 起始帧分别添加测试；验证每个有效历史帧恰好得到 16 个 token，无效 slot 不参与 attention，且训练 batch 可完成一次 forward/backward。

## 关键文件

- `opendm/data/dataset.py`：当前帧读取与 episode 内 `frame_index`。
- `opendm/exp/dm05_exp.py`：默认 SFT pipeline；这里缺少将 `is_history` 传给训练 `ChatTokenization` 的接线。
- `opendm/data/transforms.py`：图像增强、历史占位符、历史图预处理。
- `opendm/data/collator.py`：当前训练 batch 组装，尚未包含 history 字段。
- `opendm/constants/robot.py`：16 token/帧、4×4 池化尺度。
- `opendm/model/dm05/dm05_arch.py`：默认模型路径中的历史特征池化、注入与 pad 屏蔽。
- `opendm/infer/dm05_trt_utils.py`：fast 推理路径的相同池化逻辑与最多 5 帧打包。
- `third_party/robochallenge_inference/policies/logical_step_history.py`：部署期 logical-step slot 构造。
