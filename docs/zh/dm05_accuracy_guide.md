# DM0.5 / OpenDM 代码精度指南

本文从代码实现出发，说明 OpenDM 中 DM0.5 为什么能够在开放指令、长时序任务和不同机器人本体上保持较好的动作精度，以及微调时哪些设置最容易造成精度下降。

本文中的“精度”包括两层含义：

1. **预测精度**：动作轨迹是否贴近示范轨迹，训练 loss 是否真正反映有效动作。
2. **执行精度**：归一化动作经过采样、反归一化后，是否以正确的维度、坐标系和时间顺序发送给机器人。

> 参考：[DM0.5 技术介绍](https://www.dexmal.com/blog/dm0.5)。本文只解读当前仓库中可验证的实现；模型指标、训练数据规模和线上性能请以官方发布内容为准。

## 1. 先看完整的数据流

DM0.5 不是把图像直接接一个动作回归头，而是将 Gemma3 VLM 与专门的 Action Expert 组合起来：

```text
JSONL / 视频帧 / 机器人状态
        │
        ├─ LoadImages + PixelTransform
        ├─ BuildActionChunk
        ├─ ActionAbsolute 或 ActionRelative
        └─ Normalize + TrainingCollator
        │
        ▼
当前图像 + 语言 + 状态/历史图像 ──► Gemma3 VLM prefix
                                      │
                                      └─ 每层 KV cache
                                                │
随机噪声 x₁ + 时间 t ─► Action Expert suffix ─► 速度场 vθ(xₜ,t|prefix)
                                                │
                                      Euler 积分（推理）
                                                ▼
                                      action chunk → Denormalize → 控制器
```

主要代码入口：

| 环节 | 代码位置 | 关键实现 |
| --- | --- | --- |
| 模型组合 | `opendm/model/dm05/dm05_arch.py` | `DM05Model`、`DM05ForConditionalGeneration` |
| 数据构造 | `opendm/data/transforms.py` | `BuildActionChunk`、`ActionRelative`、`Normalize` |
| 训练目标 | `opendm/model/dm05/dm05_arch.py` | `DM05ForConditionalGeneration.forward` |
| 推理采样 | `opendm/model/dm05/dm05_arch.py` | `inference_action` |
| 长记忆/缓存 | `opendm/model/dm05/dm05_utils.py`、`opendm/infer/dm05_infer_arch.py` | `VLADynamicCache`、prefix/suffix fast path |
| 训练配置 | `opendm/exp/dm05_exp.py`、`playground/dm05_sft_demo.py` | `DM05*Config` |

## 2. 创新点与旧方法的对应关系

### 2.1 VLM 与 Action Expert 分工：保留语义能力，专门建模连续动作

传统做法常见两种问题：

- 在 VLM 的语言输出空间中离散化动作，量化误差会直接影响关节或末端执行器控制。
- 在 VLM 后接一个很浅的 MLP 直接回归动作，视觉语言理解和低层控制共享同一套表示，微调时容易损伤通用语义能力。

OpenDM 的 `DM05Model` 将两者分开：

```python
self.vlm = Gemma3ForConditionalGeneration(vlm_config_ref)
self.action_expert = DM05ActionExpert(action_config_ref)
self.action_in_proj = nn.Linear(config.action_dim, ae_hidden_size)
self.action_out_proj = nn.Linear(ae_hidden_size, config.action_dim)
```

Action Expert 复用 Gemma3 decoder layer 的结构，但只接受 `suffix_embeds`，不带词表 embedding 和 `lm_head`。这使动作保持连续向量形式，并让动作模块可以接收时间条件。修改 `action_dim` 时，应同步确认机器人状态描述、归一化统计和输出控制器的维度。

### 2.2 Flow Matching：从“单点回归”改成条件速度场

直接回归通常学习 `action = f(observation)`；同一观察下如果示范包含多种合理轨迹，单点 MSE 容易得到平均动作，表现为轨迹迟钝、拐角变形或抓取时机不准。

本项目的训练 forward 使用 flow matching：

```python
noise = torch.randn_like(action)
time = (
    torch.distributions.Beta(1.5, 1.0).sample((batch_size,))
    .to(action.device, dtype=action.dtype) * 0.999 + 0.001
)
x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * action
u_t = noise - action
v_t = self.model.action_out_proj(suffix_out).to(torch.float32)
fm_loss = F.mse_loss(v_t, u_t, reduction="none")
```

模型学习的是从数据动作到噪声的条件速度 `u_t = noise - action`，而不是只拟合一个动作点。推理时从噪声出发，用 `diffusion_steps` 次 Euler 更新回到动作：

```python
dt = -1.0 / diffusion_steps
for _ in range(diffusion_steps):
    ...
    v_t = self.model.action_out_proj(suffix_out)
    x_t = x_t + v_t * dt
```

精度要点：训练和推理都必须使用同一个动作空间、同一 `action_mask` 语义和匹配的 `chunk_size`。提高 `diffusion_steps` 通常增加计算量并可能改善数值积分，但不能弥补错误的归一化或坐标系。

### 2.3 Action Chunking：预测未来轨迹，而不是只预测下一帧

旧式单步控制每次只预测 `a_t`，容易产生高频抖动，并且每次决策都忽略短期未来。 `BuildActionChunk` 从当前帧构造固定长度的未来动作序列；越过 episode 末尾时重复最后一个有效值，保证 batch 形状稳定：

```python
data["action"] = np.stack(values, axis=0)[None, ...]
data["action_mask"] = np.ones_like(data["action"], dtype=bool)
```

默认 `chunk_size=50`，定义在 `DM05Config` 和实验配置中。实际部署时，控制器应明确使用 chunk 中的多少步以及多久重新推理一次；不要把 50 个动作一次性无条件发送给机器人。

### 2.4 相对动作与绝对动作：减少跨本体偏差

绝对关节角或绝对末端坐标对初始姿态、机器人尺寸和标定误差敏感。 `ActionRelative` 用当前状态计算未来动作增量，同时对夹爪维度保留绝对值：

```python
relative = action - state
relative[..., non_delta_indices] = action[..., non_delta_indices]
```

该设计兼顾了空间动作的平移/姿态变化与夹爪开合语义。 `ActionMode.RELATIVE` 是默认配置，但训练和推理必须完全一致；若训练用 relative、部署却按 absolute 发送，动作会整体失真。

### 2.5 长时序记忆：历史图像 token 化并屏蔽无效槽位

仅使用当前帧的旧方法在遮挡、物体暂时离开视野、长序列指令中容易丢失上下文。OpenDM 为每张历史图像保留 `4×4=16` 个 pooled vision tokens：

```python
grid = F.adaptive_avg_pool2d(
    grid, output_size=(HISTORY_POOL_SIZE, HISTORY_POOL_SIZE)
)
image_features = grid.permute(0, 2, 3, 1).reshape(
    -1, HISTORY_POOL_SIZE * HISTORY_POOL_SIZE, hidden
)
```

历史 token 被写入 `<unused0>` 占位位置；无效的 `<unused1>` 槽位由 `HISTORY_PAD_TOKEN_ID = 7` 标识，并同时执行三件事：embedding 置零、attention mask 置零、position id 不计入有效长度。相关实现分别位于 `DM05ForConditionalGeneration._compute_prefix_cache`、`mask_history_pad_tokens_in_attention` 和 `_build_suffix_position_ids`。

这样做的精度收益不是简单“塞更多图片”，而是避免 padding 被模型误认为真实视觉证据。修改历史帧数量或 token 模板时，必须同时修改 `opendm/constants/robot.py`、数据处理和推理 fast path。

### 2.6 Prefix KV Cache：把昂贵的视觉语言计算与多步采样解耦

Flow matching 推理需要多次调用 Action Expert。若每一步都重复计算图像和语言 prefix，延迟高且数值路径更复杂。DM05 先执行一次 VLM prefix：

```python
kv_cache, prefix_len = self._compute_prefix_cache(...)
```

随后 Action Expert 仅计算 suffix，并把 prefix 的 K/V 拼接到当前 suffix 的 K/V：

```python
key_states = torch.cat([cache_keys, key_states], dim=2)
value_states = torch.cat([cache_values, value_states], dim=2)
```

`validate_action_config_compatible` 会检查两边的层数、RoPE、attention heads、head dim 等配置；这些字段不一致时拒绝复用，避免“能运行但精度异常”。训练阶段使用 `VLADynamicCache` 的可回写 cache，推理阶段使用 `DynamicCache`；部署 fast path 进一步提供 FlexAttention、预计算时间调制和 TensorRT vision 路径。

## 3. 精度最关键的代码检查清单

### 3.1 动作维度、顺序和机器人类型

`ROBOT_STATE_DESCS` 决定每一维是 `joint`、`eef` 还是 `gripper`。例如 SO101 是 5 个关节加 1 个夹爪，UR5 是 6 个末端位姿维度加 1 个夹爪。检查：

```python
assert action.shape[-1] == state.shape[-1]
```

此外要确认 JSONL 中维度顺序与 `state_desc` 顺序一致。双臂数据不能把左右臂维度交叉；这类错误通常不会触发 shape error，却会严重损害精度。

### 3.2 归一化统计必须与动作模式绑定

默认使用每个维度的 1%/99% 分位数裁剪到 `[-1, 1]`：

```python
arr = np.clip(arr, lo, hi)
out = (arr - lo) / (hi - lo + 1e-6) * 2.0 - 1.0
```

统计文件由数据集、动作模式和 horizon 共同决定，训练 checkpoint 保存时也会复制 `norm_stats.json`。推理时必须使用同一文件，并根据 `meta_data["robot_type"]` 选择多机器人 profile。不要跨 robot type 复用统计；不要在 relative/absolute 切换后继续使用旧统计。

### 3.3 采样 mask 与 padding

训练 loss 对 `action_mask` 加权：

```python
per_sample_fm = (elem_mse * action_mask).sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2))
```

如果新增变长动作或末尾 padding，必须正确设置 mask；全零 mask 会导致除零，错误的全一 mask 会让 padding 参与训练。文本 padding 由 `TrainingCollator` 补齐到 `model_max_length`，超长输入会被截断，需重点检查图像 token 是否在截断前后仍与 `token_type_ids` 对齐。

### 3.4 时间条件和推理步数

`posemb_sincos`、`time_mlp_in`、`time_mlp_out` 共同生成 Action Expert 的 AdaRMSNorm 条件。训练时间采样范围是 `(0.001, 1.0)`，推理从 `time=1.0` 反向走到 0。修改时间范围、Euler 方向或 `diffusion_steps` 任一项，都必须重新验证动作尺度和成功率。

## 4. 推荐的精度优先配置

以 `playground/dm05_sft_demo.py` 为起点，建议先固定数据语义，再调性能参数：

```bash
torchrun --nproc_per_node 1 playground/dm05_sft_demo.py \
  --task train \
  --data-config.action-mode relative \
  --model-config.chunk-size 50 \
  --model-config.bf16 true \
  --model-config.vlm-gradient-checkpointing true \
  --model-config.ae-gradient-checkpointing true
```

说明：

- `relative` 适合多数跨初始姿态场景，但必须保证控制端做对应的增量执行。
- `chunk-size` 应与训练数据 horizon 和推理控制周期共同决定。
- `bf16` 是默认路径；vision 使用 `flash_attention_2` 时必须是 CUDA 且 `bf16=True`。
- 显存不足时优先启用 gradient checkpointing；它降低显存而不改变目标函数。
- `use_lora=True` 适合小数据快速适配，但若目标域与底座差异很大，应比较全量微调和 LoRA 的验证集轨迹误差。

## 5. 评估不要只看 fm loss

建议至少记录以下指标：

1. **masked flow-matching loss**：确认有效动作维度上的训练是否收敛。
2. **反归一化后的逐维 MAE/RMSE**：分别统计关节、末端和夹爪，避免大尺度维度掩盖小尺度维度。
3. **chunk 内平滑度**：计算相邻动作差分，检查是否出现抖动或突跳。
4. **闭环成功率**：按 robot type、任务长度、相机视角和是否有历史帧分组。
5. **扰动鲁棒性**：改变光照、遮挡、初始位姿和相机输入，验证记忆机制是否真正有效。

一个最小的离线检查应验证：

```python
assert predicted_action.shape == target_action.shape
assert torch.isfinite(predicted_action).all()
assert action_mask.any(dim=(1, 2)).all()
```

闭环评估时，必须先 `Denormalize`，再按 `ActionMode` 解释，并把动作维度映射回 `state_desc`；仅比较归一化空间的 loss 不能证明机器人执行精度。

## 6. 常见精度退化与定位顺序

| 现象 | 优先检查 | 常见原因 |
| --- | --- | --- |
| 所有动作整体偏大/偏小 | `Normalize`、`Denormalize`、norm stats | 统计文件不匹配或重复归一化 |
| 夹爪方向反了 | `ActionRelative.non_delta_ids`、控制器约定 | 把夹爪误当 delta，或开合编码相反 |
| 单臂有效、双臂错位 | `ROBOT_STATE_DESCS`、JSONL 维度顺序 | 左右臂排列不一致 |
| 长任务中途遗忘 | `history_mask`、`HISTORY_PAD_TOKEN_ID`、历史图像顺序 | 无效槽位未屏蔽或历史帧时间顺序错误 |
| 动作抖动 | `chunk_size`、`diffusion_steps`、控制重规划频率 | chunk 未平滑执行或 Euler 步数过少 |
| 训练 loss 很低但闭环失败 | 反归一化动作、坐标系、state/action 对齐 | 监督目标是错位帧或执行端解释不同 |
| 开启 fast path 后精度变化 | attention backend、cache 长度、RoPE position ids | fast path 与 eager 路径的 mask/位置不一致 |
| LoRA 适配后语义退化 | `dm05_lora.py` 的 target modules、冻结策略 | 注入层过少或误更新共享 embedding |

推荐定位顺序是：**样本可视化 → action/state 对齐 → 归一化往返测试 → eager 与 fast path 对齐 → 闭环测试**。先确认单个样本的原始动作经过所有 transform 后仍然正确，再讨论模型容量和学习率。

## 7. 可复现性与性能开关

精度基线建议固定：数据集注册名、`action_mode`、`chunk_size`、norm stats 文件、`diffusion_steps`、随机种子和 attention backend。性能优化可以逐项打开：

- `flex_attention`：用于语言和动作 attention 的高效实现。
- `flash_attention_2`：用于 vision，依赖 CUDA、`flash_attn` 和 bf16。
- Liger Kernel：`DM05ForConditionalGeneration._apply_liger_kernel` 对 RMSNorm、GeGLU、RoPE 等算子做融合。
- TensorRT / big-kernel：位于 `opendm/infer/`，应与 eager/标准 PyTorch 路径做数值回归。

性能开关的验收标准不是只看延迟，而是对同一固定输入比较：prefix mask、action mask、归一化后输出范围、反归一化动作和最终闭环结果。允许存在浮点误差，但不应出现系统性偏移或动作维度错位。

## 8. 总结

OpenDM 中最直接影响精度的改进可以概括为：

- **连续 flow matching + Action Expert**：表达多模态动作轨迹，降低离散化和平均动作问题。
- **固定长度 action chunk**：利用短期未来信息，提高动作连续性。
- **relative action + robot-specific normalization**：减少本体、姿态和量纲差异。
- **历史视觉 token + 无效槽位屏蔽**：为长时序任务保留上下文，同时避免 padding 污染。
- **prefix KV cache + fast suffix sampling**：让多步动作采样可实时运行，并保持 VLM 与控制模块的清晰分工。

真正的高精度来自“数据语义、动作空间、mask、采样器和执行器”全链路一致，而不是单独降低一个训练 loss。

