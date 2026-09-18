# 动态动作匹配：文档描述与当前实现对照

本文核对“动态动作匹配”训练设计与当前 OpenDM 代码。该设计要求：模型的固定长度预测片段与更细粒度真实轨迹之间，通过严格单调的动作锚点、动态规划和相邻轨迹连续性项寻找最小代价匹配。

## 结论

当前仓库**没有实现动态动作匹配**。训练使用的是固定时间索引的未来动作 chunk 加 Flow Matching 损失：预测序列第 `i` 个位置始终监督为原始 episode 中未来第 `i` 个位置，没有候选锚点、单调约束、动态规划或相邻锚点连续性代价。

代码中的 “Dynamic” 是 `DynamicCache`、TensorRT 动态 shape/fallback 等推理工程术语，和动态动作对齐无关；Flow Matching 是生成动作的训练目标，也不是轨迹匹配算法。

| 文档中的设计 | 当前状态 | 代码证据与说明 |
| --- | --- | --- |
| 固定长度预测动作片段 | 已实现 | `BuildActionChunk(action_horizon)` 构造长度为 `action_horizon` 的目标；模型 action expert 接收同样长度的动作序列。|
| 数据侧保留更细粒度真实动作轨迹供候选匹配 | 未实现 | 样本虽携带整个 episode 的 `raw_lines`，但 `BuildActionChunk` 仅顺序读取未来连续 `action_horizon` 帧；没有输出候选轨迹、候选 mask 或额外细粒度监督张量。|
| 为每个预测动作选择真实动作锚点 | 未实现 | 无锚点索引、候选动作代价矩阵或 backtracking 输出。训练 batch 只含 `action` 与 `action_mask`。|
| 锚点严格单调递增 | 未实现 | 没有比较/约束 `anchor[i] < anchor[i+1]` 的代码，也没有等价的单调匹配路径。固定索引 `start + step` 天然按时间递增，但这不是可选择的对齐。|
| 通过动态规划最小化整体匹配损失 | 未实现 | 仓库训练、模型和 loss 路径中没有 DTW、动态规划表、最短路径或 min-plus recurrence。|
| 相邻锚点的轨迹连续性代价 | 未实现 | 损失只有逐元素 MSE；没有预测增量与真实锚点间增量的比较、速度/加速度正则项或跳跃惩罚。|
| 降低采集节奏/时间相位噪声 | 未由该机制实现 | 当前固定逐帧监督仍与采集时间索引绑定。是否具有节奏泛化能力只能由数据覆盖和实验评估决定，不能归因于动态动作匹配。|

## 当前实际训练数据流

```text
当前 episode 第 t 帧
  └─ JsonlDataset.__getitem__(t)
      └─ 附加整个 episode 的 raw_lines
          └─ BuildActionChunk(H)
              ├─ 若当前帧有 action：target[i] = action[t + i]
              └─ 否则：target[i] = state[t + 1 + i]
                 （越过 episode 尾部时重复最后一帧）
                  └─ action: [1, H, D]，action_mask: 全 1
                      └─ TrainingCollator 拼成 [B, H, D]
                          └─ Flow Matching MSE，逐元素在同一 (b, i, d) 上比较
```

这里的 `H` 是 `action_horizon`/`chunk_size`。没有“预测位置 `i` 可匹配真实轨迹任意候选位置 `j`”的分支；它固定为 `j = start + i`。

## 固定时间监督的具体实现

### 目标 chunk 构造

`opendm/data/dataset.py` 的 `JsonlDataset.__getitem__` 为每个训练样本保存当前 `frame_index` 和该 JSONL 文件全部 `raw_lines`。随后 `opendm/data/transforms.py` 的 `BuildActionChunk`：

```python
for step in range(self.action_horizon):
    raw_idx = start + step
    if raw_idx <= episode_term:
        frame = orjson.loads(lines[raw_idx])
        last_value = np.asarray(frame[read_key], dtype=np.float32)
    values.append(last_value)
data["action"] = np.stack(values, axis=0)[None, ...]
```

- 样本已有 `action` 时，`start = frame_index`；
- 否则，`start = frame_index + 1`，以未来 state 作为动作目标；
- 轨迹尾部不足时，重复最后有效值；
- `action_mask` 被设置成和 chunk 同形状的全 `True`，它用于动作维度 padding，而非有效锚点或轨迹位置选择。

`BuildAction` 只是在 absolute 与 relative 表示之间包装这个固定 chunk；relative 模式将关节目标减去当前 state（夹爪维度保持绝对值），不会改变时间对齐。

### 模型训练损失

`opendm/model/dm05/dm05_arch.py` 的 `DM05ForConditionalGeneration.forward` 使用条件 Flow Matching。它为固定 target `action` 采样噪声和时间：

```python
x_t = time * noise + (1 - time) * action
u_t = noise - action
v_t = self.model.action_out_proj(suffix_out)
elem_mse = F.mse_loss(v_t, u_t, reduction="none")
fm_loss = ((elem_mse * action_mask).sum((1, 2))
           / action_mask.sum((1, 2))).mean()
```

这个 MSE 的索引是固定的 `[batch, action_time, action_dim]`。`action_mask` 仅排除补齐的动作维度；它没有对真实轨迹的候选时间点做最小化，也没有重排 target。训练器 `DMTrainer.compute_loss` 只记录模型返回的 `fm_loss`，不会添加第二个动作对齐损失。

## 与文档机制的差异

动态动作匹配至少需要一个预测长度 `P` 与候选真实轨迹长度 `R` 之间的代价矩阵 `C[b, p, r]`。随后动态规划应在满足严格单调递增的路径上选择 anchor，例如：

```text
dp[p, r] = C[p, r] + min(dp[p - 1, 0:r])
```

还应将相邻预测变化 `pred[p] - pred[p-1]` 与所选真实锚点变化 `gt[r] - gt[r_prev]` 的误差，或等效的跳跃/连续性代价，纳入路径转移。当前代码不产生 `R`、`C`、`dp`、anchor path 或连续性项；上式仅说明该文档所述机制需要的计算结构，并非本项目现有实现。

此外，Flow Matching 的网络输出是噪声路径上的速度 `v_t`，不是可以直接与真实动作锚点做普通 MSE 的最终去噪动作。因此若要新增动态匹配，需要明确匹配发生在以下哪一层：原始 action target、由 `x_t`/`v_t` 推出的去噪 action，还是一个独立辅助监督；不同选择会决定梯度与计算成本。

## 若要实现该机制，最小改动范围

1. 扩展数据配置和 `BuildActionChunk`：除固定 `H` 个动作外，读取更长的未来真实轨迹窗口，并输出轨迹长度/有效 mask；不得跨 episode。
2. 明确动态匹配的训练接口：预测长度 `P`、候选长度 `R`、动作距离（按机器人/夹爪维度加权）、严格单调条件，以及尾部/无可行路径的处理。
3. 在模型 loss 中构造 `[B, P, R]` 的候选代价，并实现可批处理的动态规划/回溯或可微松弛；禁止使用独立的每个位置 argmin，因为它不能保证全局单调性。
4. 在动态规划转移中加入相邻预测动作变化与相邻真实锚点变化的连续性代价，并定义跳跃上限或惩罚，防止跳过关键动作过程。
5. 在 `TrainingCollator` 中组装新增的真实轨迹和 mask；让 `DMTrainer` 记录分项损失，例如 point-match、continuity 和 total-match。
6. 添加单元测试：严格单调路径、相同轨迹的零/最小代价、不可行 mask、速度不同但进展一致的轨迹，以及“局部最近但整体非单调”的反例。再以一次 forward/backward smoke test 验证梯度可回传。

## 关键文件

- `opendm/data/dataset.py`：当前帧索引及完整 episode 的 `raw_lines`。
- `opendm/data/transforms.py`：`BuildActionChunk` 和 `BuildAction`；当前固定未来索引 target 的来源。
- `opendm/data/collator.py`：训练 batch 只包含单个固定长度 `action` target 及其 mask。
- `opendm/model/dm05/dm05_arch.py`：条件 Flow Matching 前向计算与唯一的 `fm_loss`。
- `opendm/trainer/trainer.py`：调用/记录模型 loss，不含额外匹配优化。
