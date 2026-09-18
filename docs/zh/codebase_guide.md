# OpenDM/DM05 代码与算法深度解析

> 本文档系统性地拆解 OpenDM 项目的代码结构、核心算法和工程实现，按 5 个递进阶段组织。

---

## 目录

- [Stage 1: 基础 — 数据流与常量定义](#stage-1-基础--数据流与常量定义)
- [Stage 2: 核心架构 — 双流 VLA + Flow Matching](#stage-2-核心架构--双流-vla--flow-matching)
- [Stage 3: 训练管线与实验配置](#stage-3-训练管线与实验配置)
- [Stage 4: MuonAdamW 优化器](#stage-4-muonadamw-优化器)
- [Stage 5: 快速推理 — Triton + TensorRT + CUDA Graph](#stage-5-快速推理--triton--tensorrt--cuda-graph)
- [贯穿全局的设计模式](#贯穿全局的设计模式)
- [核心文件索引](#核心文件索引)
- [下一步建议](#下一步建议)

---

## Stage 1: 基础 — 数据流与常量定义

**目标**: 理解数据从磁盘到模型输入的完整流转。

### 1.1 常量与机器人定义 (`opendm/constants/robot.py`)

项目支持 **7 种机器人具身**，每种有不同的状态描述：

| 机器人 | 状态描述 | 动作维度 |
|--------|----------|----------|
| DOS W1 / Aloha / Aloha RoboTwin2 | 6关节+夹爪 + 6关节+夹爪 | 14 |
| SO101 | 5关节+夹爪 | 6 |
| ARX5 | 6关节+夹爪 | 7 |
| UR5 | 6末端+夹爪 | 7 |

关键常量：

```python
HISTORY_TOKENS_PER_IMAGE = 16  # 每帧历史图像压缩为 4×4=16 个 token
HISTORY_POOL_SIZE = 4           # 2D 池化空间大小
HISTORY_PAD_TOKEN_ID = 7       # <unused1> token ID，用于无效历史槽
```

状态描述枚举 `RobotStateDesc`:
- `JOINT` — 关节角度
- `EEF` — 末端执行器位姿
- `GRIPPER` — 夹爪开合

动作模式枚举 `ActionMode`:
- `ABSOLUTE` — 绝对动作
- `RELATIVE` — 增量动作（相对于当前状态）

### 1.2 数据集加载 (`opendm/data/dataset.py`)

`JsonlDataset` 的工作流程：

1. 扫描 `jsonl_dir` 下所有 `.jsonl` 文件（支持 `megfile` 透明访问本地/S3 路径）
2. 构建 `index_cache.json`（每文件的行数统计），加速后续索引
3. `__getitem__` 加载单帧 JSON，**同时附加整个 episode 的 `raw_lines`**（供后续 `BuildActionChunk` 读取未来帧）
4. 应用变换管线

```python
class JsonlDataset(Dataset):
    def __getitem__(self, idx):
        file_index, frame_index = self.sample_index[idx]
        jsonl_file = self.id_to_jsonl[file_index]
        lines = _read_jsonl_lines(jsonl_file)
        result = orjson.loads(lines[frame_index])
        result["raw_lines"] = lines       # 整个 episode，供 BuildActionChunk 读取未来帧
        result["meta_data"] = {...}
        if self.transforms is not None:
            result = self.transforms(result)
        return result
```

### 1.3 归一化统计 (`opendm/data/normalize.py`)

**核心算法 — 在线分位数计算**：

`RunningStats` 维护在线统计，核心特性：
- Welford 在线均值/方差
- **5000-bin 直方图**计算 q01/q99 分位数
- 直方图在 min/max 扩展时自动重分布（`_adjust_histograms`）

```python
class RunningStats:
    def _compute_quantiles(self, quantiles):
        """基于直方图计算分位数"""
        for q in quantiles:
            target_count = q * self._count
            for hist, edges in zip(self._histograms, self._bin_edges):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
```

`NormStatsFile`：多机器人统计分派，`select(robot_type)` 选取对应配置。

### 1.4 变换管线 (`opendm/data/transforms.py`) — 最核心的数据文件

**完整管线**：

```
JsonlDataset → Pipeline([
    BuildAction,
    LoadImages,
    PixelTransform,
    Normalize,
    ChatTokenization,
    PadAction
]) → TrainingCollator
```

#### `BuildActionChunk` (L218-283)

从当前帧开始读取 `action_horizon`（通常50）个未来帧的动作值，构建 `[1, horizon, dim]` 动作块。

- 有 `action` 字段时：`read_key = "action"`, `start = frame_index`
- 无 `action` 字段时：用未来帧的 `state` 作为目标，`read_key = "state"`, `start = frame_index + 1`
- 超出 episode 末尾时重复最后一个值

#### `ActionRelative` (L286-350)

增量编码 `action - state`，**夹爪维度保持绝对值**：

```python
relative = action - state
# 夹爪维度保持原始绝对值
non_delta_indices = [i for i, sid in enumerate(state_desc) if sid in self.non_delta_ids]
relative[..., non_delta_indices] = action[..., non_delta_indices]
```

#### `Normalize` / `Denormalize` (L84-134)

**分位数归一化** — clip-and-scale 到 [-1,1]，非均值/方差标准化：

```python
# 归一化
arr = np.clip(arr, q01, q99)
out = ((arr - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0)
return np.where((q01 == 0) & (q99 == 0), 0.0, out)  # 零范围维度输出0

# 反归一化
out = ((arr + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01)
```

#### `ChatTokenization` (L474-653)

构建 Gemma3 聊天模板，示例输出：

```
Robot: Aloha
Control mode: ...
Overall speed: 0.5
Task: pick up the red block.
History images: <unused0><unused0>...<unused0>  ← 16×N 个占位符
Head image: <image_soft_token>...
Left wrist image: <image_soft_token>...
Right wrist image: <image_soft_token>...
States: 128 45 200 ...  ← 256-bin 离散化状态值
```

状态离散化：`action_to_bin_tokens` 将 [-1,1] 的归一化值映射到 [0, 255] 的整数 bin ID。

#### `PadAction` (L656-676)

所有动作填充/截断到 `shared_dim=32`，实现多具身统一动作空间：

```python
class PadAction:
    def __init__(self, shared_dim: int = 32):
        self.shared_dim = shared_dim
    
    def _pad_last_dim(self, arr, pad_value=0):
        if arr.shape[-1] >= self.shared_dim:
            return arr[..., :self.shared_dim]  # 截断
        pad = self.shared_dim - arr.shape[-1]
        return torch.nn.functional.pad(arr, (0, pad), value=pad_value)  # 填充
```

### 1.5 图像增强 (`opendm/data/augmentations.py`)

- **训练时**：`PadToSquare → Resize(448,448) → RandomResizedCrop(0.95) → Rotate(±2°) → ColorJitter(b=0.3,c=0.4,s=0.3)`
- **推理时**：`PadToSquare → Resize(448,448)`（无增强）

### 1.6 批处理 (`opendm/data/collator.py`)

- `TrainingCollator`：填充 `input_ids`/`attention_mask`/`token_type_ids` 到 `max_length=1024`，拼接 `pixel_values`/`action`/`action_mask`
- `NormStatsCollator`：按 `robot_type` 分组，为归一化统计计算聚合 state/action

### 1.7 数据集注册 (`opendm/dataset/register.py`)

数据集注册表 `CONVERSATION_DATA`，自动导入同目录下所有 `.py` 模块，并支持 `$OPENDM_DATA_PATH` 环境变量扩展：

```python
CONVERSATION_DATA = {}

def register_dataset(dataset, prefix=""):
    if prefix:
        dataset = {f"{prefix}_{k}": v for k, v in dataset.items()}
    CONVERSATION_DATA.update(dataset)

# 自动导入
_import_registry_modules(_DEFAULT_REGISTRY_DIR)
# 支持外部扩展
_extra_registry_path = os.getenv("OPENDM_DATA_PATH")
if _extra_registry_path:
    _import_registry_modules(_extra_registry_path)
```

---

## Stage 2: 核心架构 — 双流 VLA + Flow Matching

**目标**: 深入理解 DM05 模型架构。

### 2.1 整体架构：双流设计

```
┌─────────────────────────────────────────────────────┐
│              DM05ForConditionalGeneration             │
│                                                      │
│  ┌──────────────┐        ┌──────────────────┐        │
│  │   VLM 前缀    │  KV    │  Action Expert   │        │
│  │  (Gemma3)    │ Cache  │    后缀          │        │
│  │              │───────>│                  │        │
│  │ SigLIP视觉塔 │        │ + AdaRMSNorm     │        │
│  │ + Gemma3文本 │        │ + 时间条件调制    │        │
│  └──────────────┘        └──────────────────┘        │
│         ↑                       ↑                    │
│    action_in_proj         action_out_proj            │
│    time_mlp_in/out                                   │
└─────────────────────────────────────────────────────┘
```

**关键约束**：Action Expert 必须与 VLM 语言模型有**相同的层数/头维度/RoPE 参数**（`validate_action_config_compatible`），这样 AE 才能直接消费 VLM 的 KV 缓存。

### 2.2 DM05Config (`dm05_arch.py` L55-80)

```python
class DM05Config(DMBaseConfig):
    model_type = "dm05"
    tie_word_embeddings: bool = True
    
    def __init__(self, vlm_config=None, action_config=None,
                 action_dim: int = 32, chunk_size: int = 50, ...):
        self.vlm_config = vlm_config        # Gemma3Config
        self.action_config = action_config  # Gemma3TextConfig
        self.action_dim = action_dim        # 统一动作空间维度
        self.chunk_size = chunk_size        # 动作预测步数
```

### 2.3 DM05ActionExpert (`dm05_arch.py` L82-116)

继承自 `Gemma3TextModel`，额外添加时间调制器：

```python
class DM05ActionExpert(Gemma3TextModel):
    def __init__(self, config):
        # 标准 Gemma3 解码器层
        self.layers = nn.ModuleList([Gemma3DecoderLayer(config, idx) ...])
        self.norm = Gemma3RMSNorm(config.hidden_size)
        
        # 时间步条件调制器 — 每层两个 + 一个最终
        self.input_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.mlp_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.final_time_modulator = nn.Linear(hidden_size, 3 * hidden_size)
```

### 2.4 AdaRMSNorm — 时间步条件注入的核心 (`dm05_arch.py` L283-302)

```python
def _adaptive_rmsnorm(self, norm, x, adarms_cond, modulator):
    # 1. 标准 RMSNorm
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    
    # 2. 时间条件调制 — DiT 风格
    modulation = modulator(adarms_cond)       # Linear(hidden, 3*hidden)
    scale, shift, gate = torch.chunk(modulation, 3, dim=-1)  # 拆分为三部分
    
    # 3. 仿射变换 + 门控
    output = normed * (1 + scale) + shift     # 条件注入
    return output, gate                       # gate 用于残差缩放
```

**时间条件的构建路径**：

```
time → posemb_sincos(dim) → time_mlp_in → SiLU → time_mlp_out → SiLU → adarms_cond
```

正弦-余弦位置编码 (`posemb_sincos`)：
- `min_period = 4e-3`, `max_period = 256.0`
- 对数均匀分布的频率：`period = min_period × (max_period/min_period)^fraction`

### 2.5 Action Expert 单层计算 (`dm05_arch.py` L304-369)

```
输入: suffix_embeds, prefix_cache_keys, prefix_cache_values, adarms_cond

1.  pre-attention AdaRMSNorm:
    prenorm, attn_gate = AdaRMSNorm(input_layernorm, suffix_embeds, time)

2.  Q/K/V 投影:
    Q,K,V = q/k/v_proj(prenorm).view(hidden_shape).transpose(1,2)

3.  QK-Norm:
    Q = q_norm(Q), K = k_norm(K)

4.  RoPE:
    cos, sin = rotary_emb(Q, position_ids)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

5.  KV 缓存拼接 (核心！):
    K = torch.cat([prefix_cache_keys, K], dim=2)   ← 前缀KV + 后缀KV
    V = torch.cat([prefix_cache_values, V], dim=2)

6.  注意力:
    attn_out = attention(Q, K, V, suffix_attn_mask)

7.  残差 + 门控:
    residual = suffix_embeds + o_proj(attn_out) * attn_gate

8.  pre-MLP AdaRMSNorm:
    mlp_input, mlp_gate = AdaRMSNorm(pre_ff_ln, residual, time)

9.  MLP:
    mlp_out = mlp(mlp_input)

10. 残差 + 门控:
    output = residual + mlp_out * mlp_gate
```

**核心思想**：AE 从不重新编码观测前缀！它只是在 KV 维度拼接前缀缓存，让后缀 token 可以 attend 到整个前缀。

### 2.6 DM05Model — 模型组装 (`dm05_arch.py` L387-462)

```python
class DM05Model(DMPreTrainedModel):
    def __init__(self, config):
        # VLM
        self.vlm = Gemma3ForConditionalGeneration(vlm_config)
        validate_action_config_compatible(action_config, self.language_model.config)
        
        # Action Expert
        self.action_expert = DM05ActionExpert(action_config)
        
        # 动作投影层
        self.action_in_proj = nn.Linear(action_dim, ae_hidden_size)
        self.action_out_proj = nn.Linear(ae_hidden_size, action_dim)
        
        # 时间条件投影层
        self.time_mlp_in = nn.Linear(ae_hidden_size, ae_hidden_size)
        self.time_mlp_out = nn.Linear(ae_hidden_size, ae_hidden_size)
```

### 2.7 Flow Matching 训练前向 (`dm05_arch.py` L743-837)

```python
def forward(self, input_ids, pixel_values, action, action_mask, ...):
    batch_size = input_ids.shape[0]
    
    # Step 1: VLM 前缀前向 → 填充 KV 缓存（只做一次！）
    kv_cache, prefix_len = self._compute_prefix_cache(
        input_ids, attention_mask, pixel_values, token_type_ids,
        history_pixel_values, history_mask, cache_cls=VLADynamicCache
    )
    
    # Step 2: Flow Matching 噪声采样
    noise = torch.randn_like(action)                        # ε ~ N(0, I)
    time = Beta(1.5, 1.0).sample() * 0.999 + 0.001         # t ~ Beta(1.5, 1.0), 偏向噪声端
    
    # Step 3: 矫直流插值
    x_t = t * noise + (1 - t) * action                     # 线性插值
    u_t = noise - action                                    # 目标速度场（直线方向）
    
    # Step 4: 后缀前向
    suffix_embeds = action_in_proj(x_t)                     # 投影到 AE 隐藏空间
    adarms_cond = _build_adarms_cond(time)                  # 构建时间条件
    suffix_out = _suffix_forward(suffix_embeds, kv_cache, adarms_cond)
    
    # Step 5: Flow Matching 损失
    v_t = action_out_proj(suffix_out)                       # 预测速度
    elem_mse = F.mse_loss(v_t, u_t, reduction="none")      # [B, T, D]
    per_sample_fm = (elem_mse * action_mask).sum((1,2)) / action_mask.sum((1,2))
    fm_loss = per_sample_fm.mean()                          # 只在有效动作维度上计算
```

**Flow Matching 数学公式**：

$$x_t = t \cdot \epsilon + (1-t) \cdot a, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$u_t = \epsilon - a \quad \text{(目标速度场)}$$

$$\mathcal{L} = \text{MSE}(v_\theta(x_t, t), u_t) \cdot \text{action\_mask}$$

**为什么用 Beta(1.5, 1.0)？** 均匀采样会让模型过多关注接近真实动作的低噪声区域，而 Beta(1.5, 1.0) 偏向高噪声端，让模型更好学习从噪声到动作的完整轨迹。

### 2.8 Euler 积分推理 (`dm05_arch.py` L843-922)

```python
def inference_action(self, input_ids, pixel_values, diffusion_steps=10, action_mask=None, ...):
    # Step 1: VLM 前缀前向 → 填充 KV 缓存
    kv_cache, prefix_len = self._compute_prefix_cache(...)
    
    # Step 2: 从纯噪声开始
    x_t = torch.randn(batch_size, chunk_size, action_dim, ...)  # x_t ~ N(0, I)
    time_val = 1.0
    dt = -1.0 / diffusion_steps                                 # 时间步长
    
    # Step 3: Euler 积分（从 t=1 积分到 t=0）
    for _ in range(diffusion_steps):  # 默认 10 步
        time_tensor = torch.full((batch_size,), time_val, ...)
        if action_mask is not None:
            x_t = x_t * action_mask          # 只在有效维度上操作
        
        suffix_embeds = action_in_proj(x_t)
        adarms_cond = _build_adarms_cond(time_tensor)
        suffix_out = _suffix_forward(suffix_embeds, kv_cache, adarms_cond)
        v_t = action_out_proj(suffix_out)
        
        x_t = x_t + v_t * dt                # Euler 步进
        time_val += dt
    
    return x_t  # t=0 时的 x_t 就是预测的动作
```

### 2.9 历史图像注入 (`dm05_arch.py` L924-1007)

```
input_ids 中有 <unused0> 占位符 (HISTORY_TOKENS_PER_IMAGE=16 个/图)
→ embed_tokens(input_ids) 得到嵌入
→ 无效历史槽 <unused1> (token_id=7) 的嵌入置零
→ 历史图像经 SigLIP 视觉编码器
→ adaptive_avg_pool2d 到 4×4 = 16 token/图
→ masked_scatter 到 <unused0> 位置
→ <unused1> 在注意力掩码和位置 ID 中屏蔽
→ VLM 前缀前向填充 KV 缓存
```

### 2.10 后缀注意力掩码 (`dm05_utils.py` L243-283)

```python
def make_suffix_attn_mask(input_ids, prefix_len, suffix_len, ...):
    """每个后缀 token 可以 attend 到:
    - 非填充的前缀 token
    - 所有后缀 token（全注意力）
    """
    # 前缀掩码：padding 和 <unused1> 位置为 -inf
    prefix_mask = torch.zeros(batch_size, suffix_len, prefix_len)
    pad_mask = (prefix_ids == pad_token_id) | (prefix_ids == HISTORY_PAD_TOKEN_ID)
    prefix_mask = prefix_mask.masked_fill(pad_mask, NEG_INF)
    
    # 后缀掩码：全零（全注意力）
    suffix_mask = torch.zeros(batch_size, suffix_len, suffix_len)
    
    # 拼接 → [B, 1, suffix_len, prefix_len + suffix_len]
    mask = torch.cat([prefix_mask, suffix_mask], dim=2)
    return mask.unsqueeze(1)
```

### 2.11 梯度检查点安全的 KV 缓存 (`dm05_utils.py` L102-139)

**问题**：上游 `GradientCheckpointingLayer` 在重计算时会清除 `past_key_values`，但 DM05 前缀前向需要写入 KV 缓存供后缀复用。

**解决方案**：
- `SafeCacheDecoderLayer`：覆盖 `__call__`，在梯度检查点模式下保留 `past_key_values`
- `OverwriteDynamicLayer`：覆写语义的缓存层，`update` 直接赋值（保留梯度）
- `VLADynamicCache`：基于 `OverwriteDynamicLayer` 的动态缓存
- `patch_decoder_layers`：猴子补丁替换 VLM 解码器层

---

## Stage 3: 训练管线与实验配置

**目标**: 理解模型/数据/优化器/训练器如何组装。

### 3.1 配置系统：tyro 嵌套 dataclass

```
DM05Exp
├── DM05ModelConfig     → 模型加载、注意力、梯度检查点、LoRA
├── DM05OptimizerConfig → adamw/muon_adamw 选择、学习率、Muon 超参
├── DM05TrainerConfig   → FSDP1/DDP、批大小、保存策略、W&B
└── DM05InferenceConfig → Flask 服务、推理步数、快速后端
```

所有字段都可通过 CLI 访问：`--model-config.chunk-size 50 --optimizer-config.optim muon_adamw`

入口：
```python
if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train": exp.train()
    elif exp.task == "inference": exp.inference()
```

### 3.2 DM05ModelConfig (`dm05_exp.py` L62-155)

```python
@dataclass
class DM05ModelConfig(Config):
    model_name_or_path: str | None = "./checkpoints/DM05"
    chunk_size: int = 50
    bf16: bool = True
    llm_attn_implementation: str = "flex_attention"
    vision_attn_implementation: str = "flash_attention_2"
    action_attn_implementation: str = "sdpa"
    freeze_vlm_embedding: bool = True
    vlm_gradient_checkpointing: bool = True
    ae_gradient_checkpointing: bool = True
    lora_config: DM05LoraConfig = DM05LoraConfig()
```

### 3.3 模型构建流程

```
from_pretrained(checkpoint)
→ set_attention_implementation(LLM=flex, Vision=flash_attn2, Action=sdpa)
→ enable_gradient_checkpointing(vlm=True, ae=True)
→ freeze_vlm_embedding()
→ _apply_liger_kernel(RMSNorm+GeGLU+RoPE)
→ [可选] apply_lora_to_dm05_model(r=32, α=16, all-linear)
→ [可选] mark_muon_parameters()
→ FSDP/DDP wrap
```

注意力后端解析规则：
- **LLM**: `auto` → `flex_attention`（需要 PyTorch ≥2.5），否则 `sdpa`
- **Vision**: `auto` → `flash_attention_2`（需要 bf16+CUDA），否则 `sdpa`
- **Action Expert**: `auto` → `flex_attention`，否则 `sdpa`

### 3.4 DMTrainer 关键定制 (`trainer.py`)

```python
class DMTrainer(Trainer):
    def create_optimizer(self, model=None):
        """拦截优化器创建，muon_adamw 时注入 MuonAdamW"""
        if self.exp_config.optimizer_config.optim == "muon_adamw":
            self.optimizer = self.exp_config.optimizer_config.build_muon_adamw(opt_model)
            return self.optimizer
        # 否则使用 HF Trainer 默认 AdamW
    
    def compute_loss(self, model, inputs, ...):
        """提取 *_loss 字段，跨 rank all-reduce 平均后缓存"""
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        for loss_key in [_ for _ in outputs if _.endswith("_loss")]:
            # 跨 rank 归约
            buf = torch.tensor([raw, count], device=loss.device)
            torch.distributed.all_reduce(buf, op=ReduceOp.SUM)
            self.loss_cache[loss_key] = buf[0].item() / buf[1].item()
    
    def _save_checkpoint(self, model, trial, ...):
        """每个检查点后复制 norm_stats.json，确保自包含"""
        super()._save_checkpoint(model, trial)
        shutil.copyfile(norm_stats_path, os.path.join(checkpoint_dir, "norm_stats.json"))
```

### 3.5 FSDP 配置

```python
if trainer_config.fsdp1 and world_size > 1:
    linked_args["fsdp"] = "shard_grad_op"      # FSDP-1，仅分片梯度+优化器
    linked_args["fsdp_config"] = {
        "backward_prefetch": "BACKWARD_PRE",    # 反向预取
        "use_orig_params": True,                # 保持原始参数（Muon 需要）
        "sync_module_states": True,             # 同步初始状态
    }
    linked_args["seed"] += int(os.environ.get("RANK", 0))  # 每 rank 不同种子
else:
    linked_args["ddp_find_unused_parameters"] = True        # DDP 回退
```

### 3.6 LoRA 配置 (`dm05_lora.py`)

```python
@dataclass
class DM05LoraConfig:
    r: int = 32
    lora_alpha: int = 16
    target_modules: str = "all-linear"  # 除 lm_head 外所有 Linear
    modules_to_save: list = [
        "action_in_proj", "action_out_proj",  # 动作投影层全量训练
        "time_mlp_in", "time_mlp_out",        # 时间条件层全量训练
        "dm05_time_modulators",               # 时间调制器别名（展开为逐层）
    ]
```

`dm05_time_modulators` 别名展开：
```python
# 展开为
"input_time_modulators.0", "input_time_modulators.1", ..., "input_time_modulators.N",
"mlp_time_modulators.0", "mlp_time_modulators.1", ..., "mlp_time_modulators.N",
"final_time_modulator"
```

推理时 LoRA 加载：`PeftModel.from_pretrained(base_model, adapter_path) → merge_and_unload()`

### 3.7 数据管线构建 (`dm05_exp.py` L292-335)

```python
pipeline = Pipeline([
    BuildAction(action_horizon=action_horizon, action_mode=ActionMode.RELATIVE),
    LoadImages(image_keys=image_keys, image_dir=image_dir),
    PixelTransform(transform_pipeline=TrainingTransformPipeline(p=0.5)),
    Normalize(norm_stats_path=norm_stats_path, norm_keys=["state", "action"], use_quantiles=True),
    ChatTokenization(processor=processor, n_bins=256, max_length=1024, image_prompts=image_prompts),
    PadAction(32),  # 统一到 32 维
])
```

### 3.8 归一化统计计算 (`dm05_exp.py` L347-447)

- Rank 0 计算，其他 rank 轮询等待（每 5 秒检查文件出现）
- 使用 `NormStatsCollator` 按 robot_type 分组
- 文件名包含内容寻址哈希：`{dataset_name}_{sha256(dataset_name|transform)[:16]}.json`

---

## Stage 4: MuonAdamW 优化器

**目标**: 理解混合 Muon/AdamW 优化器。

### 4.1 参数选择策略

`is_default_muon_parameter` 的选择逻辑：

```python
def is_default_muon_parameter(name, param):
    if not param.requires_grad:  return False   # 必须可训练
    if param.ndim != 2:          return False   # 必须是 2D 矩阵
    if not (name contains "action_expert.layers." and ends with ".weight"):
        return False                             # 必须属于 AE 层
    if name contains "embed_tokens" or "norm.weight":
        return False                             # 排除嵌入和归一化
    return True
```

**Muon 只应用于 AE 的 Q/K/V/O 投影和 MLP 权重**，其余参数用标准 AdamW。

### 4.2 FSDP 前的参数标记 (`muon_adamw.py` L68-98)

```python
def mark_muon_parameters(model, predicate=None):
    """在 FSDP 包装前记录原始矩阵形状"""
    for name, param in model.named_parameters():
        if not predicate(name, param): continue
        setattr(param, "_opendm_muon_shape", tuple(param.shape))  # 记录原始形状
        setattr(param, "_opendm_muon_name", name)                  # 记录参数名
```

**为什么必须在 FSDP 前？** FSDP 会展平参数，丢失原始矩阵维度信息。预记录的 `_opendm_muon_shape` 保留了原始矩阵维度，供正交化时 reshape 使用。

### 4.3 Newton-Schulz 正交化 (`muon_adamw.py` L101-128)

```python
def zeropower_via_newtonschulz5(matrix, steps=5, eps=1e-7):
    """近似矩阵的极因子（零次幂）"""
    update = matrix.float()
    
    # 转置处理：确保 rows ≤ cols
    if update.shape[0] > update.shape[1]:
        update = update.T
    
    # 归一化
    update = update / update.norm().clamp_min(eps)
    
    # Newton-Schulz 迭代
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * (gram @ gram)) @ update
    
    return update.T if transposed else update
```

**直觉理解**：Muon 的核心思想是"梯度应该指向正交方向"。Newton-Schulz 迭代将梯度矩阵投影到其极因子（最接近的正交矩阵），从而约束参数更新在正交群附近。

**五次 quintic 多项式** `(a·I + b·G + c·G²) · M` 是零次幂 `M(M^T M)^{-1/2}` 的迭代近似，系数 (3.4445, -4.7750, 2.0315) 经过优化使 5 次迭代达到高精度。

### 4.4 MuonAdamW 优化器结构 (`muon_adamw.py` L178-236)

```python
class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, model, *, lr, betas, eps, weight_decay,
                 muon_momentum=0.95, muon_nesterov=True,
                 muon_ns_steps=5, muon_lr_scale=1.0,
                 muon_moonlight_coefficient=0.2):
        # 将参数分为两组
        groups = [
            {"params": adamw_params, "use_muon": False},
            {"params": muon_params, "use_muon": True, "muon_shapes": muon_shapes},
        ]
```

### 4.5 Muon 单步更新流程 (`muon_adamw.py` L362-414)

```python
def _muon_step(self, group):
    for param, matrix_shape, plan in zip(params, shapes, plans):
        # 1. 动量缓冲
        buffer = state["exp_avg"]
        buffer.mul_(momentum).add_(grad)              # buffer = μ·buffer + grad
        
        # 2. Nesterov 方向
        direction = grad + momentum * buffer           # 前瞻梯度
        
        # 3. 权重衰减
        param.mul_(1.0 - lr * weight_decay)
        
        # 4. 全矩阵收集（FSDP 兼容）
        full_direction, offset = self._gather_matrix(direction, plan)
        
        # 5. Newton-Schulz 正交化
        update = zeropower_via_newtonschulz5(
            full_direction.reshape(matrix_shape), steps=5
        ).reshape(-1)
        
        # 6. 取回本地分片
        local_update = update.narrow(0, offset, plan.local_numel).reshape_as(param)
        
        # 7. Moonlight 有效学习率
        effective_lr = lr × lr_scale × moonlight_coeff × √max(matrix_shape)
        # 例: 2.5e-5 × 1.0 × 0.2 × √1024 = 2.5e-5 × 0.2 × 32 = 1.6e-4
        
        # 8. 参数更新
        param.add_(local_update, alpha=-effective_lr)
```

**Moonlight 缩放** 的作用：Muon 的正交更新幅度与矩阵维度无关，但 AdamW 的更新幅度与 `1/√dim` 相关。`√max_dim` 缩放让 Muon 的有效学习率与维度成正比，与 AdamW 在不同维度参数上保持可比的更新幅度。

### 4.6 FSDP 分片规划 (`muon_adamw.py` L443-535)

`_build_muon_shard_plan` 将分片分类为 5 种情况：

| 类型 | 含义 | `_gather_matrix` 行为 |
|------|------|----------------------|
| `empty` | 本地无数据 | 返回空张量 |
| `local` | 单 GPU，全部数据在本地 | 直接返回 |
| `owner_only` | FSDP 下只有 1 个 rank 持有数据 | 直接返回 |
| `partial` | FSDP 下多个 rank 各持有部分 | 用 `all_gather` 重构完整矩阵 |
| `replicated` | 所有 rank 都有完整副本 | 直接返回 |

`_gather_matrix` 的 partial 路径：
```python
def _gather_matrix(local, plan):
    """用 all_gather 从各 rank 收集完整矩阵"""
    padded = local.new_zeros(max(active_sizes))     # 填充到最大分片大小
    padded[:local.numel()] = local
    shards = [torch.empty_like(padded) for _ in active_sizes]
    dist.all_gather(shards, padded, group=plan.process_group)  # 子进程组 all_gather
    return torch.cat([shard[:size] for shard, size in zip(shards, sizes)]), offset
```

---

## Stage 5: 快速推理 — Triton + TensorRT + CUDA Graph

**目标**: 理解训练架构如何转化为低延迟部署。

### 5.1 整体架构：三级加速

```
┌─────────────────────────────────────────────────────────┐
│                  DM05FastInferRuntime                    │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │  TensorRT    │   │  前缀解码器   │   │  后缀解码器  │ │
│  │  视觉编码器  │──>│  (HF FlexAttn)│──>│  (Triton    │ │
│  │  (FP16 ONNX)│   │  + CUDA Graph │   │  BigKernel) │ │
│  └──────────────┘   └──────────────┘   └─────────────┘ │
│         ↑                  ↑                   ↑        │
│    图外执行          CUDA Graph 内       CUDA Graph 内   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 TensorRT 视觉编码器 (`dm05_infer/dm05_trt_utils.py`)

- **SigLIP 视觉模型**导出为 ONNX → 编译为 TensorRT FP16 引擎
- 支持历史图像：TRT 引擎同时处理 `num_current + MAX_HISTORY_IMAGES` 张图
- 历史图像特征经 `pool_image_features_to_history` 池化到 4×4=16 token
- **在 CUDA Graph 外执行**（TRT 运行时与 CUDA Graph 不兼容）

```python
class DM05VisionTensorRTRunner:
    """TensorRT 视觉编码器执行器"""
    def __call__(self, pixel_values, output_tensor=None):
        """执行 TRT 引擎，输出图像特征"""
```

### 5.3 StaticPrefixCacheLayer — 地址稳定缓存 (`dm05_infer.py` L31-113)

```python
class StaticPrefixCacheLayer(CacheLayerMixin):
    def _allocate_like(self, key_states, value_states):
        self.keys = torch.empty_like(key_states)        # 预分配
        self.values = torch.empty_like(value_states)
        torch._dynamo.mark_static_address(self.keys)     # 锁定 GPU 地址！
        torch._dynamo.mark_static_address(self.values)
    
    def update(self, key_states, value_states):
        self.keys.copy_(key_states)      # copy_ 而非赋值！
        self.values.copy_(value_states)  # 保持地址不变
        return self.keys, self.values
```

**对比训练路径的 `OverwriteDynamicLayer`**：
- **训练**：`self.keys = key_states`（直接赋值，保留梯度）
- **推理**：`self.keys.copy_(key_states)`（拷贝，保持地址不变，否则 CUDA Graph 失效）

### 5.4 CUDA Graph 捕获流程 (`dm05_infer.py` L284-327)

```
1. 启动时，对每个桶长度 (576, 704, 768, 896, 1024):
   ├── 创建虚拟输入（dummy_pixels → TRT → dummy_image_features）
   ├── 构建虚拟 input_ids/attention_mask/token_type_ids
   ├── 预分配所有缓冲区（prefix_buffers, suffix_buffers）
   ├── mark_static_address 标记所有张量
   ├── 预热运行 2 次（_run_graph_profile_ops）
   └── 捕获 CUDA Graph（with torch.cuda.graph(graph): ...）

2. 推理时：
   ├── TRT 执行视觉编码（图外）
   ├── 将结果 copy_ 到预分配缓冲区
   ├── 选择最近桶（bisect.bisect_left）
   └── graph.replay() → 整个前缀+后缀+10步Euler 一步完成！
```

### 5.5 桶化前缀长度

默认桶：`(576, 704, 768, 896, 1024)`

选择策略：`bisect.bisect_left(buckets, request_len)` — 选取 ≥ 请求长度的最小桶。

**为什么需要桶化？** CUDA Graph 在捕获时固定了所有张量的形状和地址。不同前缀长度意味着不同的 KV 缓存大小和注意力掩码形状，需要为每种形状捕获独立的图。桶化将连续的前缀长度空间离散化为有限个桶，避免为每个可能长度捕获一个图。

### 5.6 Triton BigKernel (`dm05_infer/dm05_bigkernel.py`)

**形状特化**——为固定 AE 架构编译，避免动态形状开销：

```python
SUPPORTED_HIDDEN_SIZE = 1024
SUPPORTED_INTERMEDIATE_SIZE = 4096
SUPPORTED_HEAD_DIM = 256
SUPPORTED_Q_HEADS = 8
SUPPORTED_KV_HEADS = 4
```

**预分配缓冲区 `DM05BigKernelBuffers`**：

```python
@dataclass
class DM05BigKernelBuffers:
    qkv: torch.Tensor           # 融合 QKV 投影输出
    query: torch.Tensor         # RoPE 后的 Q
    key: torch.Tensor           # RoPE 后的 K
    value: torch.Tensor         # V
    residual: torch.Tensor      # 残差连接
    mlp_input: torch.Tensor     # MLP 输入
    mlp_hidden: torch.Tensor    # MLP 隐藏层
    mlp_out: torch.Tensor       # MLP 输出
    layer_out: torch.Tensor     # 层输出
    attn_logits: torch.Tensor   # 注意力分数
    attn_probs: torch.Tensor    # 注意力概率
    attn_out: torch.Tensor      # 注意力输出
```

**融合核函数** — 对照训练代码 `_compute_suffix_layer`：

| Triton 融合核 | 对应训练步骤 |
|--------------|------------|
| 融合 QKV+QK-norm+RoPE | Q/K/V 投影 → q_norm/k_norm → apply_rotary_pos_emb |
| 融合 AdaRMS-norm | `_adaptive_rmsnorm`: RMSNorm + scale/shift + gate |
| 融合 attention-post-norm-residual | O 投影 → post_attention_layernorm → residual+gate |
| 融合 GeGLU-tanh | pre_ff_layernorm → MLP(GeGLU) → post_ff_layernorm → residual+gate |

### 5.7 推理路径选择

| 条件 | 路径 | 延迟 |
|------|------|------|
| 无历史 + 前缀长度 ≤ 最大桶 | **CUDA Graph 回放**（最快） | ~15ms |
| 无历史 + 前缀长度 > 最大桶 | **动态 eager 回退**（无图） | ~50ms |
| 有历史图像 | **TRT + 无捕获快速 prefill/decode**（图外） | ~30ms |

### 5.8 fused_linear_euler_update (`dm05_utils.py` L26-52)

推理时融合 Euler 步进为单个 `addmm` 操作：

```python
def fused_linear_euler_update(*, hidden_states, current, linear, dt):
    """current + dt * linear(hidden_states)  →  单个 addmm"""
    hidden_flat = hidden_states.reshape(batch_size * seq_len, -1)
    current_flat = current.reshape(batch_size * seq_len, action_dim)
    updated = torch.addmm(
        current_flat,
        hidden_flat,
        linear.weight.transpose(0, 1),
        beta=1.0,
        alpha=float(dt),
    )
    return updated.view_as(current)
```

---

## 贯穿全局的设计模式

### 1. 前缀/后缀分离

每条代码路径（训练、推理、快速推理）都是 VLM 前缀产 KV 缓存 → AE 后缀消费之。这是架构的不变量。

### 2. 多具身填充

32 维动作空间 + `action_mask` 贯穿数据、训练损失、推理：
- **数据**：`PadAction(32)` 填充动作到 32 维
- **训练**：`MSE(v_t, u_t) * action_mask` 只在有效维度计算损失
- **推理**：`x_t = x_t * action_mask` 每步屏蔽无效维度

### 3. 历史图像处理

池化+散布逻辑在三个位置一致：
- `_compute_prefix_cache`（训练/推理）
- `pool_image_features_to_history`（TRT utils）
- `ChatTokenization`（placeholder 插入）

### 4. 注意力后端分派

`set_attention_implementation` 为 LLM/vision/AE 三条路径分别解析后端：
- LLM: `flex_attention` (默认) — 可组合注意力掩码
- Vision: `flash_attention_2` (默认) — 高效 FlashAttention
- AE: `sdpa` (默认) — PyTorch Scaled Dot Product Attention

### 5. 配置驱动

所有组件通过 `dataclass + tyro` 参数化，CLI 与编程控制统一：
```bash
python exp.py --task train \
    --model-config.chunk-size 50 \
    --optimizer-config.optim muon_adamw \
    --data-config.dataset-name libero_goal \
    --inference-config.backend fast
```

---

## 核心文件索引

| 文件 | 行数 | 角色 |
|------|------|------|
| `opendm/model/dm05/dm05_arch.py` | 1079 | 中心架构：双流模型、Flow Matching 训练前向、Euler 推理、前缀缓存、AdaRMSNorm |
| `opendm/model/dm05/dm05_utils.py` | 283 | 工具：VLADynamicCache、SafeCacheDecoderLayer、后缀注意力掩码、时间嵌入、融合 Euler |
| `opendm/model/dm05/dm05_lora.py` | 395 | LoRA 配置与包装：别名展开、target 解析、FSDP 兼容补丁 |
| `opendm/data/transforms.py` | 725 | 数据管线：动作分块、归一化、分词、填充 |
| `opendm/data/normalize.py` | 365 | 归一化统计：RunningStats、NormStatsFile、多机器人分派 |
| `opendm/data/dataset.py` | 103 | 数据集：JsonlDataset、索引缓存 |
| `opendm/data/collator.py` | 136 | 批处理：TrainingCollator、NormStatsCollator |
| `opendm/optimizer/muon_adamw.py` | 535 | 混合优化器：Newton-Schulz 正交化、FSDP 分片规划 |
| `opendm/exp/dm05_exp.py` | 1171 | 实验编排：嵌套配置、组装、训练循环、推理服务 |
| `opendm/trainer/trainer.py` | 197 | 训练器：DMTrainer、优化器注入、损失归约 |
| `opendm/infer/dm05_bigkernel.py` | 1405 | Triton 融合核函数：形状特化、预分配缓冲区 |
| `opendm/infer/dm05_infer.py` | 911 | CUDA Graph + 静态缓存推理运行时 |
| `opendm/infer/dm05_infer_arch.py` | 1366 | 快速推理模型架构：图可捕获变体 |
| `opendm/infer/dm05_trt_utils.py` | 615 | TensorRT 工具：ONNX 导出、引擎构建/执行 |
| `opendm/constants/robot.py` | 47 | 机器人常量：RobotType、RobotStateDesc、历史图像配置 |

---

## 下一步建议

1. **实际运行 demo 推理**：
   ```bash
   script/dm05_launcher.sh --task inference
   tests/curl_demo.sh
   ```

2. **深入读 playground 实验**：`playground/dm05_sft_demo.py`，理解如何自定义实验配置

3. **尝试修改超参数**：调整 `chunk_size` 或 `diffusion_steps`，观察对推理质量的影响

4. **对照 Triton 核函数**：在 `dm05_bigkernel.py` 中识别每个融合操作对应训练代码的哪个步骤

5. **手动追踪数据流**：从 `assets/demo/` 的 JSONL 数据出发，完整走通训练和推理两条路径
