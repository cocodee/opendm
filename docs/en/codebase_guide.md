# OpenDM/DM05 Code and Algorithms Deep Dive

> This document systematically breaks down the OpenDM project's code structure, core algorithms, and engineering implementation, organized into 5 progressive stages.

---

## Table of Contents

- [Stage 1: Fundamentals — Data Flow and Constant Definitions](#stage-1-fundamentals--data-flow-and-constant-definitions)
- [Stage 2: Core Architecture — Dual-Stream VLA + Flow Matching](#stage-2-core-architecture--dual-stream-vla--flow-matching)
- [Stage 3: Training Pipeline and Experiment Configuration](#stage-3-training-pipeline-and-experiment-configuration)
- [Stage 4: MuonAdamW Optimizer](#stage-4-muonadamw-optimizer)
- [Stage 5: Fast Inference — Triton + TensorRT + CUDA Graph](#stage-5-fast-inference--triton--tensorrt--cuda-graph)
- [Design Patterns Across the Whole System](#design-patterns-across-the-whole-system)
- [Core File Index](#core-file-index)
- [Suggested Next Steps](#suggested-next-steps)

---

## Stage 1: Fundamentals — Data Flow and Constant Definitions

**Goal**: Understand the complete flow of data from disk to model input.

### 1.1 Constants and Robot Definitions (`opendm/constants/robot.py`)

The project supports **7 robot embodiments**, each with a different state description:

| Robot | State Description | Action Dim |
|--------|----------|----------|
| DOS W1 / Aloha / Aloha RoboTwin2 | 6 joints + gripper + 6 joints + gripper | 14 |
| SO101 | 5 joints + gripper | 6 |
| ARX5 | 6 joints + gripper | 7 |
| UR5 | 6 EEF dims + gripper | 7 |

Key constants:

```python
HISTORY_TOKENS_PER_IMAGE = 16  # 每帧历史图像压缩为 4×4=16 个 token
HISTORY_POOL_SIZE = 4           # 2D 池化空间大小
HISTORY_PAD_TOKEN_ID = 7       # <unused1> token ID，用于无效历史槽
```

State description enum `RobotStateDesc`:
- `JOINT` — joint angles
- `EEF` — end-effector pose
- `GRIPPER` — gripper open/close

Action mode enum `ActionMode`:
- `ABSOLUTE` — absolute action
- `RELATIVE` — incremental action (relative to the current state)

### 1.2 Dataset Loading (`opendm/data/dataset.py`)

The `JsonlDataset` workflow:

1. Scan all `.jsonl` files under `jsonl_dir` (supports transparent access to local/S3 paths via `megfile`)
2. Build `index_cache.json` (per-file line counts) to speed up subsequent indexing
3. `__getitem__` loads a single frame's JSON and **additionally attaches the whole episode's `raw_lines`** (so that `BuildActionChunk` can read future frames afterwards)
4. Apply the transform pipeline

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

### 1.3 Normalization Statistics (`opendm/data/normalize.py`)

**Core algorithm — online quantile computation**:

`RunningStats` maintains online statistics. Key features:
- Welford online mean/variance
- **5000-bin histogram** for computing q01/q99 quantiles
- Histograms redistribute automatically when min/max expand (`_adjust_histograms`)

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

`NormStatsFile`: multi-robot statistics dispatch; `select(robot_type)` selects the matching configuration.

### 1.4 Transform Pipeline (`opendm/data/transforms.py`) — the most essential data file

**Full pipeline**:

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

Reads action values from the current frame plus `action_horizon` (usually 50) future frames, building a `[1, horizon, dim]` action chunk.

- With an `action` field: `read_key = "action"`, `start = frame_index`
- Without an `action` field: the future frames' `state` is used as the target, `read_key = "state"`, `start = frame_index + 1`
- Beyond the end of an episode, the last value is repeated

#### `ActionRelative` (L286-350)

Incremental encoding of `action - state`, **gripper dimensions keep their absolute values**:

```python
relative = action - state
# 夹爪维度保持原始绝对值
non_delta_indices = [i for i, sid in enumerate(state_desc) if sid in self.non_delta_ids]
relative[..., non_delta_indices] = action[..., non_delta_indices]
```

#### `Normalize` / `Denormalize` (L84-134)

**Quantile normalization** — clip-and-scale into [-1,1], not mean/std standardization:

```python
# 归一化
arr = np.clip(arr, q01, q99)
out = ((arr - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0)
return np.where((q01 == 0) & (q99 == 0), 0.0, out)  # 零范围维度输出0

# 反归一化
out = ((arr + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01)
```

#### `ChatTokenization` (L474-653)

Builds the Gemma3 chat template. Example output:

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

State discretization: `action_to_bin_tokens` maps normalized values in [-1,1] onto integer bin IDs in [0, 255].

#### `PadAction` (L656-676)

All actions are padded/truncated to `shared_dim=32`, realizing a unified multi-embodiment action space:

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

### 1.5 Image Augmentation (`opendm/data/augmentations.py`)

- **During training**: `PadToSquare → Resize(448,448) → RandomResizedCrop(0.95) → Rotate(±2°) → ColorJitter(b=0.3,c=0.4,s=0.3)`
- **During inference**: `PadToSquare → Resize(448,448)` (no augmentation)

### 1.6 Batch Collation (`opendm/data/collator.py`)

- `TrainingCollator`: pads `input_ids`/`attention_mask`/`token_type_ids` to `max_length=1024`, concatenates `pixel_values`/`action`/`action_mask`
- `NormStatsCollator`: groups by `robot_type`, computing aggregated state/action for the normalization statistics

### 1.7 Dataset Registration (`opendm/dataset/register.py`)

The dataset registry `CONVERSATION_DATA` automatically imports all `.py` modules in its directory and supports extension via the `$OPENDM_DATA_PATH` environment variable:

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

## Stage 2: Core Architecture — Dual-Stream VLA + Flow Matching

**Goal**: Gain a deep understanding of the DM05 model architecture.

### 2.1 Overall Architecture: Dual-Stream Design

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

**Key constraint**: the Action Expert must share the **same layer count / head dimensions / RoPE parameters** as the VLM language model (`validate_action_config_compatible`), so that the AE can directly consume the VLM's KV cache.

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

Inherits from `Gemma3TextModel`, adding time modulators on top:

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

### 2.4 AdaRMSNorm — the Core of Timestep Conditioning Injection (`dm05_arch.py` L283-302)

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

**How the time condition is built**:

```
time → posemb_sincos(dim) → time_mlp_in → SiLU → time_mlp_out → SiLU → adarms_cond
```

Sinusoidal-cosine positional encoding (`posemb_sincos`):
- `min_period = 4e-3`, `max_period = 256.0`
- Log-uniformly distributed frequencies: `period = min_period × (max_period/min_period)^fraction`

### 2.5 Per-Layer Computation in the Action Expert (`dm05_arch.py` L304-369)

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

**Core idea**: the AE never re-encodes the observation prefix! It simply concatenates the prefix cache along the KV dimension, allowing suffix tokens to attend over the whole prefix.

### 2.6 DM05Model — Model Assembly (`dm05_arch.py` L387-462)

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

### 2.7 Flow Matching Training Forward Pass (`dm05_arch.py` L743-837)

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

**Flow Matching mathematical formulation**:

$$x_t = t \cdot \epsilon + (1-t) \cdot a, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$u_t = \epsilon - a \quad \text{(target velocity field)}$$

$$\mathcal{L} = \text{MSE}(v_\theta(x_t, t), u_t) \cdot \text{action\_mask}$$

**Why use Beta(1.5, 1.0)?** Uniform sampling makes the model pay too much attention to the low-noise region near the ground-truth action, whereas Beta(1.5, 1.0) biases toward the high-noise end, helping the model learn the full trajectory from noise to action more effectively.

### 2.8 Euler Integration Inference (`dm05_arch.py` L843-922)

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

### 2.9 History Image Injection (`dm05_arch.py` L924-1007)

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

### 2.10 Suffix Attention Mask (`dm05_utils.py` L243-283)

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

### 2.11 Gradient-Checkpoint-Safe KV Cache (`dm05_utils.py` L102-139)

**Problem**: upstream `GradientCheckpointingLayer` clears `past_key_values` during recomputation, yet DM05's prefix forward pass needs to write the KV cache for reuse by the suffix.

**Solution**:
- `SafeCacheDecoderLayer`: overrides `__call__`, preserving `past_key_values` under gradient checkpointing
- `OverwriteDynamicLayer`: a cache layer with overwrite semantics; `update` assigns directly (preserving gradients)
- `VLADynamicCache`: dynamic cache based on `OverwriteDynamicLayer`
- `patch_decoder_layers`: monkey-patches the VLM decoder layers with replacements

---

## Stage 3: Training Pipeline and Experiment Configuration

**Goal**: Understand how the model / data / optimizer / trainer are assembled.

### 3.1 Configuration System: Nested tyro Dataclasses

```
DM05Exp
├── DM05ModelConfig     → 模型加载、注意力、梯度检查点、LoRA
├── DM05OptimizerConfig → adamw/muon_adamw 选择、学习率、Muon 超参
├── DM05TrainerConfig   → FSDP1/DDP、批大小、保存策略、W&B)  → Flask 服务、推理步数、快速后端
```

Every field is reachable from the CLI: `--model-config.chunk-size 50 --optimizer-config.optim muon_adamw`

Entry point:
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

### 3.3 Model Building Flow

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

Attention backend resolution rules:
- **LLM**: `auto` → `flex_attention` (requires PyTorch ≥2.5), otherwise `sdpa`
- **Vision**: `auto` → `flash_attention_2` (requires bf16+CUDA), otherwise `sdpa`
- **Action Expert**: `auto` → `flex_attention`, otherwise `sdpa`

### 3.4 Key Customizations in DMTrainer (`trainer.py`)

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

### 3.5 FSDP Configuration

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

### 3.6 LoRA Configuration (`dm05_lora.py`)

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

Alias expansion for `dm05_time_modulators`:
```python
# 展开为
"input_time_modulators.0", "input_time_modulators.1", ..., "input_time_modulators.N",
"mlp_time_modulators.0", "mlp_time_modulators.1", ..., "mlp_time_modulators.N",
"final_time_modulator"
```

LoRA loading at inference: `PeftModel.from_pretrained(base_model, adapter_path) → merge_and_unload()`

### 3.7 Data Pipeline Construction (`dm05_exp.py` L292-335)

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

### 3.8 Normalization Statistics Computation (`dm05_exp.py` L347-447)

- Computed on rank 0; other ranks poll and wait (checking every 5 seconds for the file to appear)
- Uses `NormStatsCollator` to group by robot_type
- The filename embeds a content-addressed hash: `{dataset_name}_{sha256(dataset_name|transform)[:16]}.json`

---

## Stage 4: MuonAdamW Optimizer

**Goal**: Understand the hybrid Muon/AdamW optimizer.

### 4.1 Parameter Selection Strategy

The selection logic of `is_default_muon_parameter`:

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

**Muon is applied only to the AE's Q/K/V/O projections and MLP weights**; all remaining parameters use standard AdamW.

### 4.2 Marking Parameters Before FSDP (`muon_adamw.py` L68-98)

```python
def mark_muon_parameters(model, predicate=None):
    """在 FSDP 包装前记录原始矩阵形状"""
    for name, param in model.named_parameters():
        if not predicate(name, param): continue
        setattr(param, "_opendm_muon_shape", tuple(param.shape))  # 记录原始形状
        setattr(param, "_opendm_muon_name", name)                  # 记录参数名
```

**Why must this happen before FSDP?** FSDP flattens parameters, losing the original matrix dimension information. The pre-recorded `_opendm_muon_shape` retains the original matrix dimensions so they can be reshaped for orthogonalization later.

### 4.3 Newton-Schulz Orthogonalization (`muon_adamw.py` L101-128)

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

**Intuition**: Muon's core idea is that "gradients should point in an orthogonal direction". Newton-Schulz iteration projects the gradient matrix onto its polar factor (the closest orthogonal matrix), thereby constraining parameter updates to lie near the orthogonal group.

**The quintic polynomial** `(a·I + b·G + c·G²) · M` is an iterative approximation of the zeroth power `M(M^T M)^{-1/2}`; the coefficients (3.4445, -4.7750, 2.0315) were tuned so that 5 iterations reach high precision.

### 4.4 MuonAdamW Optimizer Structure (`muon_adamw.py` L178-236)

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

### 4.5 Single Muon Update Step (`muon_adamw.py` L362-414)

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

**Purpose of the Moonlight scaling**: Muon's orthogonal update magnitude is independent of matrix dimension, while AdamW's update magnitude scales like `1/√dim`. The `√max_dim` scaling makes Muon's effective learning rate proportional to dimensionality, keeping update magnitudes comparable to AdamW across parameters of different dimensions.

### 4.6 FSDP Shard Planning (`muon_adamw.py` L443-535)

`_build_muon_shard_plan` classifies shards into 5 cases:

| Type | Meaning | `_gather_matrix` Behavior |
|------|------|----------------------|
| `empty` | No local data | Returns an empty tensor |
| `local` | Single GPU, all data local | Returns directly |
| `owner_only` | Under FSDP, only 1 rank holds the data | Returns directly |
| `partial` | Under FSDP, several ranks each hold part | Reconstructs the full matrix via `all_gather` |
| `replicated` | Every rank holds a full replica | Returns directly |

The `partial` path of `_gather_matrix`:
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

## Stage 5: Fast Inference — Triton + TensorRT + CUDA Graph

**Goal**: Understand how the training architecture translates into low-latency deployment.

### 5.1 Overall Architecture: Three Levels of Acceleration

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

### 5.2 TensorRT Vision Encoder (`dm05_infer/dm05_trt_utils.py`)

- The **SigLIP vision model** is exported to ONNX → compiled into a TensorRT FP16 engine
- History image support: the TRT engine processes `num_current + MAX_HISTORY_IMAGES` images at once
- History image features are pooled to 4×4=16 tokens via `pool_image_features_to_history`
- **Executed outside the CUDA Graph** (the TRT runtime is incompatible with CUDA Graphs)

```python
class DM05VisionTensorRTRunner:
    """TensorRT 视觉编码器执行器"""
    def __call__(self, pixel_values, output_tensor=None):
        """执行 TRT 引擎，输出图像特征"""
```

### 5.3 StaticPrefixCacheLayer — Address-Stable Cache (`dm05_infer.py` L31-113)

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

**Contrast with the training path's `OverwriteDynamicLayer`**:
- **Training**: `self.keys = key_states` (direct assignment, preserving gradients)
- **Inference**: `self.keys.copy_(key_states)` (copy, keeping addresses fixed; otherwise the CUDA Graph breaks)

### 5.4 CUDA Graph Capture Flow (`dm05_infer.py` L284-327)

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

### 5.5 Bucketed Prefix Length

Default buckets: `(576, 704, 768, 896, 1024)`

Selection strategy: `bisect.bisect_left(buckets, request_len)` — choose the smallest bucket ≥ the requested length.

**Why bucketing is needed?** CUDA Graphs pin all tensor shapes and addresses at capture time. Different prefix lengths mean different KV cache sizes and attention mask shapes, so a separate graph must be captured per shape. Bucketing discretizes the continuous prefix-length space into a finite set of buckets, avoiding a graph captured for every possible length.

### 5.6 Triton BigKernel (`dm05_infer/dm05_bigkernel.py`)

**Shape specialization** — compiled against a fixed AE architecture, avoiding dynamic-shape overhead:

```python
SUPPORTED_HIDDEN_SIZE = 1024
SUPPORTED_INTERMEDIATE_SIZE = 4096
SUPPORTED_HEAD_DIM = 256
SUPPORTED_Q_HEADS = 8
SUPPORTED_KV_HEADS = 4
```

**Preallocated buffers `DM05BigKernelBuffers`**:

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

**Fused kernels** — cross-referenced against the training code `_compute_suffix_layer`:

| Triton Fused Kernel | Corresponding Training Step |
|--------------|------------|
| Fused QKV+QK-norm+RoPE | Q/K/V projection → q_norm/k_norm → apply_rotary_pos_emb |
| Fused AdaRMS-norm | `_adaptive_rmsnorm`: RMSNorm + scale/shift + gate |
| Fused attention-post-norm-residual | O projection → post_attention_layernorm → residual+gate |
| Fused GeGLU-tanh | pre_ff_layernorm → MLP(GeGLU) → post_ff_layernorm → residual+gate |

### 5.7 Inference Path Selection

| Condition | Path | Latency |
|------|------|------|
| No history + prefix length ≤ max bucket | **CUDA Graph replay** (fastest) | ~15ms |
| No history + prefix length > max bucket | **Dynamic eager fallback** (no graph) | ~50ms |
| With history images | **TRT + uncaptured fast prefill/decode** (outside graph) | ~30ms |

### 5.8 fused_linear_euler_update (`dm05_utils.py` L26-52)

At inference, the Euler step is fused into a single `addmm` operation:

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

## Design Patterns Across the Whole System

### 1. Prefix/Suffix Separation

Every code path (training, inference, fast inference) follows the same pattern: the VLM prefix produces the KV cache → the AE suffix consumes it. This is the invariant of the architecture.

### 2. Multi-Embodiment Padding

The 32-dim action space + `action_mask` run through data, training loss, and inference:
- **Data**: `PadAction(32)` pads actions to 32 dims
- **Training**: `MSE(v_t, u_t) * action_mask` computes the loss only on valid dims
- **Inference**: `x_t = x_t * action_mask` masks invalid dims at every step

### 3. History Image Handling

The pool + scatter logic stays consistent across three places:
- `_compute_prefix_cache` (training/inference)
- `pool_image_features_to_history` (TRT utils)
- `ChatTokenization` (placeholder insertion)

### 4. Attention Backend Dispatch

`set_attention_implementation` resolves the backend separately for the three paths LLM/vision/AE:
- LLM: `flex_attention` (default) — composable attention masks
- Vision: `flash_attention_2` (default) — efficient FlashAttention
- AE: `sdpa` (default) — PyTorch Scaled Dot Product Attention

### 5. Configuration-Driven

All components are parameterized through `dataclass + tyro`, unifying CLI and programmatic control:
```bash
python exp.py --task train \
    --model-config.chunk-size 50 \
    --optimizer-config.optim muon_adamw \
    --data-config.dataset-name libero_goal \
    --inference-config.backend fast
```

---

## Core File Index

| File | Lines | Role |
|------|------|------|
| `opendm/model/dm05/dm05_arch.py` | 1079 | Central architecture: dual-stream model, Flow Matching training forward, Euler inference, prefix cache, AdaRMSNorm |
| `opendm/model/dm05/dm05_utils.py` | 283 | Utilities: VLADynamicCache, SafeCacheDecoderLayer, suffix attention mask, time embedding, fused Euler |
| `opendm/model/dm05/dm05_lora.py` | 395 | LoRA config and wrapping: alias expansion, target resolution, FSDP-compatible patches |
| `opendm/data/transforms.py` | 725 | Data pipeline: action chunking, normalization, tokenization, padding |
| `opendm/data/normalize.py` | 365 | Normalization statistics: RunningStats, NormStatsFile, multi-robot dispatch |
| `opendm/data/dataset.py` | 103 | Dataset: JsonlDataset, index cache |
| `opendm/data/collator.py` | 136 | Batch collation: TrainingCollator, NormStatsCollator |
| `opendm/optimizer/muon_adamw.py` | 535 | Hybrid optimizer: Newton-Schulz orthogonalization, FSDP shard planning |
| `opendm/exp/dm05_exp.py` | 1171 | Experiment orchestration: nested configs, assembly, training loop, inference service |
| `opendm/trainer/trainer.py` | 197 | Trainer: DMTrainer, optimizer injection, loss reduction |
| `opendm/infer/dm05_bigkernel.py` | 1405 | Triton fused kernels: shape specialization, preallocated buffers |
| `opendm/infer/dm05_infer.py` | 911 | CUDA Graph + static cache inference runtime |
| `opendm/infer/dm05_infer_arch.py` | 1366 | Fast-inference model architecture: graph-capturable variant |
| `opendm/infer/dm05_trt_utils.py` | 615 | TensorRT utilities: ONNX export, engine build/execution |
| `opendm/constants/robot.py` | 47 | Robot constants: RobotType, RobotStateDesc, history image config |

---

## Suggested Next Steps

1. **Run the demo inference for real**:
   ```bash
   script/dm05_launcher.sh --task inference
   tests/curl_demo.sh
   ```

2. **Read the playground experiment in depth**: `playground/dm05_sft_demo.py`, to understand how to customize experiment configurations

3. **Try tweaking hyperparameters**: adjust `chunk_size` or `diffusion_steps`, and observe the impact on inference quality

4. **Cross-reference the Triton kernels**: identify which step of the training code each fused operation in `dm05_bigkernel.py` corresponds to

5. **Trace the data flow manually**: start from the JSONL data under `assets/demo/` and walk fully through both the training and inference paths
