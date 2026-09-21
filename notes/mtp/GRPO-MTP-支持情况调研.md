# verl 项目中 GRPO 训练 + MTP（Multi-Token Prediction）支持情况调研

> 调研对象：`/Users/williamvvli/SourceCode/AI_Infra/verl-playground`（字节跳动火山引擎 Verl 项目）
> 调研目标：确认整个项目中**同时开启 GRPO 训练与训练侧 MTP** 的代码究竟位于何处
> 调研日期：2026-09-21

---

## 0. 结论速览（TL;DR）

1. **你找到的 `examples/mtp_trainer/` 是准确的，但不完整。** 该目录是 MTP 的"官方主入口/推荐示例"，但它只是**启动脚本层**；真正实现 MTP 训练的代码在 `verl/` 库内部。
2. **MTP 训练是 Megatron-Only 的。** 在全部训练引擎（fsdp / megatron / mindspeed / torchtitan / veomni / automodel）中，**只有 `verl/workers/engine/megatron/transformer_impl.py` 引用 MTP**。FSDP、torchtitan、veomni 等引擎代码中完全不存在 MTP 处理逻辑。因此 **GRPO + MTP 只能是 `model_engine=megatron` + `actor_rollout_ref.actor.megatron.use_mbridge=True`**。
3. **除了 `examples/mtp_trainer/`，仓库里还有 2 个真正启用 GRPO + MTP 训练的脚本**，位于 `verl/experimental/fully_async_policy/shell/`：
   - `grpo_qwen35_35b_megatron_async.sh`（`enable_train=True`，真训练 MTP）
   - `run_qwen35_35b_a3b_math_dynamic_megatron.sh`（`enable_train=False`，只加载 MTP 参数 + rollout 投机解码）
4. **`main_ppo.py` 本身不含任何 MTP 代码**。MTP 完全通过配置项 `actor_rollout_ref.model.mtp.*` 流动：`HFModelConfig.mtp` → `check_mtp_config()` 注入 `override_transformer_config` → Megatron 引擎 → `patch_engine_mtp()` 打补丁。GRPO 与 MTP 是**正交**的两条链路，通过同一个 PPO/GRPO trainer 汇合。
5. GRPO 相关的其余示例（`examples/grpo_trainer/*`）要么**显式关闭 MTP**（`mtp.enable=False`），要么在注释里声明暂不支持。

---

## 1. MTP 配置的唯一权威定义

### 1.1 配置 dataclass

文件：`verl/workers/config/model.py`

```29:67:verl/workers/config/model.py
@dataclass
class MtpConfig(BaseConfig):
    """
    Configuration for MTP model.

    enable: Enable loading and saving of MTP parameters, but do not use them

    enable_train: Whether to enable using MTP parameters during training
    enable_rollout: Whether to enable using MTP parameters during rollout

    Training parameters:
        detach_encoder: Whether to detach encoder parameters during MTP training
        mtp_loss_scaling_factor: Loss scaling factor during MTP training

    vLLM rollout parameters:
        method: "mtp"
        num-speculative-tokens: 1

    SGLang rollout parameters:
        speculative-algorithm: EAGLE
    ...
    """
    enable: bool = False
    enable_train: bool = False
    enable_rollout: bool = False

    detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1

    speculative_algorithm: str = "EAGLE"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4

    method: str = "mtp"
    num_speculative_tokens: int = 1
```

挂载点：

- `verl/workers/config/model.py:26` — `__all__ = ["HFModelConfig", "MtpConfig"]`
- `verl/workers/config/model.py:146` — `mtp: MtpConfig = field(default_factory=MtpConfig)`（`HFModelConfig`）
- `verl/workers/config/model.py:85` — `"mtp"` 被加入 `_mutable_fields` 集合
- `verl/workers/config/rollout.py:23` — rollout 侧复用同一个 `MtpConfig`
- `verl/workers/config/rollout.py:270` — `mtp: MtpConfig = field(default_factory=MtpConfig)`

### 1.2 关闭 MTP 时的自动清零逻辑（关键分支）

```234:244:verl/workers/config/model.py
        # When MTP is disabled, zero out MTP layer counts from hf_config so that
        # downstream engine/worker code does not need to handle each MTP field format
        # individually. Supports both DeepSeek-style (num_nextn_predict_layers) and
        # Qwen3.5-style (mtp_num_hidden_layers, possibly nested under text_config).
        if not self.mtp.enable:
            if hasattr(self.hf_config, "num_nextn_predict_layers"):
                self.hf_config.num_nextn_predict_layers = 0
            if hasattr(self.hf_config, "mtp_num_hidden_layers"):
                self.hf_config.mtp_num_hidden_layers = 0
            if hasattr(self.hf_config, "text_config") and hasattr(self.hf_config.text_config, "mtp_num_hidden_layers"):
                self.hf_config.text_config.mtp_num_hidden_layers = 0
```

### 1.3 YAML 默认值

- `verl/trainer/config/model/hf_model.yaml:80-98`（`# MTP` 段，`_target_: verl.workers.config.MtpConfig`）
- `verl/trainer/config/rollout/rollout.yaml:478-479`：`mtp: ${oc.select:actor_rollout_ref.model.mtp, null}`
- 生成态配置（`_generated_ppo_*.yaml`）中均可见 `mtp:` 段：
  - `verl/trainer/config/_generated_ppo_megatron_trainer.yaml:439,474-486,736-748,871`
  - `verl/trainer/config/_generated_ppo_trainer.yaml:424,459-471,700-712,835`
  - torchtitan / veomni 的 generated yaml 也含该字段，但**引擎侧不消费**（见 §3）

> 注意：`recipe/` 目录下**没有任何 MTP 引用**（全文检索 0 命中）。

---

## 2. GRPO + MTP 的完整调用链（训练侧）

GRPO 与 MTP 的汇合点如下（箭头方向为数据/配置流动方向）：

```
启动脚本  examples/mtp_trainer/*.sh
   └─ algorithm.adv_estimator=grpo            ← GRPO 由这里决定
   └─ actor_rollout_ref.model.mtp.*           ← MTP 由这里决定
        │
        ▼
  verl/trainer/main_ppo.py  （★ 无任何 MTP 代码，纯透传配置）
        │
        ▼
  verl/trainer/ppo/ray_trainer.py           （★ 无 MTP 训练代码）
  verl/trainer/ppo/v1/trainer_base.py       （仅消费 rollout 侧 spec-decode 指标）
        │
        ▼
  verl/workers/engine_workers.py
     :51     import MtpConfig
     :223-227 聚合 mtp_losses* 指标（跨 DP 求平均）
     :553-557 ref 模型强制关闭 MTP：ref_config.model_config.mtp = MtpConfig(enable=False)
        │
        ▼
  verl/workers/engine/megatron/transformer_impl.py   ★ MTP 训练实现核心
        │
        ▼
  verl/utils/megatron_utils.py  →  verl/models/mcore/mtp_patch.py
        │
        ▼
  Megatron-Core 的 MultiTokenPredictionLayer / MTPLossLoggingHelper
```

### 2.1 训练侧关键代码点（`verl/workers/engine/megatron/transformer_impl.py`）

| 行号 | 作用 |
|---|---|
| `66-77` | 导入 `check_mtp_config` / `get_megatron_mtp_loss` / `patch_engine_mtp` |
| `229-232` | NPU(>=0.16.0) 时 `apply_mtp_inference_patch()` |
| `290` | `check_mtp_config(self.model_config, self.engine_config)` — 把 `mtp.*` 注入 `override_transformer_config` |
| `377-385` | `deepseek_v4` 且 `not mtp.enable` 时强制 `mtp_num_layers=0` 并裁剪 `csa_compress_ratios` |
| `526-532` | MTP 启用时**禁用 fused kernels**（`use_fused_kernels=False`） |
| `590-591` | `if self.model_config.mtp.enable: patch_engine_mtp(self.module, self.model_config)` |
| `592-599` | forward_only 且 `mtp_num_layers==0` 时只 `patch_postprocess` |
| `975-983` | `if mtp.enable and mpu.is_pipeline_last_stage(...)` → `metrics = get_megatron_mtp_loss(n_micro_batch)` |
| `1387-1413` | 计算 `mtp_loss_normalization_factor`（依赖 `mtp.enable and mtp.enable_train and calculate_per_token_loss`），并传 `mtp_enable_train` 给 forward |

### 2.2 配置校验与补丁装配（`verl/utils/megatron_utils.py`）

四个分支的完整逻辑：

```1850:1881:verl/utils/megatron_utils.py
def check_mtp_config(model_config: HFModelConfig, engine_config: McoreEngineConfig):
    """
    Check and configure MTP (Multi-Token Prediction) settings.

    Cases:
        - mtp.enable == False and no MTP layers: force provider MTP config to None
        - mtp.enable == False and has MTP layers: clear HF MTP fields and force provider MTP config to None
        - mtp.enable == True and no MTP layers: raise ValueError
        - mtp.enable == True and has MTP layers: configure override_transformer_config
    """
    ...
    if not enable_mtp:
        _set_mtp_num_layers(hf_config, 0)
        engine_config.override_transformer_config["mtp_num_layers"] = None
        engine_config.override_transformer_config.pop("mtp_loss_scaling_factor", None)
        return
    elif enable_mtp and not has_mtp:
        raise ValueError("enable mtp while model has no mtp layer, please use a model with mtp layer")
    elif enable_mtp and has_mtp:
        if "mtp_loss_scaling_factor" not in engine_config.override_transformer_config:
            engine_config.override_transformer_config["mtp_loss_scaling_factor"] = (
                model_config.mtp.mtp_loss_scaling_factor
            )
    return
```

```1884:1906:verl/utils/megatron_utils.py
def patch_engine_mtp(module, model_config):
    ...
    modules = module if isinstance(module, list) else [module]
    for m in modules:
        patch_postprocess(m)
        patch_mtp_layer_checkpointed_forward(m)
        if model_config.mtp.detach_encoder:
            patch_mtp_layer_get_embeddings(m)
```

其余工具函数：

- `verl/utils/megatron_utils.py:1771-1795` — `get_megatron_mtp_loss(n_micro_batch)`，从 MCore tracker 取 `mtp_losses/mtp_N_loss`
- `verl/utils/megatron_utils.py:1823-1837` — `_get_mtp_num_layers()`，兼容 `num_nextn_predict_layers`（DeepSeek/Qwen3）与 `mtp_num_hidden_layers`（Qwen3.5，含 `text_config` 嵌套）
- `verl/utils/megatron_utils.py:1840-1847` — `_set_mtp_num_layers()`
- `verl/utils/megatron/router_replay_utils.py:576-583` — router replay 遍历时**跳过 MTP 层**（MTP 层编号从 1 开始，避免与 decoder 层别名冲突）

### 2.3 核心补丁实现（`verl/models/mcore/mtp_patch.py`）

这是 MTP 训练的"真正实现"，约 548 行：

| 行号 | 函数 | 作用 |
|---|---|---|
| `82-90` | `patch_postprocess` / `unpatch_postprocess` | 替换 `GPTModel._postprocess` |
| `97-258` | `_megatron_gptmodel_postprocess` | 调用 `self.mtp(...)` 并计算 MTP loss；**兼容新旧两代 MCore API**（`process_mtp_loss` 新版 vs. 手工 `roll_tensor` + `functional_call` 旧版） |
| `57-67` | `_get_mtp_loss_config` | Dynamic CP / `calculate_per_token_loss` 下按 routed token 数缩放 `mtp_loss_scaling_factor` |
| `46-54` | `_resolve_cp_group` | 支持 per-microbatch Dynamic CP group |
| `261-299` | `patch_mtp_layer_get_embeddings` | **`detach_encoder=True` 的关键**：patch `MultiTokenPredictionLayer._get_embeddings` |
| `300-343` | `patch_mtp_layer_checkpointed_forward` | 支持 recompute（激活重计算）下的 MTP forward |
| `418-486` | `_patched_get_embeddings_for_detach` | detach 掉 token embedding 与主干 hidden states，使 MTP loss 只更新 MTP 模块参数 |
| `487-547` | `_patched_checkpointed_forward` | 把非 tensor 参数排除出 checkpoint 保存的 tensor（参考 THUDM/slime 的 megatron patch） |

### 2.4 前向与 loss mask 对齐

`verl/models/mcore/model_forward.py`：

- `:22` `from verl.workers.config import MtpConfig`
- `:49` `mtp_config: MtpConfig = None`
- `:72` `mtp_enable_train = mtp_config and mtp_config.enable_train`
- `:213-228` `_build_mtp_loss_mask_nested(...)` — 构造与 `[prompt; response]` 对齐的 MTP loss_mask
- `:274, :280` `mtp_enable_train` / `mtp_loss_normalization_factor` 参数
- `:305,317,325,358`（thd 路径）与 `:411,416,426`（bshd 路径）分别处理

`verl/models/mcore/model_forward_1f1b_overlap.py`（1F1B overlap 场景的独立 MTP 实现）：

- `:71` `mtp_in_postprocess=None`
- `:100-115` `if mtp_in_postprocess: hidden_states = self.mtp(...)`
- `:119-157` 内置 MTP loss（`chunk(hidden_states, 1 + config.mtp_num_layers)`、`MTPLossLoggingHelper.save_loss_to_tracker`、`mtp_loss_scaling_factor / mtp_num_layers`）

### 2.5 建模 / 权重转换 / Checkpoint

| 文件 | 行号 | 内容 |
|---|---|---|
| `verl/models/mcore/model_initializer.py` | `:73, :85` | `mtp_block_spec=mtp_block_spec` 传入 `GPTModel` |
| `verl/models/mcore/model_initializer.py` | `:190-196` | `if self.tfconfig.mtp_num_layers > 0:` → `get_gpt_mtp_block_spec(...)` |
| `verl/models/mcore/weight_converter.py` | `:382-402` | `_convert_mtp_param()`：`mtp.layers.0.*` → `model.layers.61.*` 映射（enorm/hnorm/eh_proj/final_layernorm→shared_head.norm） |
| `verl/models/mcore/weight_converter.py` | `:412-413` | `if "mtp" in name: return self._convert_mtp_param(...)` |
| `verl/models/mcore/config_converter.py` | `:256-304` | Qwen3.5 MoE：`mtp_num_hidden_layers` → `transformer_config.mtp_num_layers` |
| `verl/models/mcore/config_converter.py` | `:335-339` | 该路径断言 `num_nextn_predict_layers == 0`（"MTP is not supported for now"） |
| `verl/models/mcore/config_converter.py` | `:383-386` | DeepSeek：`mtp_num_layers = hf_config.num_nextn_predict_layers`、`mtp_loss_scaling_factor = 0.1` |
| `verl/models/mcore/patch.py` | `:577-591` | `apply_mtp_inference_patch()`（NPU 推理路径） |
| `verl/models/mcore/patch.py` | `:645-663` | recomputation backward 补丁中识别并跳过 MTP 层 checkpoint |

---

## 3. 关键结论：MTP 训练是 Megatron-Only

### 证据 A：引擎目录全文检索

对 `verl/workers/engine/*/transformer_impl.py` 检索 `mtp|MTP`，**仅 `megatron/transformer_impl.py` 命中**。以下引擎文件均无任何 MTP 逻辑：

- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/engine/torchtitan/transformer_impl.py`
- `verl/workers/engine/veomni/transformer_impl.py`
- `verl/workers/engine/mindspeed/transformer_impl.py`
- `verl/workers/engine/automodel/transformer_impl.py`

### 证据 B：官方文档明确声明

```11:11:docs/advance/mtp.md
- **Training Engine**: Only supports the `mbridge/Megatron-Bridge + megatron` combination; other training engines are not compatible at this time;
```

```3:3:examples/mtp_trainer/README.md
MTP uses an auxiliary token-prediction head (speculative / draft head) during training. Currently supported on MiMo-7B-RL with Megatron backend.
```

### 证据 C：示例脚本必备参数

所有 GRPO+MTP 脚本都同时设置：

```bash
actor_rollout_ref.actor.megatron.use_mbridge=True
model_engine=megatron        # 或 config-name='fully_async_ppo_megatron_trainer.yaml'
```

因此判定条件为：**GRPO + MTP 训练 = `algorithm.adv_estimator=grpo` + `actor_rollout_ref.model.mtp.enable=True` + `actor_rollout_ref.model.mtp.enable_train=True` + Megatron 引擎 + `use_mbridge=True`**。

---

## 4. Rollout 侧 MTP（投机解码，可选、独立于训练）

Rollout 侧 MTP 通过 `mtp.enable_rollout=True` 单独控制，**与 `enable_train` 相互独立**（见 `docs/advance/mtp.md` §2 表格）。

### 4.1 vLLM 路径

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
  - `:69` `from ... import build_mtp_speculative_config`
  - `:377-382` `if self.config.mtp is not None and enable and enable_rollout:` → 设置 `args["speculative_config"]`
  - `:741-749` spec-decode 指标映射（`request_spec_decode_stats`）
  - `:1300-1308` `_resolve_sleep_level()`：MTP rollout 启用时 sleep level=1（避免 level2 丢弃 drafter 权重）
- `verl/workers/rollout/vllm_rollout/utils.py`
  - `:218-221` `_use_mtp_drafter_weight_sync()`（`spec.method == "mtp"`）
  - `:229-239` 权重同步时把 MTP drafter 一并 yield
  - `:355-397` buffer / 量化权重同步到 drafter
  - `:522-534` `build_mtp_speculative_config(method, num_speculative_tokens, ...)`
- `verl/utils/vllm/vllm_quant_utils.py:465-474` — `disable_mtp_completeness_check`

### 4.2 SGLang 路径

- `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
  - `:407-418` `enable and enable_rollout` 时设置 `speculative_algorithm / speculative_num_steps / speculative_eagle_topk / speculative_num_draft_tokens`，要求 sglang>=0.5.6，并开启 `enable_weights_cpu_backup` / `enable_draft_weights_cpu_backup`
  - `:714-719` 回填 `spec_num_draft_tokens / spec_num_accepted_tokens / spec_num_verify_steps`

### 4.3 指标聚合（GRPO trainer 侧）

- `verl/trainer/ppo/v1/trainer_base.py:1902-1917` — 仅当 `mtp.enable and mtp.enable_rollout` 时抓取 spec 指标
- `verl/trainer/ppo/v1/trainer_base.py:1964` — `metrics.update(compute_spec_decode_metrics(...))`
- `verl/trainer/ppo/ray_trainer.py:138-184` — `compute_spec_decode_metrics()`，输出 `rollout/spec_accept_rate`、`rollout/spec_accept_length`
- `verl/trainer/ppo/ray_trainer.py:1792-1799` — 每请求 spec-decode 指标聚合
- `verl/workers/engine_workers.py:223-227` — `mtp_losses*` 跨 DP 平均

---

## 5. GRPO + MTP 启动脚本全景清单

### 5.1 主入口：`examples/mtp_trainer/`（你已找到，推荐入口）

| 脚本 | adv_estimator | 训练侧 MTP | Rollout | 模式 |
|---|---|---|---|---|
| `run_mimo_7b_mtp_megatron.sh` | `grpo` (`:51`) | `enable=True`, `enable_train=True`, `detach_encoder=True`, `factor=0.1` (`:66-69`) | SGLang（无 spec） | Sync hybrid-engine |
| `run_mimo_7b_mtp_rl_vllm_sgl_megatron.sh` | `grpo` (`:64`) | `enable=True`, `enable_train=True`, `detach_encoder=True`, `factor=0.2` (`:79-82`) | SGLang 或 vLLM，可选 `enable_rollout=True` (`:126-144`) | Sync，对齐 slime/EAGLE |
| `run_mimo_7b_mtp_fully_async_megatron_multinode.sh` | `grpo` (`:83`) | `enable=True`, `enable_train=True`, `detach_encoder=True`, `factor=0.1`, **`enable_rollout=True`** (`:88-92`) | SGLang spec | Fully-async split-placement（DAPO 配置） |

> `main_ppo.py` 只被前两个脚本调用；第三个调用 `verl.experimental.fully_async_policy.fully_async_main`。
> README 把第三个标为 "DAPO"，实际是 `adv_estimator=grpo` + DAPO 的 clip/reward 配置（`clip_ratio_low/high` + `reward_manager=dapo`），本质仍是 GRPO 家族。

### 5.2 补充入口：`verl/experimental/fully_async_policy/shell/`（★ 容易被漏掉）

| 脚本 | adv_estimator | MTP 配置 |
|---|---|---|
| `grpo_qwen35_35b_megatron_async.sh` | `grpo` (`:32`) | `mtp_params` 于 `:98-104`：`enable=True`、**`enable_train=True`**、`factor=0.1`、`detach_encoder=True`、`enable_rollout=True`；SGLang 追加 `speculative_algorithm=NEXTN` 等（`:106-113`）；`:252` 注入 |
| `run_qwen35_35b_a3b_math_dynamic_megatron.sh` | GRPO（动态调度） | `:28-38`：`enable=True`、**`enable_train=False`**、`enable_rollout=True` + spec 参数；`:132` 注入 |

**这是除 `examples/mtp_trainer/` 之外唯一真正把 MTP 训练跑起来的 GRPO 脚本组。**

### 5.3 明确"不启用"或"不支持" MTP 的 GRPO 示例（排除项）

| 脚本 | 情况 |
|---|---|
| `examples/grpo_trainer/run_qwen3_8_27b_megatron.sh` | `:126` `actor_rollout_ref.model.mtp.enable=False`；`:224` `override_transformer_config.mtp_num_layers=0` — **显式关闭** |
| `examples/grpo_trainer/run_deepseek_v3_671b_megatron.sh` | `:9-10` 注释 `set num_nextn_predict_layers=0 (MTP not yet supported)` |
| `examples/ascend_extras/grpo_trainer/run_glm5_2_megatron.sh` | `:134` `actor_rollout_ref.actor.checkpoint.strict=False # MTP layers are unused ... omitted from exported HF weights` |
| `examples/sft/gsm8k/run_mimo_7b_mtp_megatron.sh` | MTP 但为 **SFT**，非 GRPO（`:92 model.mtp.enable=True`） |

### 5.4 文档

- `docs/advance/mtp.md` — **官方 MTP 权威指南**（作者 meituan-search）：支持范围、三类训练配置、实验结果、rollout 性能说明、SFT 说明
- `docs/index.rst`、`docs/blog/v0.7.md`、`docs/perf/dpsk.md`、`docs/advance/deepseek_v4_integration.rst`、`docs/hardware/multi_chip_support.rst`、`docs/ascend_tutorial/**` 亦有 MTP 提及
- `examples/README.md:120` — `| mtp_trainer/ | DAPO + MTP (MiMo-7B) | adv_estimator=grpo, MTP flags |`

---

## 6. 相关测试清单

| 测试文件 | 覆盖点 |
|---|---|
| `tests/utils/test_megatron_mtp_dcp.py` | `test_process_mtp_loss_uses_dynamic_group_and_token_scale`(`:64`)、`test_detached_embeddings_roll_with_dynamic_group`(`:94`)、`test_mtp_metric_scale_matches_mcore_tracker`(`:129`) |
| `tests/special_distributed/test_megatron_dynamic_cp_features.py` | `test_mtp_roll_and_backward_use_each_microbatch_cp_group`(`:134`) — 构造 `mtp_num_layers=1`(`:180`) |
| `tests/workers/rollout/rollout_vllm/test_mtp_hybrid_sleep_acceptance_on_cpu.py` | `test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance`(`:52`) |
| `tests/workers/rollout/test_vllm_weight_update_utils_on_cpu.py` | `test_vllm_update_weights_syncs_buffers_to_mtp_drafter`(`:220`) |
| `tests/utils/megatron/test_router_replay_model_walk_on_cpu.py` | `test_mtp_routers_are_not_addressed`(`:128`) |
| `tests/utils/test_vllm_weight_name_normalization_on_cpu.py` | `test_drafter_drops_base_layer_wholesale`(`:569`) |
| `tests/models/test_model_forward_fused.py` | `mtp_num_layers=0` 场景(`:171`) |
| `tests/special_sanity/check_example_naming.py` | `:91-93` 将 `examples/mtp_trainer` 列入命名豁免 |

> **无端到端 GRPO+MTP 的集成测试**；现有测试均为单元/组件级（CPU 或单机分布式）。

---

## 7. 依赖版本要求（来自 `docs/advance/mtp.md` §1）

- **mbridge**：需 PR [ISEEKYAN/mbridge#62](https://github.com/ISEEKYAN/mbridge/pull/62)（已合入 main）
- **Megatron-Bridge**：MiMo-7B-RL 需 PR [NVIDIA-NeMo/Megatron-Bridge#2387](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2387)
- **megatron**：需最新 dev 版（commit `23e092f41ec8bc659020e401ddac9576c1cfed7e`，支持 MTP + CP）。若额外开 `recompute_granularity=full`，需含 [NVIDIA/Megatron-LM#3457](https://github.com/NVIDIA/Megatron-LM/pull/3457)（`ffd66a3e6`）；否则 `MultiTokenPredictionLayer.forward` 会因 `padding_mask` 关键字引发 `TypeError`。已发布的 `megatron-core` 0.18.0 / 0.18.2 均不含该修复（[#4933](https://github.com/NVIDIA/Megatron-LM/issues/4933)）
  > 代码侧已做兼容：`mtp_patch.py:135-138` 用 `signature()` 检测 `padding_mask` 参数是否存在，`mtp_patch.py:28-35` 检测 `process_mtp_loss` 是否可用
- **sglang**：需分支 [ArronHZG/sglang `fix_mtp_update_weights_from_tensor`](https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor)（PR [sgl-project/sglang#17870](https://github.com/sgl-project/sglang/pull/17870)）
- **vLLM**：`vllm_quant_utils.py:465-474` 提供 `disable_mtp_completeness_check` 以放宽校验

**模型前提**：下载 MiMo-7B-RL 后需手动把 `config.json` 的 `max_position_embeddings` 改为 `32768`（见 `examples/mtp_trainer/README.md:13`）。

---

## 8. 配置场景对照表（GRPO 视角）

| 场景 | `enable` | `enable_train` | `detach_encoder` | `enable_rollout` | `mtp_loss_scaling_factor` | 说明 |
|---|---|---|---|---|---|---|
| 仅加载 MTP 参数（导出用） | True | False | — | False | — | 显存增加，导出权重含 MTP 模块 |
| **全参数 MTP 训练** ★效果显著 | True | True | False | False | 0.1 | MTP Loss 作用到全部模型参数 |
| **MTP 参数独立训练**（推荐） | True | True | True | False | 0.1 | 冻结 Encoder，只更新 MTP 模块 |
| MTP 加速 Rollout | True | False | — | True | — | vLLM: `method=mtp`,`num_speculative_tokens=1`；SGLang: EAGLE 系列参数 |
| **训练 + Rollout 全开** | True | True | True | True | 0.1 | `examples/mtp_trainer/run_mimo_7b_mtp_fully_async_megatron_multinode.sh` |

> `docs/advance/mtp.md:64-70` 结论：只有"基础模型带 MTP 参数 + MTP Loss 作用全参数 + `factor=0.1`"才有显著效果；官方**推荐 `detach_encoder=True`**。

---

## 9. 对你初始判断的回答

> 你的原话："我已经初步找到一份支持 MTP 的代码，目录在这里 `examples/mtp_trainer`。但我不确定自己找得是否准确。"

**准确度评估：找对了入口，但需要补充三层认识。**

| 判断 | 结论 |
|---|---|
| `examples/mtp_trainer/` 是否支持 GRPO + MTP？ | ✅ **是**。3 个脚本全部 `adv_estimator=grpo`，且全部开启 `mtp.enable=True` + `mtp.enable_train=True` |
| 它是否涵盖项目全部 GRPO+MTP 支持？ | ❌ **不是**。另有 `verl/experimental/fully_async_policy/shell/grpo_qwen35_35b_megatron_async.sh`（MTP 真训练）与 `run_qwen35_35b_a3b_math_dynamic_megatron.sh` |
| 它是不是"实现代码"？ | ⚠️ **它是启动脚本层**。真正实现位于 `verl/workers/engine/megatron/transformer_impl.py` + `verl/utils/megatron_utils.py` + `verl/models/mcore/mtp_patch.py` 等 |
| MTP 训练是否 Megatron 独占？ | ✅ **是**。仅 `model_engine=megatron` + `use_mbridge=True` 可用 |
| GRPO 与 MTP 是否在 trainer 层耦合？ | ⚠️ **否**。`main_ppo.py` / `ray_trainer.py` 无 MTP 训练代码，二者通过配置正交汇合于 `engine_workers.py` → Megatron 引擎 |

---

## 10. 快速定位索引（按需查代码用）

| 想了解的内容 | 去这里 |
|---|---|
| MTP 配置字段与默认值 | `verl/workers/config/model.py:29-67, 146` |
| MTP YAML 默认 | `verl/trainer/config/model/hf_model.yaml:80-98` |
| 配置校验与注入 | `verl/utils/megatron_utils.py:1850-1881` |
| 补丁装配入口 | `verl/utils/megatron_utils.py:1884-1906` |
| MTP loss 计算（核心） | `verl/models/mcore/mtp_patch.py:97-258` |
| detach_encoder 实现 | `verl/models/mcore/mtp_patch.py:261-299, 418-486` |
| 引擎侧接线 | `verl/workers/engine/megatron/transformer_impl.py:290, 590-599, 975-983, 1387-1413` |
| loss mask 对齐 | `verl/models/mcore/model_forward.py:213-228` |
| 1F1B overlap 场景 | `verl/models/mcore/model_forward_1f1b_overlap.py:100-157` |
| 权重转换 | `verl/models/mcore/weight_converter.py:382-413` |
| HF→MCore 配置转换 | `verl/models/mcore/config_converter.py:256-304, 383-386` |
| Rollout 侧 spec 配置 | `verl/workers/rollout/vllm_rollout/utils.py:522-534`、`verl/workers/rollout/sglang_rollout/async_sglang_server.py:407-418` |
| 指标上报 | `verl/workers/engine_workers.py:223-227`、`verl/trainer/ppo/ray_trainer.py:138-184` |
| 官方使用指南 | `docs/advance/mtp.md` |
