# OpenClaw CoRLA

本目录按仓库根目录的 [ICLR2027.pdf](../ICLR2027.pdf) 第 3 节及 Algorithm 1
实现 **CoRLA（Conformal Reward and Long-horizon Attribution）**。
目录名和 rollout 入口保留 `ualca`，方便已有部署迁移。

## 与旧 UALCA 版本的区别

| 部分 | 旧实现 | 当前实现及论文位置 |
| --- | --- | --- |
| Judge | PRM 投票均值、标准差 | 冻结初始骨干 + critic LoRA + 两输出 MLP；`mu=tanh(m)`，`sigma=sqrt(sigma_min²+softplus(v))`；式 4–5 |
| 教师查询 | 每一步多个 judge | 初始数据 warm start；每轮选最大 sigma 的 K 个新样本；新旧标签回放各半；式 6 |
| 校准 | 投票残差滚动分位数 | 独立随机教师探针，只用于校准；每次 judge 更新后重新评分保存的输入；式 7–9 |
| Potential | 短窗口可靠回报之差，未来项会抵消 | 学习历史前缀的终局折扣回报，加区间外的平方惩罚；真终止步骤不施加区间惩罚；式 10–11 |
| Gate | uncertainty、投票熵和一致性的指数门控 | `omega = 1 - clip(b/R, 0, 1)`；不裁剪区间端点，不依赖 mu；式 12 |
| Credit | 局部均值 − 风险惩罚 + 门控 credit | `c_t = r_env_t + beta_phi * omega_t * (gamma*Phi_(t+1)-Phi_t)`；式 13 |
| Advantage | 每一步一个 shaped reward | 从后向前累计 `A_t = c_t + gamma*lambda_a*A_(t+1)`，不再减第二个 baseline；式 14 |
| OPD | top-k 词表、多个 hint、独立 PRM 蒸馏模型 | 仅 `b <= tau_opd` 时提取一个 actionable hint；由**本轮冻结的 rollout policy**评分已采样 token；式 15–16 |
| Policy | 原有 Hybrid GRPO/top-k loss | RL 与 OPD 分别 PPO 裁剪；按 token 求和、按轨迹平均，乘 `gamma^t`；仅更新 policy LoRA；式 17–18 |

旧的 `LOOKAHEAD`、`BETA_RISK`、`LAMBDA_CREDIT`、`BETA_SIGMA`、`BETA_D`、
`BETA_C`、`TAU_C`、`W_MIN`、`SCORE_CLIP` 及 `OPENCLAW_TOPK_*` 配置不再参与本方法。
特别注意：`TAU_OPD` 现在是**区间半宽上限**，不是 gate 下限。

## 模型和训练流程

1. 用初始策略的教师标注数据训练 judge，并留出独立校准集。
2. 每轮固定 rollout policy，收集指定数量的完整轨迹或显式截断轨迹。
3. 根据 sigma 选择 judge 查询，再从剩余样本随机选择校准探针。
4. 更新 judge、重新校准，缓存均值、区间和 gate。
5. 更新共享 critic adapter 和 potential head；judge head 固定，缓存区间不变。
6. 重新预测 potential，缓存 credit 和 advantage。
7. 仅在可靠步骤提取 hint；在同一 rollout policy 下分别评分原上下文和增强上下文。
8. 更新 policy LoRA，再将权重同步到 SGLang，开始下一轮。

策略和 critic 使用同一初始模型的**独立物理副本**，骨干都冻结；critic 的两个 head
共享 critic LoRA。这等价于分开保存两个适配器，但会额外占用一份骨干显存。
保留仓库现有 FSDP 的初始化、保存和 SGLang 权重同步代码；通过本目录的 actor 子类
实现 CoRLA 损失，不修改 Slime 的通用训练代码。

使用专用同步入口 `train_corla.py`。旧 `train_async.py` 会提前采集下一轮，不能保证这里
要求的策略版本一致。每轮按轨迹累积梯度并做一次策略更新；policy dropout 在计算
概率比时关闭。为控制显存，potential 更新逐 transition 累积完整轨迹目标；这是对论文
目标的完整 batch 求值，而不是第 4.1 节描述的 128-transition 随机 mini-batch。

默认还计算第 4.1 节的完整词表 `KL(pi_current || pi_initial)`，系数 0.01。
初始策略由关闭 policy LoRA 得到；参考 logits 暂存 CPU，损失按 token 分块重计算。

## 启动

使用本仓库配置好的 Slime / FSDP / SGLang 环境，并安装 `transformers`、`peft`、
`torch`、`ray`、`fastapi`、`httpx` 等训练依赖。默认 GPU 分配：policy 4、rollout 2、
冻结 PRM 1、critic 1，共 8 张。critic 的独立 Ray actor 需要一张未被其他角色占用的 GPU。

```bash
export HF_CKPT=/path/to/Qwen3-4B
export PRM_MODEL_PATH=/path/to/frozen-Qwen3-4B
export OPENCLAW_UALCA_BOOTSTRAP_PATH=/path/to/bootstrap.jsonl
export SAVE_CKPT=/path/to/checkpoints/corla
bash openclaw-ualca/run_qwen3_4b_openclaw_ualca.sh
```

脚本会复用已有 Ray 集群，或启动新集群。它不会终止已有 Python/Ray 进程。
可用 `NUM_GPUS`、`ACTOR_GPUS`、`ROLLOUT_GPUS`、`ROLLOUT_TP`、`PRM_GPUS`、`PRM_TP`
调整资源，用 `NUM_ROLLOUT`、`ROLLOUT_BATCH_SIZE`、`SAVE_INTERVAL` 调整轮次。

### Bootstrap 数据

JSONL 每行格式如下，`ids` 是初始策略完整 transition `h_t, a_t, o_(t+1)` 的 token IDs，
`label` 是冻结教师按固定进度评分协议给出的标签，范围 `[-1,1]`：

```json
{"ids": [151644, 872, 198, 1234, 151645], "label": 1.0}
```

示例 token 仅说明格式。实际数据必须使用 `HF_CKPT` 的 tokenizer 和与在线阶段一致的
chat template，包含动作后的观测，末尾使用 `add_generation_prompt=True`。
在线评分协议见 `ualca_api_server.py::_teacher_prompt`，PRM 使用 greedy decoding。
默认建议准备 4,096 条训练数据和 1,024 条校准数据；代码随机打散去重后，留出
`BOOTSTRAP_CALIBRATION` 条校准，其余训练。校准输入不会进入 judge regression 或 replay。
同一输入存在冲突标签时直接报错，不通过平均标签隐藏冲突。

### 客户端必须提供终局反馈

普通生成仍走 `/v1/chat/completions`，每条轨迹使用独立的 `session_id` 或
`X-Session-Id`。为匹配论文的概率比，代理将生成温度和 top-p 设为 1，单次生成最多
4,096 tokens。每次生成传入完整历史，最后一个 user/tool 消息作为前一动作的 next-state。

最后一步动作执行后，调用独立接口；这次调用**不会生成新动作**：

```http
POST /v1/ualca/feedback
Authorization: Bearer <与代理相同的 API key>
Content-Type: application/json

{
  "session_id": "task-001",
  "terminated": true,
  "terminal_reward": 1.0,
  "messages": [
    {"role": "user", "content": "请完成任务"},
    {"role": "assistant", "content": "已提交最终结果"},
    {"role": "user", "content": "环境验证通过"}
  ]
}
```

`messages` 必须包含真实完整历史，包括最后动作及其反馈；上面只是格式示例。
`terminal_reward` 必须来自环境验证器或真实结果，成功可为 1、失败可为 0，不能使用
PRM 分数代替。训练不会把“缺少终局标签”自动当成失败。

如果是时间或步数限制导致的**非终止截断**，传 `terminated: false` 并省略
`terminal_reward`。此时保留末状态 potential 的 bootstrap 值，不生成虚构终局标签，
也不使用该轨迹监督终局回报回归；它仍可参与区间约束和策略更新。

旧 `session_done` 请求会收到明确的迁移错误。新接口保留最后一步动作，同一轮内重复
提交完成反馈不会重复入队。每轮最多接受 `ROLLOUT_BATCH_SIZE` 条轨迹；达到名额后，
已有会话可继续完成，新会话及训练期间请求收到 503，客户端应等待重试。必须为所有
已启动轨迹发送终止或截断反馈，训练轮才能结束。客户端负责环境的步数和总 token 上限。

## 主要参数

所有参数使用 `OPENCLAW_UALCA_` 前缀，默认值集中在 `UALCAConfig`：

| 后缀 | 默认值 | 作用 |
| --- | --- | --- |
| `ALPHA` | 0.1 | 教师标签区间的误覆盖目标 |
| `GAMMA`, `LAMBDA_A` | 0.99, 0.95 | 环境折扣、反向信用 trace |
| `LAMBDA_CP`, `BETA_PHI` | 0.1, 1.0 | 区间约束、potential 差权重 |
| `TAU_OPD`, `OPD_CLIP`, `LAMBDA_OPD` | 0.5, 2.0, 0.1 | 半宽阈值、token target 裁剪、OPD 权重 |
| `KL_COEF` | 0.01 | 对冻结初始策略的完整 forward KL 权重 |
| `SIGMA_MIN`, `HUBER_DELTA` | 0.001, 1.0 | Judge 正 scale 下限、Huber 参数 |
| `JUDGE_QUERIES`, `CALIBRATION_QUERIES` | 64, 64 | 每轮标签预算；K 还受 `floor(N/2)` 限制 |
| `CALIBRATION_WINDOW`, `CALIBRATION_ROUNDS` | 2048, 32 | 保留的探针数和轮数 |
| `REPLAY_WINDOW` | 10000 | Judge 标签回放容量 |
| `JUDGE_UPDATES`, `POTENTIAL_UPDATES` | 5, 5 | 每轮辅助更新次数 |
| `BOOTSTRAP_UPDATES`, `BOOTSTRAP_CALIBRATION` | 200, 1024 | Warm start 更新次数、留出数 |
| `BATCH_SIZE`, `HEAD_HIDDEN` | 128, 1024 | Judge mini-batch 大小、MLP 隐层大小 |
| `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` | 16, 32, 0.05 | 适配器配置 |
| `LR`, `WARMUP_ROUNDS` | 1e-6, 20 | 学习率、线性 warm-up |

校准分位数使用 `ceil((n+1)*(1-alpha))`，包含 `n+1` 位置的正无穷哨兵。
样本不足时得到无界区间：门控增量为零，OPD 关闭，终局奖励继续参与学习。
日志用 `null` 和 `unbounded_interval` 表示无界区间。教师查询失败不会伪造为 0 标签。
覆盖率在新随机探针上使用**更新前**的预测和上一分位数衡量；它是教师标签的经验覆盖率，
不是对真实奖励或自适应 RL 下分布无关覆盖率的保证。

## 断点恢复

策略保存沿用 Slime FSDP 格式。匹配的 critic adapter、两个 head、optimizer、随机状态、
replay、探针及分位数保存在 `SAVE_CKPT/corla/round_000099/` 等目录。
例如完成第 100 次更新后，策略为 `iter_0000100`，critic 为 `round_000099`：

```bash
export LOAD_CKPT=/path/to/checkpoints/corla
export OPENCLAW_UALCA_RESUME=$LOAD_CKPT/corla/round_000099
bash openclaw-ualca/run_qwen3_4b_openclaw_ualca.sh
```

`LOAD_CKPT` 下的策略 tracker 必须对应同一轮，且保持相同 CoRLA 配置。
恢复时检查策略和 critic 的下一轮编号是否一致；critic 目录单独存在不代表策略更新成功。

## 文件与检查

| 文件 | 职责 |
| --- | --- |
| `ualca_signals.py` | 校准区间、门控、终局回报和 trace |
| `ualca_loss.py` | Judge、potential、PPO、OPD、forward KL 目标 |
| `ualca_critic.py` | HF critic LoRA、head 和优化器 |
| `ualca_training.py` | Algorithm 1、数据隔离、校准和 checkpoint |
| `ualca_api_server.py` | 会话收集、终局反馈、教师查询和策略评分 |
| `ualca_rollout.py`, `ualca_data.py` | 完整轨迹到缓存训练目标及 DP 分发 |
| `ualca_actor.py`, `train_corla.py` | FSDP 策略更新及同步训练驱动 |

CPU 检查（需要 PyTorch，不需要模型权重或 GPU）：

```bash
python -m unittest discover -s openclaw-ualca -p 'test_ualca*.py' -v
bash -n openclaw-ualca/run_qwen3_4b_openclaw_ualca.sh
```

测试覆盖公式、stop-gradient、辅助 head 交替更新、轨迹终止/截断、校准隔离、teacher
失败、checkpoint、并发会话和 token 对齐。外部 HTTP / Ray 服务在 CPU 集成测试中使用
替身；这些检查不替代真实 Qwen3 / SGLang / 多卡 FSDP 的训练验证，也不证明论文中的指标。
论文第 4.1 节还将新增超参数标为 provisional defaults，这里采用其给出的默认值。
