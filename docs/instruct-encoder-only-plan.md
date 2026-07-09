# Encoder-Only XBridge：切换 Llama-3-8B-Instruct 修改方案

> 目标：在 **仅训练 `mapping_enc2llm`** 的前提下，用 NLLB encoder 将低资源语言语义对齐到 **冻结的 Llama-3-8B-Instruct** 空间；训练阶段仍用 **因果续写（teacher forcing）**，推理阶段改用 **官方 `apply_chat_template`**，以验证「语义对齐 + Instruct 对话格式」能否零样本迁移 LLM 的英文推理能力。

---

## 1. 背景与动机

### 1.1 当前方案（Base 模型）

```
低资源语 x → NLLB Encoder (frozen) → mapping_enc2llm (trainable)
  → boundary (learnable sep) → [可选 prompt] → Llama-3-8B Base (frozen) → 英文输出
```

- **训练**：仅 `L_LLM`（`dec_lambda=0`, `ot_lambda=0`, `llm_only=True`），平行语料 x→en 续写。
- **推理**：手写 Alpaca 模板（`### Instruction` / `### Response`），非 Instruct 官方格式。
- **LLM**：`model/Meta-Llama-3-8B`（Base，未 chat-tune）。

### 1.2 已有基线结果

| 评测 | 指标 | 结果 | 参考 |
|------|------|------|------|
| FLORES x→en（encoder-only） | Avg BLEU | **37.27** | 论文 36.21；zh→en **26.54** |
| MGSM 零样本 zh | Acc | **3.2%** | XBridge-SFT **56.8%** |
| MGSM 零样本 en | Acc | **5.6%** | XBridge-SFT **64.8%** |

**结论**：翻译语义对齐有效，但 Base + Alpaca prompt **几乎无法**零样本迁移数学推理；Stage-2 任务微调仍是主要增益来源。

### 1.3 本次改动核心假设

- Instruct 模型在预训练/对齐阶段已习得「遵循 user 指令 → assistant 续写」的模式。
- 若 `mapping_enc2llm` 将低资源语 embedding 对齐到 Instruct 的语义空间，推理时使用 **与 Instruct 一致的 chat 格式**，可能比 Base + Alpaca 更好地触发已有能力。
- **不引入 Stage-2 任务训练**，仅改 LLM 底座与 prompt 格式。

---

## 2. 目标架构

```
低资源语 x → NLLB Encoder (frozen) → mapping_enc2llm (trainable, ~21M)
  → boundary → [chat user 段] → Llama-3-8B-Instruct (frozen) → assistant 续写
```

| 组件 | 变更 |
|------|------|
| LLM | `Meta-Llama-3-8B` → **`Meta-Llama-3-8B-Instruct`**（冻结） |
| 训练损失 | 不变：仅 `L_LLM`，mapping 可训练 |
| 训练数据 | 不变：`data/encoder_only/opus100_zh_en_100k.jsonl` 等 x→en 平行语料 |
| 推理 prompt | Alpaca 手写 → **`tokenizer.apply_chat_template`** |

**模型路径**（已存在）：

- `model/Meta-Llama-3-8B-Instruct`
- Token ID 与 Base 一致：`bos=128000`, `eos=128001`, `pad=128002`

---

## 3. 两种实现方案

### Plan A：最小改动（训练不变，仅改推理）

| 阶段 | 序列结构 |
|------|----------|
| **训练** | `Enc(x)` + `boundary` + **英文 tgt**（无 chat 模板） |
| **推理** | `Enc(x)` + `boundary` + **chat_template(user=问题)** → 生成 assistant |

**优点**：改动小，可快速对比 Base vs Instruct 的推理增益。  
**缺点**：训练/推理格式不一致（train-infer mismatch），MGSM 等任务收益可能受限。

### Plan B：推荐方案（训练与推理格式一致）

| 阶段 | 序列结构 |
|------|----------|
| **训练（翻译）** | `Enc(x)` + `boundary` + **user 模板 tokens** + **assistant 前缀** + **英文 tgt** |
| **训练（loss）** | 仅对 **assistant/tgt 段**（`aug=3`）计算 `L_LLM` |
| **推理（MGSM 等）** | `Enc(x)` + `boundary` + **chat_template(messages=[{role:user,...}])** → 生成 assistant |

**翻译训练示例**（概念）：

```python
messages = [
    {"role": "user", "content": "Translate the following to English:\n{src}"},
    {"role": "assistant", "content": "{tgt}"},
]
# apply_chat_template(..., add_generation_prompt=False) 用于训练
# loss 仅覆盖 assistant content + eos
```

**MGSM 推理示例**：

```python
messages = [{"role": "user", "content": question}]
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # 追加 assistant 开头
    return_tensors=None,
)
# 序列: Enc(zh_question) + boundary + prompt_ids → generate
```

**优点**：与 Instruct 预训练分布更一致，利于零样本任务迁移。  
**缺点**：需改 `collate_fn` 与数据字段；翻译任务需在 JSONL 或 collate 中构造 user 文案。

---

## 4. 训练流程（Plan B 详细）

### 4.1 数据格式

现有 JSONL 字段（`train_encoder_only.py`）：

```json
{"src_lang": "zho_Hans", "src": "...", "tgt": "...", "prompt": ""}
```

- **纯翻译**（`prompt` 为空）：在 collate 中动态生成 chat user 段，例如  
  `"Translate the following text to English:\n{src}"`，tgt 作为 assistant content。
- **带任务 prompt**（MGSM 等，仅用于可选实验）：`prompt` 字段填入 user content，tgt 为 assistant。

### 4.2 Collate 序列与 augmentation

沿用现有 `aug` 语义（`modeling_xbridge.py`）：

| aug 值 | 含义 |
|--------|------|
| 0 | padding |
| 1 | MT encoder 段（`Enc(x)`，经 mapping） |
| 2 | prompt 段（chat user 模板 token） |
| 3 | label 段（assistant 续写 / tgt，参与 loss） |

Plan B 下 `collate_fn` 逻辑：

1. `mt_ids` ← NLLB tokenizer(`src`)
2. `chat_ids` ← `apply_chat_template` 得到的 **user + assistant 前缀**（或整条模板中 user 部分 + generation prompt 前缀）
3. `tgt_ids` ← assistant content + `eos`
4. `seq = mt_ids + chat_user_ids + tgt_ids`（具体切分以实现为准）
5. `aug`：mt→1，user 模板→2，tgt→3

### 4.3 超参与脚本

```bash
/home/yrh/.conda/envs/xbridge/bin/python train_encoder_only.py \
  --train_file data/encoder_only/opus100_zh_en_100k.jsonl \
  --llm_path model/Meta-Llama-3-8B-Instruct \
  --mt_path model/nllb-200-1.3B \
  --output_dir outputs/encoder_only_instruct \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_epochs 3 \
  --use_chat_template true   # 新增参数（Plan B）
```

- `dec_lambda=0`, `ot_lambda=0` 已在 `build_model` 中写死。
- 精度：BF16（已与 FP16 对比，BLEU 差异 ~+0.5，可接受）。

---

## 5. 推理流程

### 5.1 翻译（FLORES x→en）

**Plan A**：

```
Enc(x) + boundary + 直接续写英文（与现 eval_encoder_only.py 相同）
```

**Plan B / MGSM**：

```python
def build_chat_prompt(tokenizer, user_content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
```

拼接：`mt_ids + chat_prompt_ids`，`aug` 中 prompt 段为 2。

### 5.2 零样本任务（MGSM）

- **user content**：可直接用英文题干（与现方案一致），或中文题干 + 英文指令 wrapper。
- **移除** Alpaca 字符串：

```text
Below is an instruction that describes a task...
### Instruction: ...
### Response: Let's think step by step.
```

- 改用 Instruct 官方模板后，CoT 触发语可写入 user content，例如：  
  `"Solve the following math problem step by step.\n\n{question}"`

---

## 6. 需修改的文件

| 文件 | 修改内容 |
|------|----------|
| `train_encoder_only.py` | 默认 `llm_path` → Instruct；新增 `--use_chat_template`；Plan B 下 collate 调用 `apply_chat_template` |
| `scripts/merge_encoder_only_ckpt.py` | `llm_path` 指向 Instruct；合并后 config 更新 |
| `inference_xbridge_stage2_and_3.py` | `llm_input_features` 改为 `apply_chat_template`；移除硬编码 Alpaca |
| `scripts/eval_encoder_only_zeroshot_tasks.py` | 同上；`MGSM_PROMPT` → chat messages |
| `scripts/eval_encoder_only.py` | 翻译评测可选 chat user 包装（Plan B 时与训练一致） |
| `scripts/prepare_encoder_only_data.py` | `MGSM_INSTRUCTION` 改为 chat user 文案（非 Alpaca） |
| `model/XBridge-base/config.json` | 新 checkpoint 的 `llm_path` 改为 Instruct 路径（训练产出后） |

**无需修改**（仅换加载路径）：

- `modeling_xbridge.py`：forward / loss 逻辑已支持 `aug` 分段。
- NLLB encoder 与 `mapping_enc2llm` 结构不变。

---

## 7. 评测协议

训练完成后，与 **Base + Alpaca 基线** 对比：

| 评测项 | 脚本 | 主要指标 | 基线 |
|--------|------|----------|------|
| FLORES x→en | `scripts/eval_encoder_only.py` | BLEU（含 zh→en） | Avg 37.27 |
| MGSM 零样本 zh | `scripts/eval_encoder_only_zeroshot_tasks.py` | Accuracy | 3.2% |
| MGSM 零样本 en | 同上 | Accuracy | 5.6% |
| 可选：官方推理脚本交叉验证 | `inference_xbridge_stage2_and_3.py` | MGSM zh | ~2.8% |

**对照实验矩阵**（建议）：

1. Base + Alpaca（已有）
2. Instruct + Alpaca（Plan A 推理，旧训练）
3. Instruct + chat template，Plan A 训练
4. Instruct + chat template，Plan B 训练（主实验）

---

## 8. 预期结果

| 评测 | 预期 |
|------|------|
| FLORES BLEU | 与 Base 接近（±1 BLEU）；语义对齐主要由 mapping 决定，LLM 底座影响次要 |
| MGSM 零样本 | **高于 3%** 有可能（Instruct + 正确 chat 格式）；**仍可能远低于 SFT ~57%**（无任务微调） |
| 风险 | Plan A train-infer mismatch 可能限制增益；需 Plan B 验证 |

---

## 9. 实施步骤清单

- [ ] **1. 确认模型**：`model/Meta-Llama-3-8B-Instruct` 可正常加载（已完成）
- [ ] **2. 实现 Plan B collate**：`train_encoder_only.py` + `--use_chat_template`
- [ ] **3. 更新推理/评测**：`apply_chat_template` 替换 Alpaca（3 个脚本）
- [ ] **4. 冒烟训练**：`max_steps=2` 验证 loss 下降、checkpoint 可保存
- [ ] **5. 全量训练**：`opus100_zh_en_100k.jsonl`，输出 `outputs/encoder_only_instruct`
- [ ] **6. 合并 checkpoint**：`scripts/merge_encoder_only_ckpt.py` → 可 `from_pretrained` 的目录
- [ ] **7. 评测**：FLORES + MGSM，写入 `outputs/encoder_only_instruct_eval/`
- [ ] **8. 对比分析**：与 `outputs/encoder_only_zeroshot/metrics.json` 基线制表

---

## 10. 关键代码参考

### 10.1 当前训练 collate（`train_encoder_only.py`）

```python
# mt_ids + prompt_ids + tgt_ids
# aug: 1=mt, 2=prompt, 3=tgt
seq = mt_ids + prompt_ids + tgt_ids
```

### 10.2 当前 MGSM Alpaca prompt（待替换）

`scripts/eval_encoder_only_zeroshot_tasks.py`：

```python
MGSM_PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{question}\n\n"
    "### Response: Let's think step by step."
)
```

### 10.3 模型 forward 拼接顺序（`modeling_xbridge.py`）

```
<bos> // mt_hidden_state (src) // sep (boundary) // prompt // label (tgt)
```

loss 仅作用于 `label` 段（`aug=3`）。

---

## 11. 不在本次范围内

- Stage-2 / Stage-3 任务微调（SFT）
- 训练 `mapping_llm2dec` 或 NLLB decoder（`dec_lambda` 保持 0）
- 多语言 OPUS 三语句对 + OT 损失（论文完整 Stage-1）
- 更换 NLLB 或增大 mapping 层数

---

*文档版本：2026-07-06 · 对应仓库 `XBridge-for-multilingual-extension-main`*
