# Encoder-Only 半句续写语料生成说明

> 文档版本：2026-07-06  
> 目标：为 encoder-only XBridge（`Enc(src) + boundary + tgt`，无 prompt）构建英文续写训练数据，并用 Instruct 贪婪因果推理生成 `tgt` 标签，同时保留字面切分结果作对照。

---

## 1. 背景

### 1.1 训练格式

```
Enc(src) + boundary + tgt
```

- `src`：NLLB encoder 输入（`eng_Latn`）
- `tgt`：冻结 LLM 的因果续写目标（teacher forcing，仅 `aug=3` 算 loss）
- **无 `prompt`**，不用 `chat_template`

### 1.2 为何采用「半句续写」

曾尝试的方案及问题：

| 方案 | 问题 |
|------|------|
| `Repeat the following sentence exactly: {src}` + 贪婪续写 | Instruct 大量循环重复指令 |
| `{src} 的英文翻译为：` + 贪婪续写 | 英文 src 配中文提示，仍可能循环 |
| `src` 整句 + 贪婪续写作复述 | 扩写/翻译，非精准复述 |

**半句续写**更贴近因果 LM：给定上文前半，预测后半。

---

## 2. 数据流水线（两阶段）

```
opus100_zh_en_100k.jsonl（10 万英句 tgt）
        │
        ▼  scripts/build_continue_half_corpus.py
continue_en_half_50k.jsonl（5 万，字面切分）
        │
        ▼  scripts/build_continue_half_instruct_greedy.py
continue_en_half_50k_instruct.jsonl（5 万，Instruct 贪婪 tgt）
```

---

## 3. 阶段一：半句切分（对照集）

### 3.1 脚本

`scripts/build_continue_half_corpus.py`

### 3.2 切分规则

- 来源：`data/encoder_only/opus100_zh_en_100k.jsonl` 的 `tgt` 字段（英文）
- 在中点附近**最近空格**处切分
- 去重、过滤：原句 ≥8 字符，前后半各 ≥3 字符
- 输出上限：**50,000** 条

### 3.3 命令

```bash
python scripts/build_continue_half_corpus.py \
  --input data/encoder_only/opus100_zh_en_100k.jsonl \
  --output data/encoder_only/continue_en_half_50k.jsonl \
  --limit 50000
```

### 3.4 格式

```json
{
  "task": "continue",
  "src_lang": "eng_Latn",
  "src": "Sixty-first",
  "tgt": "session",
  "orig_tgt": "Sixty-first session"
}
```

| 字段 | 含义 |
|------|------|
| `src` | 原句前半（Enc 输入） |
| `tgt` | 原句后半（字面切分，可直接训练） |
| `orig_tgt` | 完整原句（对照：`src + 空格 + tgt` ≈ `orig_tgt`） |

### 3.5 状态

- **已完成**：`continue_en_half_50k.jsonl`，**50,000** 条

---

## 4. 阶段二：Instruct 贪婪续写生成 tgt

### 4.1 脚本

`scripts/build_continue_half_instruct_greedy.py`

### 4.2 生成逻辑

对每个 `continue_en_half_50k.jsonl` 样本：

```
prefix = src（仅前半句，无 prompt）
         ↓
Meta-Llama-3-8B-Instruct 贪婪因果续写（max_new_tokens=64）
         ↓
tgt = 新生成 token 解码结果（原样保存，无后处理）
```

**示例：**

```
src:  "Sixty-first"
tgt:  " session\nItem 123: Report of the Special Committee..."
```

续写发生在 `Sixty-first` **之后**，正确结果以 ` session` 开头，不重复 `src`。

### 4.3 命令

**单卡：**

```bash
python scripts/build_continue_half_instruct_greedy.py \
  --input data/encoder_only/continue_en_half_50k.jsonl \
  --output data/encoder_only/continue_en_half_50k_instruct.jsonl \
  --llm_path model/Meta-Llama-3-8B-Instruct \
  --batch_size 16 \
  --max_new_tokens 64
```

**多卡后台（4×5090，关闭窗口不中断）：**

```bash
# 启动 4 路分片（每卡一个 nohup 进程）
bash scripts/run_build_continue_half_instruct_multigpu.sh

# 查看进度
tail -f outputs/continue_instruct_shards/shard0.log

# 全部完成后合并
python scripts/merge_continue_instruct_shards.py
```

分片输出：`data/encoder_only/continue_en_half_50k_instruct.shard{0..3}.jsonl`  
日志 / PID：`outputs/continue_instruct_shards/`

参数（可选环境变量）：

```bash
NUM_GPUS=4 BATCH_SIZE=16 bash scripts/run_build_continue_half_instruct_multigpu.sh
```

断点续跑：各 shard 独立 `--resume`；若已有主文件 `continue_en_half_50k_instruct.jsonl`，会自动 `--also_skip_from` 跳过已生成 `src`。

### 4.4 输出格式

```json
{
  "task": "continue_greedy",
  "src_lang": "eng_Latn",
  "src": "Sixty-first",
  "tgt": " session\nItem 123: Report of...",
  "literal_tgt": "session",
  "orig_tgt": "Sixty-first session",
  "label_source": "Meta-Llama-3-8B-Instruct_greedy_raw",
  "max_new_tokens": 64
}
```

| 字段 | 含义 |
|------|------|
| `src` | 前半句（与阶段一相同） |
| `tgt` | **Instruct 贪婪续写**（训练标签） |
| `literal_tgt` | 阶段一字面后半（对照） |
| `orig_tgt` | 完整原句（对照） |

### 4.5 状态

- **生成中**：`continue_en_half_50k_instruct.jsonl`，目标 50,000 条
- 使用修复后脚本从头重新生成

---

## 5. 重要 Bug 与修复

### 5.1 现象

早期批量生成时，约 82% 的 `tgt` **看似重复了 `src`**：

```
src: "Sixty-first"
tgt: "Sixty-first session\n..."   ← 错误
```

### 5.2 原因

左 padding 批量推理时，切片位置错误：

```python
# 错误：用非 pad token 计数
cont_ids = out[i, attention_mask.sum():]

# 正确：用 pad 后整条 input 的长度
cont_ids = out[i, input_ids.shape[1]:]
```

`generate()` 在**整条 pad 后序列**末尾追加新 token；用 `attention_mask.sum()` 会误把 `src` 的一部分 decode 进 `tgt`。

### 5.3 修复后

```
src: "Sixty-first"
tgt: " session\n..."   ← 正确，只在后面续写
```

`Meta-Llama-3-8B` Base 与 Instruct 在正确切片下行为一致，均在前缀后接续。

### 5.4 处理

- 含 bug 的半成品已删除
- `build_continue_half_instruct_greedy.py` 已修复（2026-07-06）
- 必须用修复后脚本**全量重跑**

---

## 6. 训练用法

### 6.1 用字面切分（阶段一）

```bash
python train_encoder_only.py \
  --train_file data/encoder_only/continue_en_half_50k.jsonl \
  --llm_path model/Meta-Llama-3-8B-Instruct \
  --output_dir outputs/encoder_only_continue_half
```

### 6.2 用 Instruct 贪婪 tgt（阶段二）

```bash
python train_encoder_only.py \
  --train_file data/encoder_only/continue_en_half_50k_instruct.jsonl \
  --llm_path model/Meta-Llama-3-8B-Instruct \
  --output_dir outputs/encoder_only_continue_instruct
```

`train_encoder_only.py` 只读取 `src`、`tgt`、`prompt`（空）；`literal_tgt`、`orig_tgt` 仅用于评测对照。

### 6.3 训练序列

```
Enc(src) + boundary + tgt
aug: 1=encoder段, 3=tgt段（loss）
```

---

## 7. 相关文件一览

| 路径 | 说明 |
|------|------|
| `data/encoder_only/opus100_zh_en_100k.jsonl` | 原始 10 万 zh→en 英句来源 |
| `data/encoder_only/continue_en_half_50k.jsonl` | 5 万字面半句切分（对照/可直接训练） |
| `data/encoder_only/continue_en_half_50k_instruct.jsonl` | 5 万 Instruct 贪婪 tgt（主训练集） |
| `scripts/build_continue_half_corpus.py` | 阶段一：半句切分 |
| `scripts/build_continue_half_instruct_greedy.py` | 阶段二：Instruct 贪婪续写 |
| `outputs/build_continue_half_instruct.log` | 阶段二生成日志 |

### 已弃用 / 未完成

| 路径 | 说明 |
|------|------|
| `data/encoder_only/repeat_en_instruct_greedy.jsonl` | `Repeat exactly` 方案，约 3k 条后停用 |
| `scripts/build_repeat_corpus_instruct_greedy.py` | 同上方案脚本 |

---

## 8. 评测对照建议

生成完成后，可抽样对比：

1. **`tgt` vs `literal_tgt`**：Instruct 续写是否贴近真实后半句
2. **`src + tgt` vs `orig_tgt`**：前缀 + 续写能否还原整句
3. **字面切分 vs Instruct 标签**：哪种训练 FLORES / 下游效果更好

---

*关联文档：[instruct-encoder-only-plan.md](./instruct-encoder-only-plan.md)*
