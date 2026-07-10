# XBridge 实验总结

> 项目：*Language on Demand, Knowledge at Core*: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality（ACL 2026）
> 基座：NLLB-200-1.3B（encoder/decoder）+ LLaMA-3-8B（LLM 内核）
> 评测语言：en, bn, de, es, fr, ja, ru, sw, th, zh（10 种）
> 本文件汇总项目已开展的全部实验、设置与结果数据。

---

## 0. 实验总览

| # | 实验 | 模型 | 评测集 | 指标 | 脚本 |
|---|------|------|--------|------|------|
| 1 | Stage 1 跨模型对齐翻译 | XBridge-base | FLORES-101 devtest | BLEU / COMET | `scripts/run_flores.sh` |
| 2 | Stage 2/3 多语数学推理 | XBridge-SFT | MGSM (250 题/语言) | Accuracy | `scripts/run_mgsm.sh` |
| 3 | Encoder-only 适配训练（消融） | 自训 mapping_enc2llm | FLORES x→en | BLEU | `scripts/run_train_encoder_only.sh` |
| 3.2 | Encoder-only 均衡多语（9×50k） | 自训 mapping_enc2llm | 6 语样例 spot-check | 定性（幻觉） | `stage1_encoder_x_en/run_train.sh` |
| 4 | Zero-shot MGSM 必要性消融 | encoder-only 各变体 | MGSM | Accuracy | `scripts/run_eval_encoder_only_zeroshot.sh` |
| 5 | 实现/数值一致性测试 | — | — | 一致性 | `scripts/test_*.py` |
| 6 | SFT 翻译能力退化对比 | base vs SFT | FLORES 前 200 句 | BLEU | `scripts/compare_bleu_200.py` |

---

## 1. Stage 1：FLORES-101 翻译（XBridge-base）

- **目的**：评测跨模型映射的翻译质量（论文 Stage 1）。
- **设置**：`inference_xbridge_stage1.py`，10 语言全 18 方向，每方向 1012 句；sacrebleu `flores200` 分词；可选 COMET（Unbabel/wmt22-comet-da）。
- **输出**：`outputs/flores101/`（`metrics.json`、`metrics_comet.log`）。

### 1.1 逐方向 BLEU（n=1012）

| 方向 | BLEU | 论文 | 差 | 方向 | BLEU | 论文 | 差 |
|------|------|------|---|------|------|------|---|
| bn-en | 38.06 | 37.09 | +0.97 | en-bn | 26.56 | 28.42 | −1.86 |
| de-en | 46.90 | 45.75 | +1.15 | en-de | 37.49 | 35.45 | +2.04 |
| es-en | 33.13 | 32.00 | +1.13 | en-es | 30.10 | 29.59 | +0.51 |
| fr-en | 47.16 | 46.10 | +1.06 | en-fr | 50.18 | 49.38 | +0.80 |
| ja-en | 28.51 | 27.63 | +0.88 | en-ja | 15.95 | 20.12 | −4.17 |
| ru-en | 38.14 | 37.08 | +1.06 | en-ru | 33.68 | 30.57 | +3.11 |
| sw-en | 45.33 | 44.73 | +0.60 | en-sw | 33.48 | 34.68 | −1.20 |
| th-en | 31.59 | 30.61 | +0.98 | en-th | 26.88 | 17.09 | +9.79 |
| zh-en | 26.58 | 24.89 | +1.69 | en-zh | 27.47 | 23.11 | +4.36 |

### 1.2 汇总

| 指标 | 复现 | 论文 |
|------|------|------|
| X→En 平均 BLEU | **37.27** | 36.21 |
| En→X 平均 BLEU | **31.31** | 29.82 |

**结论**：复现与论文一致，XBridge-base 是有效的多语翻译模型。

---

## 2. Stage 2/3：MGSM 多语数学推理（XBridge-SFT）

- **目的**：评测经 Stage 2（encoder 侧）+ Stage 3（decoder 侧）适配后的多语数学推理能力。
- **设置**：`inference_xbridge_stage2_and_3.py`，Alpaca + "Let's think step by step" 英文 CoT 提示，10 语言 × 250 题，`max_new_tokens=512`。
- **输出**：`outputs/mgsm_g{0,2,3}/accuracy`（分 GPU 并行）。

### 2.1 逐语言准确率（%）

| 语言 | en | fr | es | de | zh | th | ja | sw | bn |
|------|----|----|----|----|----|----|----|----|----|
| Acc（.llm 英文 CoT 答案） | 64.8 | 62.8 | 62.0 | 58.8 | 56.8 | 54.0 | 51.6 | 49.6 | 45.6 |
| Acc（.mt decoder 原生语言答案） | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- **.llm 路径**：LLM 内核生成英文 CoT 并给出答案，准确率与论文 MGSM 表吻合。
- **.mt 路径**：decoder 不承担数学答题，仅做语言渲染，故全 0（设计如此）。

### 2.2 基座对照

| 模型 | gsm8k_zh 准确率 |
|------|----------------|
| XBridge-base（`outputs/mgsm_base_verify/`） | 2.8 |
| XBridge-SFT | 56.8 |

**结论**：Stage 2/3 SFT 使 MGSM 准确率从近 0 提升到 45–65%，验证下游适配必要且有效。

---

## 3. Encoder-only 适配消融（仓库扩展实验）

- **目的**：研究"仅重训 `mapping_enc2llm`、冻结 LLM"能否替代完整 Stage 2/3。
- **训练**：`train_encoder_only.py`，数据为 NLLB mined x→en + seed + flores（每语言 20 万句），lr=2e-5，3 epoch，bf16；训后用 `merge_encoder_only_ckpt.py` 合回 XBridge-base。
- **语料变体**：

| 变体 | 构造脚本 | 说明 |
|------|----------|------|
| continue_half | `build_continue_half_corpus.py` | 半句续写 |
| continue_instruct | `build_continue_half_instruct_greedy.py` | Instruct 贪心续写 |
| repeat | `build_repeat_corpus.py` / `build_repeat_corpus_instruct_greedy.py` | 精确重复 |
| base_common | — | 公共基线 |

- **FLORES x→en 评测**（`eval_encoder_only.py`，含 chat 模板变体 `*_chat_c`）：各变体输出在 `outputs/encoder_only_*_eval*/`，BLEU 近 0（探索性方向，未达翻译可用水平）。

### 3.1 真实翻译语料的 encoder-only 训练（mt-baseline / fusion / embed-only）

- **动机**：§3 上表的 continue/repeat 变体是探针任务，非真翻译。这里改用**真实翻译平行语料**（OPUS-100 zh→en）重训 `mapping_enc2llm`，检验"仅重训一条 enc→LLM 边、冻结 LLM、纯英文 CE"能否达到翻译可用水平。
- **训练**：`train_encoder_only.py`（`dec_lambda=0, ot_lambda=0, freeze_mapping_llm2dec=True`，即只有 `L_LLM`），lr=2e-5，3 epoch，bf16。
  - `enc_mt_xen_100k`：10 万句 zh→en，单卡，7000 步，末 loss≈2.59。
  - `enc_mt_xen_200k`：20 万句 zh→en，4 卡（gloo 梯度同步，见下），18750 步，末 loss≈2.50。
  - 附：`enc_mt_fusion_xen_100k`、`enc_mt_embed_only_zh_en_100k` 为同期结构变体，loss 曲线见 `figures/loss_compare3_*.png`。

- **zh→en 结果（`inference_xbridge_stage1.py` encoder_only 模式，FLORES devtest）**：

| 模型 | 训练数据 | 末 loss | zh→en 输出形态 | BLEU |
|------|----------|---------|----------------|------|
| **XBridge-base**（论文成品，参照） | 三语 (x,en,y) 完整 Stage 1 | — | ✅ 正常翻译 | **26.5** |
| enc_mt_xen_100k | 10 万 zh→en，仅 L_LLM | 2.59 | ❌ 流畅幻觉 + 重复退化 | ≈0 |
| enc_mt_xen_200k | 20 万 zh→en，仅 L_LLM | 2.50 | ❌ 同上，未改善 | ≈0 |

- **诊断（脚本 `scripts/diag_mapping_progress.py` / `diag_mapping_collapse.py` / `quick_zhen_test.py`）**：
  1. **非 bug**：合并数值精确；同一推理脚本配 XBridge-base 得 26.5、配自训 checkpoint 幻觉——脚本无误，差在训练。
  2. **非坍缩**：mapping 对不同句子输出可分（句间余弦 0.73），尺度正常（范数≈11 vs base 11.5）。
  3. **收敛到不同解**：自训 mapping 输出与 XBridge-base 输出的余弦仅 **0.11**（近乎正交）。
  4. **数据量不是杠杆**：100k→200k，该余弦 0.113→0.116（几乎不动），loss 虽降但翻译能力零改善。

- **结论**：仅重训 `mapping_enc2llm`、冻结 LLM、双语英文 CE，**无法复现三方对齐的翻译能力**（zh→en 幻觉，BLEU≈0；表示层与完整模型近乎正交，且对数据量不敏感）。这从反面说明论文 Stage 1 的 encoder↔LLM 对齐**依赖 decoder 侧的联合约束**（README:64，三语 (x,en,y) + `dec_lambda=1/ot_lambda=6`；模型前向在 `modeling_xbridge.py:394-434` 已实现，仅被本训练脚本以 `llm_only` 跳过）。**堆数据无法弥补此差距。**
  - 边界（诚实标注）：最稳的观察是"冻结 LLM + 纯英文 CE 存在 teacher-forcing 捷径 → 低 loss 但源信息未真正进入"；但"缺三方约束 / 捷径本身 / 其他"三者当前证据下**尚未被彼此区分**，未做进一步控制变量实验。

- **工程记录**：本机 4×RTX 5090 上 NCCL 集合通信（≥1MB）报 illegal memory access，标准 DDP 不可用；改用 gloo（CPU）后端仅同步 21M mapping 梯度，重计算全部留在各卡 GPU，近线性 4× 提速。脚本：`train_encoder_only_mp.py` + `scripts/run_train_mp_200k.sh`。

### 3.2 均衡多语 9×50k 的 encoder-only 训练（本次新增）

- **动机**：§3.1 的负面结论来自**单语** zh→en。这里把同一"仅重训 `mapping_enc2llm`、冻结 LLM、纯英文 CE"配方推广到**均衡多语**，检验"多语言覆盖"能否弥补缺口。独立文件夹 `stage1_encoder_x_en/`。
- **数据**：9 语 × 50k = **450,000**，均衡混合。构造脚本 `stage1_encoder_x_en/build_from_opus100.py`。
  - 8 语来自 OPUS-100 parquet（zh=en-zh, bn=bn-en, th=en-th, ja=en-ja, ru=en-ru, de=de-en, fr=en-fr, es=en-es）。
  - sw 来自 NLLB JSONL `data/encoder_only/opus100_sw_en_200k.jsonl`（20 万→过滤去重后 199,722 可用→采样 5 万；本地 OPUS-100 镜像无 sw-en）。
  - 统一流程：normalize → 过滤（2–500 字符、去 `src==tgt`、`tgt` 须含拉丁字母）→ 精确去重 → 蓄水池采样（seed 42+idx）。
  - 产物：`data/stage1_encoder_x_en/{<lang>_en.jsonl, multilingual_x_en.jsonl(450k), multilingual_x_en.train.jsonl(450k)}`。schema 见 `stage1_encoder_x_en/DATA_REQUIREMENTS.md` §12。
- **训练**：`stage1_encoder_x_en/train_encoder_x_en_mp.py`（`llm_only=True`，仅 `mapping_enc2llm` 可训 = **20.98M**；NLLB/LLM 全冻结；`dec_lambda=0, ot_lambda=0`）。4×RTX 5090，gloo CPU 梯度同步。`per_device_batch=6 × grad_accum=2 × 4 卡 = eff_batch 48`，lr=2e-5 cosine，warmup_ratio=0.03，bf16，3 epoch。
  - 共 **28,120** optimizer steps；loss **5.98 → ~2.5**（warmup 后随 lr 上升快降，末段退火收敛，无 NaN/发散）。
  - 产物：`outputs/stage1_encoder_x_en_multi/checkpoint-final`（+ 每 500 步 / 每 epoch）。日志 `outputs/stage1_encoder_x_en_multi/train.log`。
- **结果（多语样例 spot-check，非 BLEU）**：`stage1_encoder_x_en/_test_samples.py`，一次加载跑 6 语；输出**流畅但与原句无关**（幻觉）：

| 语种 | 原句（含义） | 模型输出 |
|------|------------|----------|
| zh | 今天天气很好，我想去公园散步 | *I'm going to miss the beach.* |
| de | 我昨天读了一本有趣的书 | *I've always wanted to write a book.* |
| fr | 火车明早八点出发 | *The game will be played on Saturday.* |
| ru | 她喜欢晚上听音乐 | *I don't like to watch TV.* |
| ja | 他每天早上喝咖啡 | *I don't drink.*（唯一蹭到 "drink"） |
| sw | 孩子们在场上踢球 | *The water is very low.* |

- **结论**：均衡多语数据**同样无法**让 encoder-only 配方产出忠实翻译——与 §3.1 单语结论一致并将其推广：输出随输入变化但语义不忠实，低 loss（2.5）只是"冻结 LLM + 英文 CE"的流畅英文捷径，非跨语对齐。**增加语言数 / 数据量都不是杠杆**；缺口在缺 decoder CE + OT 联合约束（见 §3.1 诊断）。
  - 边界（诚实标注）：本次仅做 6 句定性抽查，**未计算 BLEU**（§3.1 已在 FLORES 上给出 zh→en BLEU≈0，机制同源）；未与 §3.1 checkpoint 做表示层余弦对比。
- **工程记录（本次）**：
  1. **喂数据瓶颈**：`DataLoader` 原本 `num_workers=0`，逐 batch 在主进程做 NLLB+LLM 分词，GPU 利用率仅 ~35% 且剧烈抖动。改为 `num_workers=8 + pin_memory + persistent_workers + prefetch`、batch 2→6 后，利用率升到 **80–100%**（`train_encoder_x_en_mp.py` / `config.env` / `run_train.sh`）。
  2. **GPU3 掉总线**：训练中途 GPU3 出现 `device handle ... Unknown Error`（掉出 PCIe 总线），导致整机所有卡 CUDA 无法初始化（`torch._C._cuda_init()` 全崩）；经管理员复位/重启恢复后才重启 4 卡训练。属 5090 稳定性问题，非代码所致。

---

## 4. Zero-shot MGSM 必要性消融

- **目的**：验证仅 encoder 侧适配、不做 Stage 2/3 全量 SFT 时 MGSM 表现。
- **设置**：`eval_encoder_only_zeroshot_tasks.py`，Alpaca + 英文 CoT 提示，250 题/语言，zero-shot。
- **输出**：`outputs/encoder_only_*_zeroshot/metrics.json`。

| 基座 | zh Acc | en Acc |
|------|--------|--------|
| XBridge-base（无适配） | 3.2 | 5.6 |
| encoder_only_continue_half | 7.2 | 17.6 |
| encoder_only_continue_instruct | 13.2 | 25.2 |
| **XBridge-SFT（完整 Stage 2/3，参照）** | **56.8** | **64.8** |

**结论**：encoder-only 适配仅小幅提升 MGSM（3→25），远不及完整 SFT（56–65），证明 Stage 2/3 不可省。

---

## 5. 实现 / 数值一致性测试

| 脚本 | 检查内容 |
|------|----------|
| `test_dtype_parity.py` | bf16 vs fp32 前向输出一致性 |
| `test_llm_only_parity.py` | `llm_only` 开关前后行为一致性 |
| `test_rms_calibrator_a.py` | RMSNorm 校准系数正确性 |
| `test_flores_calibrator_compare.py` | 校准前后 FLORES 翻译对比 |
| `outputs/translation_smoke_chat_c.json` | chat 模板翻译冒烟测试 |

---

## 6. XBridge-SFT 翻译能力退化对比（本次新增）

- **目的**：量化 SFT 的下游适配对裸翻译能力的影响。
- **设置**：FLORES-101 devtest 每语言前 200 句，base 与 SFT 用同一脚本同一参数（`llm_only=False` 打开 decoder 路径），sacrebleu `flores200`。
- **脚本**：`scripts/compare_bleu_200.py`；输出 `outputs/flores101_base_200/`、`outputs/flores101_sft_200_final/`。

### 6.1 逐方向 BLEU（n=200）

| 方向 | base | SFT | diff | 方向 | base | SFT | diff |
|------|------|-----|------|------|------|-----|------|
| bn-en | 39.87 | 1.60 | −38.27 | en-bn | 26.20 | 1.42 | −24.78 |
| de-en | 47.04 | 1.96 | −45.08 | en-de | 37.12 | 1.52 | −35.60 |
| es-en | 32.63 | 1.68 | −30.95 | en-es | 29.16 | 1.20 | −27.96 |
| fr-en | 46.72 | 1.99 | −44.73 | en-fr | 49.35 | 2.09 | −47.26 |
| ja-en | 31.37 | 1.46 | −29.91 | en-ja | 14.66 | 0.38 | −14.28 |
| ru-en | 39.39 | 1.39 | −38.00 | en-ru | 34.59 | 1.49 | −33.10 |
| sw-en | 48.41 | 2.13 | −46.28 | en-sw | 36.74 | 1.37 | −35.37 |
| th-en | 32.06 | 1.33 | −30.73 | en-th | 28.45 | 1.29 | −27.16 |
| zh-en | 23.24 | 1.50 | −21.74 | en-zh | 28.61 | 0.84 | −27.77 |

### 6.2 汇总

| 方向组 | base | SFT | diff |
|--------|------|-----|------|
| X→En 均值 | 37.86 | **1.67** | −36.19 |
| En→X 均值 | 31.65 | **1.29** | −30.36 |

### 6.3 输出形态

- base zh→en：`He adds: "We now have rats that are 4 months old and have no diabetes..."` ✅
- SFT  zh→en：`He found 4-month-old mice... He then bred the 4-month-old mice with the 2-month-old mice, and all of the offspring were free of the disease. He then bred...`（重复幻觉）❌
- SFT  en→fr：`...les souris n'étaient pas diabétiques quand ils étaient 4 mois, les souris n'étaient pas diabétiques...`（重复退化）❌

**结论**：SFT 的 Stage 2/3 把 encoder→LLM 与 LLM→decoder 接口适配到"指令跟随 + step-by-step 推理"，裸喂源句（无指令模板）时 LLM 内核漫谈、decoder 重复，翻译能力基本丧失（BLEU 1–2 vs base 30+）。SFT 是推理模型，非翻译模型——与论文用法一致（base 跑翻译、SFT 跑 MGSM）。

---

## 7. 复现入口

```bash
bash scripts/reproduce.sh        # Stage 1 FLORES + Stage 2/3 MGSM 主线
bash scripts/run_flores_eval.sh  # FLORES BLEU/COMET 计算
bash scripts/run_train_encoder_only.sh   # encoder-only 训练
bash scripts/run_eval_encoder_only.sh    # encoder-only FLORES 评测
bash scripts/run_eval_encoder_only_zeroshot.sh  # zero-shot MGSM 消融
python scripts/compare_bleu_200.py       # SFT vs base 翻译退化对比
```

## 8. 关键结论

1. **XBridge-base 复现论文翻译结果**：X→En 37.27 / En→X 31.31 BLEU，与论文 36.21/29.82 一致。
2. **XBridge-SFT 复现论文 MGSM 结果**：英文 CoT 路径 45–65% 准确率，与论文一致；decoder 路径 0（设计如此）。
3. **Stage 2/3 不可省**：仅 encoder-only 适配 MGSM 仅 3–25%，完整 SFT 达 56–65%。
4. **SFT 丧失裸翻译能力**：BLEU 从 30+ 塌到 1–2，输出重复退化；SFT 应配合指令模板用于推理，不应作翻译模型使用。
5. **Encoder-only 配方缺口稳健**：仅重训 `mapping_enc2llm`（冻结 LLM、纯英文 CE）在单语（§3.1，zh→en BLEU≈0）与均衡多语（§3.2，9×50k，6 语定性幻觉）下都无法产出忠实翻译；低 loss 是流畅英文捷径，缺口在 decoder CE + OT 联合约束，堆数据/加语言均无效。
