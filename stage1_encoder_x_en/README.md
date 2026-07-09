# Stage 1 encoder x->English alignment

This folder is for the lighter Stage 1 objective:

```text
multilingual input x
  -> NLLB / MT encoder
  -> mapping_enc2llm
  -> frozen LLM
  -> English output en
```

It intentionally trains only `mapping_enc2llm`. It does **not** train `mapping_llm2dec`, NLLB decoder cross-attention, decoder CE, or OT. Use this when the immediate goal is to make multilingual encoder representations enter the LLM reliably and make the frozen LLM generate stable English.

## Why this folder exists

The full paper Stage 1 uses trilingual `(x, en, y)` data and three losses. For debugging and warm-starting, it is often useful to first train only the encoder-to-LLM bridge on bilingual `x,en` data. This isolates the source of hallucination to the encoder-to-LLM side.

## Files

- `DATA_REQUIREMENTS.md`: detailed data schema, examples, scale recommendations, and quality checks.
- `build_x_en_data.py`: converts parallel text or JSON/JSONL into the local JSONL format.
- `train_encoder_x_en_mp.py`: multi-GPU/gloo training script for `mapping_enc2llm` only.
- `infer_encoder_x_en.py`: batch inference script for English generation from multilingual input.
- `run_train.sh`: shell runner using `config.env`.
- `run_infer.sh`: shell runner for inference using `config.env`.
- `sample_config.env`: editable paths and hyperparameters.

## Data format

Training JSONL uses one sample per line:

```json
{"src_lang": "zho_Hans", "src": "今天阳光很好。", "tgt": "The weather is sunny today.", "prompt": "Translate into English:"}
```

Fields:

- `src_lang`: NLLB source language code, for example `zho_Hans`, `uig_Arab`, `ben_Beng`, `swh_Latn`.
- `src`: multilingual input sentence `x`.
- `tgt`: English target `en`.
- `prompt`: optional LLM instruction. The default is `Translate into English:`.

For complete data requirements, see `stage1_encoder_x_en/DATA_REQUIREMENTS.md`.

## Quick start

From the repository root:

```bash
cp stage1_encoder_x_en/sample_config.env stage1_encoder_x_en/config.env
# edit paths in stage1_encoder_x_en/config.env
bash stage1_encoder_x_en/run_train.sh
```

For inference:

```bash
bash stage1_encoder_x_en/run_infer.sh
```

## Build data directly

From parallel files:

```bash
python stage1_encoder_x_en/build_x_en_data.py \
  --source_file data/raw/zh.txt \
  --english_file data/raw/en.txt \
  --output_file data/stage1_encoder_x_en/zh_en.jsonl \
  --src_lang zho_Hans \
  --prompt "Translate into English:"
```

From JSONL:

```bash
python stage1_encoder_x_en/build_x_en_data.py \
  --input_file data/raw/zh_en_raw.jsonl \
  --output_file data/stage1_encoder_x_en/zh_en.jsonl \
  --src_lang zho_Hans
```

## Train directly

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nproc_per_node=3 stage1_encoder_x_en/train_encoder_x_en_mp.py \
  --train_file data/stage1_encoder_x_en/zh_en.jsonl \
  --output_dir outputs/stage1_encoder_x_en_zh \
  --mt_path model/nllb-200-1.3B \
  --llm_path model/Meta-Llama-3-8B \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 5 \
  --learning_rate 2e-5 \
  --num_epochs 3
```

Effective batch size is:

```text
per_device_batch_size * gradient_accumulation_steps * world_size
```

## Expected checkpoint

The training script saves `mapping_enc2llm.pt` plus config and metadata:

```text
outputs/stage1_encoder_x_en_zh/checkpoint-final/
```

This checkpoint can be used as a warm start for full Stage 1 by loading `mapping_enc2llm.pt` into the trilingual training script.
