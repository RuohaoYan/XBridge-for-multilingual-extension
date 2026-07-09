# Data requirements for Stage 1 encoder x->English alignment

This folder trains only the encoder-to-LLM bridge:

```text
multilingual input x -> MT encoder -> mapping_enc2llm -> frozen LLM -> English en
```

Therefore the required data is bilingual `x-en` parallel data. It is **not** the full paper Stage 1 trilingual `(x, en, y)` data.

## 1. Required JSONL schema

Each line must be one JSON object:

```json
{"src_lang": "zho_Hans", "src": "今天阳光很好。", "tgt": "The weather is sunny today.", "prompt": "Translate into English:"}
```

Required fields:

| Field | Meaning | Example |
|---|---|---|
| `src_lang` | NLLB source language code for `src` | `zho_Hans` |
| `src` | multilingual source sentence `x` | `今天阳光很好。` |
| `tgt` | English target sentence `en` | `The weather is sunny today.` |

Optional field:

| Field | Meaning | Default |
|---|---|---|
| `prompt` | LLM instruction inserted before English labels | `Translate into English:` |

The training script also accepts aliases when reading JSONL:

- source: `src`, `source`, `input`
- English target: `tgt`, `target`, `english`, `en`
- source language: `src_lang`, `source_lang`, `lang`
- prompt: `prompt`, `instruction`

## 2. What the data should represent

For each sample:

```text
src_lang + src  -> encoded by NLLB / MT encoder
tgt             -> English labels for frozen LLM CE loss
prompt          -> optional English instruction tokens
```

The model input is internally built as:

```text
MT(src) + prompt + English target
```

with augmentation labels:

```text
1 = MT source tokens
2 = prompt tokens
3 = English target tokens
```

Only the target-token positions are used as LLM labels by `modeling_xbridge.py`.

## 3. Correct examples

Chinese to English:

```json
{"src_lang": "zho_Hans", "src": "今天阳光很好。", "tgt": "The weather is sunny today.", "prompt": "Translate into English:"}
```

Bengali to English:

```json
{"src_lang": "ben_Beng", "src": "আজ আবহাওয়া ভালো।", "tgt": "The weather is good today.", "prompt": "Translate into English:"}
```

Hausa to English:

```json
{"src_lang": "hau_Latn", "src": "Yanayi yana da kyau yau.", "tgt": "The weather is good today.", "prompt": "Translate into English:"}
```

Uyghur to English:

```json
{"src_lang": "uig_Arab", "src": "بۈگۈن ھاۋا ياخشى.", "tgt": "The weather is good today.", "prompt": "Translate into English:"}
```

Swahili is valid for NLLB tokenization as `swh_Latn`, but `Helsinki-NLP/opus-100` does not provide an `sw-en` / `swh-en` subset. If you train Swahili here, use an external Swahili-English parallel corpus and convert it to the same JSONL schema.

## 4. Incorrect examples

Do not use target-language decoder supervision here:

```json
{"src_lang": "zho_Hans", "src": "今天阳光很好。", "tgt": "The weather is sunny today.", "decoder_target": "今天阳光很好。", "tgt_lang": "zho_Hans"}
```

`decoder_target` and `tgt_lang` are for full trilingual Stage 1, not for this folder.

Do not put non-English text into `tgt`:

```json
{"src_lang": "zho_Hans", "src": "今天阳光很好。", "tgt": "今天阳光很好。"}
```

For this folder, `tgt` must be English.

Do not omit `src_lang`:

```json
{"src": "今天阳光很好。", "tgt": "The weather is sunny today."}
```

NLLB tokenization requires the correct source language code.

## 5. Recommended data sources

Recommended source for languages covered by the Hugging Face dataset: `Helsinki-NLP/opus-100` English-centric parallel data.

Important caveat: the Hugging Face `Helsinki-NLP/opus-100` subset list does **not** include `sw-en`, `en-sw`, `swh-en`, or `so-en`. Do not assume every XBridge/FLORES language is available in OPUS-100.

For each covered non-English language, use the corresponding `x-en` or `en-x` subset and normalize it to this folder's `src/tgt` schema. Examples available in `Helsinki-NLP/opus-100` include:

```text
zh-en / en-zh
bn-en
en-ha
en-ig
en-rw
en-xh
en-yo
en-zu
th-en / en-th
ja-en / en-ja
ru-en / en-ru
de-en / en-de
fr-en / en-fr
es-en / en-es
ug-en / en-ug
```

For a Chinese-only sanity run, `zh-en` or `en-zh` normalized to `src=zh, tgt=en` is enough. For real multilingual encoder alignment, mix multiple covered `x-en` language pairs into a single JSONL file and keep a correct `src_lang` on every line.

For languages not present in OPUS-100, use another parallel corpus and convert it to the same schema. The training code is dataset-agnostic as long as each line has `src_lang`, `src`, and English `tgt`.

## 6. Recommended scale

Use the following levels depending on the experiment:

| Experiment | Recommended data |
|---|---|
| Smoke test | 1k-5k examples for one language |
| Chinese-only debug | 50k-200k `zh-en` examples |
| Multilingual warmup | 50k examples per language |
| Paper-like encoder-side coverage | 50k per selected language; use OPUS-100 where available and external corpora for missing languages |

This folder does not require `y`, NLLB-200-3.3B synthetic target generation, decoder labels, or OT labels.

## 7. Build data from parallel text files

Given aligned files:

```text
data/raw/zh.txt
data/raw/en.txt
```

Run:

```bash
python stage1_encoder_x_en/build_x_en_data.py \
  --source_file data/raw/zh.txt \
  --english_file data/raw/en.txt \
  --output_file data/stage1_encoder_x_en/zh_en.jsonl \
  --src_lang zho_Hans \
  --prompt "Translate into English:"
```

For another language, change `--src_lang` and input files.

## 8. Build data from JSONL

Input JSONL may use flexible field names:

```json
{"source": "今天阳光很好。", "english": "The weather is sunny today.", "src_lang": "zho_Hans"}
```

Run:

```bash
python stage1_encoder_x_en/build_x_en_data.py \
  --input_file data/raw/zh_en_raw.jsonl \
  --output_file data/stage1_encoder_x_en/zh_en.jsonl \
  --src_lang zho_Hans
```

If a record already contains `src_lang`, it is used. Otherwise, the command-line `--src_lang` is used.

## 9. Mixing multiple languages

Build separate files first:

```bash
python stage1_encoder_x_en/build_x_en_data.py --source_file data/raw/zh.txt --english_file data/raw/zh.en.txt --output_file data/stage1_encoder_x_en/zh_en.jsonl --src_lang zho_Hans
python stage1_encoder_x_en/build_x_en_data.py --source_file data/raw/bn.txt --english_file data/raw/bn.en.txt --output_file data/stage1_encoder_x_en/bn_en.jsonl --src_lang ben_Beng
python stage1_encoder_x_en/build_x_en_data.py --source_file data/raw/ha.txt --english_file data/raw/ha.en.txt --output_file data/stage1_encoder_x_en/ha_en.jsonl --src_lang hau_Latn
```

Then concatenate and shuffle:

```bash
cat data/stage1_encoder_x_en/*_en.jsonl > data/stage1_encoder_x_en/multilingual_x_en.raw.jsonl
shuf data/stage1_encoder_x_en/multilingual_x_en.raw.jsonl > data/stage1_encoder_x_en/multilingual_x_en.jsonl
```

Use the mixed file in `config.env`:

```bash
DATA_MODE="json"
INPUT_FILE="data/stage1_encoder_x_en/multilingual_x_en.jsonl"
TRAIN_FILE="data/stage1_encoder_x_en/multilingual_x_en.jsonl"
```

## 10. Data quality checks

Before training, check:

- `src` and `tgt` are aligned sentence pairs.
- `tgt` is English.
- `src_lang` is a valid NLLB language code.
- No empty `src` or `tgt` fields.
- Very long examples are filtered or truncated.
- The file is shuffled if multiple languages are mixed.
- The prompt is consistent across samples unless prompt variation is intentional.
- The requested language pair actually exists in the source dataset. For example, do not request `sw-en` from `Helsinki-NLP/opus-100`.

A quick line-count check:

```bash
wc -l data/stage1_encoder_x_en/*.jsonl
```

A quick JSON validity check:

```bash
python -m json.tool < data/stage1_encoder_x_en/zh_en.jsonl > /dev/null
```

For JSONL, validate line by line:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('data/stage1_encoder_x_en/zh_en.jsonl')
for i, line in enumerate(path.open(encoding='utf-8'), 1):
    row = json.loads(line)
    assert row.get('src_lang'), (i, row)
    assert row.get('src'), (i, row)
    assert row.get('tgt'), (i, row)
print('ok', path)
PY
```

## 11. Difference from full paper Stage 1 data

Full paper Stage 1 needs trilingual data:

```text
x / y language input + English pivot en + target-language decoder output y or x
```

That full setup trains:

```text
mapping_enc2llm + mapping_llm2dec + decoder cross-attention
```

with:

```text
LLM CE + decoder CE + OT
```

This folder intentionally does not use that data. It only trains the encoder-side bridge so that multilingual inputs can be mapped into the frozen LLM and generate English reliably.

## 12. Data already built in this repo

The concrete Stage 1 encoder x->English data currently generated in this repo is a
balanced 9-language mix, **50,000 examples per language = 450,000 total**.

### 12.1 Builder

Built with `stage1_encoder_x_en/build_from_opus100.py` (not `build_x_en_data.py`, which
cannot read parquet). For every language the pipeline is identical:

```text
read source -> normalize whitespace -> filter -> dedupe -> reservoir-sample to cap
```

Filters applied (`keep()`):

- `2 <= len(src)` and `2 <= len(tgt)` (`--min_chars 2`)
- `len(src) <= 500` and `len(tgt) <= 500` (`--max_chars 500`)
- drop `src == tgt` (untranslated / copied lines)
- `tgt` must contain at least one Latin letter (English target sanity)
- exact `(src, tgt)` de-duplication within each language

Reservoir sampling uses `seed = 42 + language_index`, so runs are deterministic and each
language is decorrelated.

### 12.2 Languages and sources

Eight languages come from the local OPUS-100 mirror
(`data/Helsinki-NLP_opus-100/<pair>/train-*.parquet`, each row
`{"translation": {"en": ..., "<lang>": ...}}`). Swahili is **not** in that mirror, so it
comes from the NLLB x-en JSONL instead.

| lang | `src_lang` (NLLB) | source kind | source path | cap |
|---|---|---|---|---|
| zh | `zho_Hans` | opus-100 parquet | `data/Helsinki-NLP_opus-100/en-zh` | 50,000 |
| bn | `ben_Beng` | opus-100 parquet | `data/Helsinki-NLP_opus-100/bn-en` | 50,000 |
| th | `tha_Thai` | opus-100 parquet | `data/Helsinki-NLP_opus-100/en-th` | 50,000 |
| ja | `jpn_Jpan` | opus-100 parquet | `data/Helsinki-NLP_opus-100/en-ja` | 50,000 |
| ru | `rus_Cyrl` | opus-100 parquet | `data/Helsinki-NLP_opus-100/en-ru` | 50,000 |
| de | `deu_Latn` | opus-100 parquet | `data/Helsinki-NLP_opus-100/de-en` | 50,000 |
| fr | `fra_Latn` | opus-100 parquet | `data/Helsinki-NLP_opus-100/en-fr` | 50,000 |
| es | `spa_Latn` | opus-100 parquet | `data/Helsinki-NLP_opus-100/en-es` | 50,000 |
| sw | `swh_Latn` | NLLB JSONL | `data/encoder_only/opus100_sw_en_200k.jsonl` | 50,000 |

Swahili source has 200,000 raw pairs; 199,722 survive filtering/dedup and 50,000 are
sampled for the balanced mix.

### 12.3 Output files

All under `data/stage1_encoder_x_en/`:

| File | Lines | Role |
|---|---|---|
| `<lang>_en.jsonl` (9 files) | 50,000 each | per-language split |
| `multilingual_x_en.jsonl` | 450,000 | mixed + shuffled (seed 42); build source |
| `multilingual_x_en.train.jsonl` | 450,000 | json-mode passthrough; the `TRAIN_FILE` used by training |

Every line follows the schema in section 1, e.g.:

```json
{"src_lang": "swh_Latn", "src": "...", "tgt": "English ...", "prompt": "Translate into English:"}
```

### 12.4 Regenerate

Balanced 9x50k (current state):

```bash
python3 stage1_encoder_x_en/build_from_opus100.py \
  --langs zh,bn,th,ja,ru,de,fr,es,sw --per_lang 50000 --seed 42 \
  --out_dir data/stage1_encoder_x_en
```

Per-language caps use `lang[:cap]` syntax, e.g. keep all Swahili at 200k:

```bash
python3 stage1_encoder_x_en/build_from_opus100.py \
  --langs zh,bn,th,ja,ru,de,fr,es,sw:200000 --per_lang 50000 --seed 42
```

### 12.5 Config wiring

`stage1_encoder_x_en/config.env` consumes this data with `DATA_MODE="json"`:

```bash
DATA_MODE="json"
INPUT_FILE="data/stage1_encoder_x_en/multilingual_x_en.jsonl"
TRAIN_FILE="data/stage1_encoder_x_en/multilingual_x_en.train.jsonl"
```

`INPUT_FILE` and `TRAIN_FILE` **must be different paths**: `build_x_en_data.py` truncates
the output before reading the input, so pointing both at the same file yields an empty
train set. Then `bash stage1_encoder_x_en/run_train.sh` regenerates `TRAIN_FILE` and trains.
