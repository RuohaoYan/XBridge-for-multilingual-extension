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

Swahili is valid for NLLB tokenization as `swh_Latn`, but `Helsinki-NLP/opus-100` does not provide an `sw-en`, `en-sw`, or `swh-en` subset. If you train Swahili here, use an external Swahili-English parallel corpus and convert it to the same JSONL schema.

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

OPUS-100 subset names are not always bidirectional. For example, the subset may be named `en-zh` rather than `zh-en`, but the record still contains both languages in a `translation` dictionary. For this folder, always normalize the record as:

```text
src = non-English sentence
tgt = English sentence
src_lang = NLLB source language code for src
```

Examples of relevant `Helsinki-NLP/opus-100` subsets include:

```text
bn-en
de-en
en-es
en-fr
en-ja
en-ru
en-th
en-zh
en-ha
en-ig
en-rw
en-xh
en-yo
en-zu
en-ug
```

For a Chinese-only sanity run, use `en-zh` and normalize each row to `src=zh, tgt=en, src_lang=zho_Hans` or `zho_Hant`, depending on the Chinese script. For real multilingual encoder alignment, mix multiple covered language pairs into a single JSONL file and keep a correct `src_lang` on every line.

For languages not present in OPUS-100, use another parallel corpus and convert it to the same schema. The training code is dataset-agnostic as long as each line has `src_lang`, `src`, and English `tgt`.

## 6. Recommended scale

Use the following levels depending on the experiment:

| Experiment | Recommended data |
|---|---|
| Smoke test | 1k-5k examples for one language |
| Chinese-only debug | 50k-200k `zh-en` examples after normalization |
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

## 8. Build data from flat JSONL

Input JSONL may use flexible top-level field names:

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

The current `build_x_en_data.py` expects flat fields such as `src/tgt` or `source/english`. It does **not** directly parse the raw Hugging Face OPUS-100 nested format:

```json
{"translation": {"zh": "...", "en": "..."}}
```

Convert raw OPUS-100 rows to flat JSONL or parallel text files first.

## 9. Convert raw Hugging Face OPUS-100 rows

Example for a raw OPUS-100 JSONL-like export with nested `translation` objects:

```bash
python - <<'PY'
import json
from pathlib import Path
src_code = 'zh'
src_lang = 'zho_Hans'
in_path = Path('data/raw/opus100_en_zh.jsonl')
out_path = Path('data/stage1_encoder_x_en/zh_en.jsonl')
out_path.parent.mkdir(parents=True, exist_ok=True)
with in_path.open(encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
    for line in fin:
        row = json.loads(line)
        trans = row['translation']
        record = {
            'src_lang': src_lang,
            'src': trans[src_code],
            'tgt': trans['en'],
            'prompt': 'Translate into English:',
        }
        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
print('wrote', out_path)
PY
```

Change `src_code` and `src_lang` for other languages.

## 10. Mixing multiple languages

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

## 11. Data quality checks

Before training, check:

- `src` and `tgt` are aligned sentence pairs.
- `tgt` is English.
- `src_lang` is a valid NLLB language code.
- No empty `src` or `tgt` fields.
- Very long examples are filtered or truncated.
- The file is shuffled if multiple languages are mixed.
- The prompt is consistent across samples unless prompt variation is intentional.
- The requested language pair actually exists in the source dataset. For example, do not request `sw-en` from `Helsinki-NLP/opus-100`.
- For `en-x` OPUS-100 subsets, verify that you did not accidentally set `src` to English and `tgt` to the non-English sentence.

A quick line-count check:

```bash
wc -l data/stage1_encoder_x_en/*.jsonl
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
    assert row.get('src') != row.get('tgt'), (i, row)
print('ok', path)
PY
```

## 12. Difference from full paper Stage 1 data

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
