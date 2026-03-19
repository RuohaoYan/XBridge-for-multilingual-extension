# *Language on Demand, Knowledge at Core*: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality

> [Mengyu Bu](https://bingo123122121.github.io/), [Yang Feng](https://people.ucas.edu.cn/~yangfeng?language=en)

![Paper](https://img.shields.io/badge/arXiv-2603.17512-b31b1b?logo=arXiv) ![code](https://img.shields.io/badge/github-XBridge-keygen?logo=GitHub&link=https%3A%2F%2Fgithub.com%2Fictnlp%2FXBridge)

Official code for Paper "*Language on Demand, Knowledge at Core*: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality".

![framework](figures/framework.png)

## 📖Introduction

XBridge leverages a compositional encoder-LLM-decoder architecture that offloads multilingual capability to the composed NMT model while preserving the LLM as an English-centric core for general knowledge processing. XBridge brings low-resource and unseen language performance close to that of composed NMT models, substantially narrowing the gap across languages without retraining the LLM.

## 🚀Key Features

* **Compositional multilinguality**: separates responsibilities across modules: encoder for multilingual understanding, LLM for general knowledge processing, and decoder for multilingual generation.
* **Strong cross-lingual generalization**: the cross-model mapping layers are language-agnostic that even generalizes well to the untuned languages.
* **Controllable language generation:** controls output languages by the target language token of the decoder.
* **Lossless language switching:** supports arbitrary language-to-language generation through the LLM pivot without degrading performance.
* **Mitigating catastrophic forgetting in multilingual extension:** boosts low-resource or unseen languages understanding and generation of LLM to near-NMT performance, while maintaining or improving high-resource languages performance, avoiding the common new–old language trade-off in multilingual extension.
* **Efficient training**: requires only minimal additional parameters, limited training data (mostly bilingual pairs), and modest overhead.

## 🛠️Installation

### 1. Clone this repository

``` shell
git clone https://github.com/ictnlp/XBridge.git
```

### 2. Prepare training environment

``` shell
conda create -n xbridge python=3.9.12
conda activate xbridge
pip install -r requirements.txt
```

### 3. Prepare evaluation environment

For evaluation, we use **MMT-LLM** for translation task of base LLMs.

``` shell
git clone https://github.com/NJUNLP/MMT-LLM.git
```

## 📄Dataset Preparation

We extract multilingual translation data from [OPUS-100](https://github.com/EdinburghNLP/opus-100-corpus), multilingual mathematical reasoning data from [MultilingualMath](https://drive.google.com/drive/folders/1evjD7HMLPBel1GKXtg-z77dR8DuCquPl?dmr=1&ec=wgc-drive-hero-goto), and multilingual abstractive summarization data from [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum). Please refer to the paper for detailed data construction procedures.

## 🔥Training

XBridge composes LLMs with NMT models in three stages:

* **Stage 1: Cross-Model Mapping** 

  Establish coarse-grained semantic alignment among the multilingual encoder, the LLM, and the multilingual decoder using trilingual translation data `(x, en, y)`

* **Stage 2: Encoder-Side Adaptation**  

  Adapt multilingual input representations to downstream instruction-following tasks.

* **Stage 3: Decoder-Side Adaptation**

  Adapt the LLM-decoder interface for robust multilingual generation.

Below is an example training script.

```shell
# Stage 1
finetune=finetune_xbridge_stage1.py
mt_path=/path/to/your/NMT/model
mt_tokenizer_path=/path/to/your/NMT/model
llm_path=/path/to/your/LLM
llm_tokenizer_path=/path/to/your/LLM
data_path=/path/to/your/data
output_dir=/path/to/your/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python $finetune \
    --mt_path $mt_path --mt_tokenizer_path $mt_tokenizer_path \
    --llm_path $llm_path --llm_tokenizer_path $llm_tokenizer_path \
    --data_path $data_path \
    --output_dir $output_dir \
    --num_epochs=2 --batch_size=128 --micro_batch_size=8 \
    --max_seq_len=512 --group_by_length \
    --freeze_enc=True --freeze_llm=True --freeze_dec=True \
    --freeze_mapping_enc2llm=False --freeze_mapping_llm2dec=True \
    --learning_rate=2e-5 --dec_lambda=1.0 --ot_lambda=6.0

# Stage 2
finetune=finetune_xbridge_stage2_and_3.py
mt_path=/path/to/your/NMT/model
mt_tokenizer_path=/path/to/your/NMT/model
llm_path=/path/to/your/LLM
llm_tokenizer_path=/path/to/your/stage1/checkpoint
data_path=/path/to/your/data
output_dir=/path/to/your/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python $finetune \
    --mt_path $mt_path --mt_tokenizer_path $mt_tokenizer_path \
    --llm_path $llm_path --llm_tokenizer_path $llm_tokenizer_path \
    --data_path $data_path \
    --output_dir $output_dir \
    --num_epochs=3 --batch_size=128 --micro_batch_size=8 \
    --max_seq_len=512 --group_by_length \
    --freeze_enc True --freeze_llm True --freeze_dec True \
    --freeze_mapping_enc2llm False --freeze_mapping_llm2dec True \
    --task="math" \
    --learning_rate=2e-5


# Stage 3
finetune=finetune_xbridge_stage2_and_3.py
mt_path=/path/to/your/NMT/model
mt_tokenizer_path=/path/to/your/NMT/model
llm_path=/path/to/your/LLM
llm_tokenizer_path=/path/to/your/stage2/checkpoint
data_path=/path/to/your/data
output_dir=/path/to/your/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python $finetune \
    --mt_path $mt_path --mt_tokenizer_path $mt_tokenizer_path \
    --llm_path $llm_path --llm_tokenizer_path $llm_tokenizer_path \
    --data_path $data_path \
    --output_dir $output_dir \
    --num_epochs=3 --batch_size=128 --micro_batch_size=8 \
    --max_seq_len=512 --group_by_length \
    --freeze_enc True --freeze_llm True --freeze_dec True \
    --freeze_mapping_enc2llm True --freeze_mapping_llm2dec False \
    --task="math" \
    --learning_rate=2e-5
```

## 📚Citation

If you find this repository useful, please star this repository and cite our paper:

```tex
@misc{bu2026languagedemandknowledgecore,
      title={Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality}, 
      author={Mengyu Bu and Yang Feng},
      year={2026},
      eprint={2603.17512},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.17512}, 
}
```



