# *Language on Demand, Knowledge at Core*: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality

> [Mengyu Bu](https://bingo123122121.github.io/), [Yang Feng](https://people.ucas.edu.cn/~yangfeng?language=en)

[![arXiv](https://img.shields.io/badge/arXiv-2603.17512-b31b1b%3Flogo%3DarXiv?logo=arxiv&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2603.17512)](https://arxiv.org/abs/2603.17512) [![github](https://img.shields.io/badge/GitHub-Code-keygen%3Flogo%3DGitHub?logo=github&link=https%3A%2F%2Fgithub.com%2Fictnlp%2FXBridge)](https://github.com/ictnlp/XBridge) [![github](https://img.shields.io/badge/Hugging%20Face-Model-b31b1b%3Flogo%3Dhuggingface?logo=Hugging%20Face&color=blue&link=https%3A%2F%2Fhuggingface.co%2Fcollections%2FICTNLP%2Fxbridge)](https://huggingface.co/collections/ICTNLP/xbridge)

Official code for **ACL 2026 Main Conference** paper "*Language on Demand, Knowledge at Core*: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality".

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

For training, we extract multilingual translation data from [OPUS-100](https://github.com/EdinburghNLP/opus-100-corpus), multilingual mathematical reasoning data from [MultilingualMath](https://drive.google.com/drive/folders/1evjD7HMLPBel1GKXtg-z77dR8DuCquPl?dmr=1&ec=wgc-drive-hero-goto), and multilingual abstractive summarization data from [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum). Please refer to the paper for detailed data construction procedures.

For evaluation, we test cross-model mapping quality with [FLORES-101](https://github.com/facebookresearch/flores/tree/main/previous_releases/flores101) for stage 1, test multilingual mathematical reasoning with [MGSM](https://huggingface.co/datasets/juletxara/mgsm), and multilingual abstract summarization with [XL-Sum test set](https://huggingface.co/datasets/csebuetnlp/xlsum).

## 🔥Training

XBridge composes LLMs with NMT models in three stages:

* **Stage 1: Cross-Model Mapping** 

  Establish coarse-grained semantic alignment among the multilingual encoder, the LLM, and the multilingual decoder using trilingual translation data `(x, en, y)`

* **Stage 2: Encoder-Side Adaptation**  

  Adapt multilingual input representations to downstream instruction-following tasks.

* **Stage 3: Decoder-Side Adaptation**

  Adapt the LLM-decoder interface for robust multilingual generation.

See our paper for details about training strategy.

## 💭Inference

Below is an example evaluation script.

```shell
# evaluation on stage1
generate_batch_from_file=inference_xbridge_stage1.py
mt_tokenizer_path=/path/to/your/NMT/model
llm_tokenizer_path=/path/to/your/LLM
base_model=/path/to/your/stage1/checkpoint
testset_dir=/path/to/your/FLOERS-101
output_dir=/path/to/your/output/dir
test_langs=en,bn,de,es,fr,ja,ru,sw,th,zh

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=0 python $generate_batch_from_file \
    --mt_tokenizer_path $mt_tokenizer_path --llm_tokenizer_path $llm_tokenizer_path \
    --base_model $base_model \
    --batch_size 12 \
    --testset_dir $testset_dir --output_dir $output_dir \
    --test_langs $test_langs --max_new_tokens 512

# evaluation on stage2&3
generate_batch_from_file=inference_xbridge_stage2_and_3.py
mt_tokenizer_path=/path/to/your/NMT/model
llm_tokenizer_path=/path/to/your/LLM
base_model=/path/to/your/stage3/checkpoint
testset_dir=/path/to/your/MGSM
output_dir=/path/to/your/output/dir
test_langs=en,bn,de,es,fr,ja,ru,sw,th,zh

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=0 python $generate_batch_from_file \
    --mt_tokenizer_path $mt_tokenizer_path --llm_tokenizer_path $llm_tokenizer_path \
    --base_model $base_model \
    --batch_size 12 \
    --testset_dir $testset_dir --output_dir $output_dir \
    --test_langs $test_langs --max_new_tokens 512
```

## ✨Released Checkpoints

We release `XBridge-base` and `XBridge-SFT` in the [Hugging Face collection](https://huggingface.co/collections/ICTNLP/xbridge): 

* `XBridge-base` is trained with stage 1 (cross-model alignment) using trilingual translation data, composing [`LLaMA3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) with [`NLLB-200-1.3B`](https://huggingface.co/facebook/nllb-200-1.3B).
* `XBridge-SFT` further extends `XBridge-base` by training stage 2 (encoder-side adaptation) and stage 3 (decoder-side adaptation) for multilingual reasoning task [MGSM](https://huggingface.co/datasets/juletxara/mgsm).

Language coverage:  *Bn, De, En, Es, Fr, Ja, Ru, Sw, Th, Zh*.

## 🌍BayLing-MLingual

XBridge serves as the research foundation of **BayLing-MLingual**. BayLing-MLingual extends XBridge from a research setting to practical multilingual question answering across 50 languages and 2500 cross-lingual pairs. **Try our BayLing-MLingual for general QA among 50 languages!**

👉 https://github.com/BayLing-Models/BayLing-MLingual

## ⚖️LICENSE
Our code is released under the Apache-2.0 License. Our model is intended for academic research purposes only and may **NOT** be used for commercial purposes.

You are free to use, modify, and distribute this model in academic settings, provided that the following conditions are met:

* **Non-commercial use**: The model may not be used for any commercial purposes.
* **Citation**: If you use this model in your research, please cite the original work.

### ❗Commercial Use Restriction
For any commercial use inquiries or to obtain a commercial license, please contact `fengyang@ict.ac.cn`.


## 📚Citation

If you have any questions, please feel free to submit an issue or contact `bumengyu23z@ict.ac.cn`. 

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



