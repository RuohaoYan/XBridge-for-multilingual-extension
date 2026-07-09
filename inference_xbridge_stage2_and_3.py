import os
import re
import sys
import ast
import json
import fire
import torch
import transformers
from transformers import GenerationConfig, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

from safetensors.torch import load_file


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

def main(
    load_8bit: bool = False,
    mt_tokenizer_path: str = "",
    llm_tokenizer_path: str = "",
    base_model: str = "",
    batch_size: int = "",
    max_new_tokens: int = 512,
    testset_dir: str = "",
    output_dir: str = "",
    test_langs: str = "",
    no_src_in_prompt: bool = False,
    wrong_src_in_prompt: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(test_langs, str):
        test_langs = [x.strip() for x in test_langs.split(",") if x.strip()]
    elif isinstance(test_langs, tuple):
        test_langs = list(test_langs)
    else:
        test_langs = list(test_langs)

    lang_map_mm2l = {
        'en': 'English', 'zh': 'Chinese', 'es': 'Spanish', 'fr': 'French', 
        'th': 'Thai', 'sw': 'Swahili', 'ja': 'Japanese', 'bn': 'Bengali', 
        'de': 'German', 'ru': 'Russian', 'mn': 'Mongolian', 'kk': 'Kazakh',
        'ar': 'Arabic', 'vi': 'Vietnamese', 'ur': 'Urdu', 'nl': 'Dutch', 'it': 'Italian'
    }

    lang_map_flores2mm = {
        'en': 'eng', 'zh': 'zho_simpl', 'es': 'spa', 'fr': 'fra', 
        'th': 'tha', 'sw': 'swh', 'ja': 'jpn', 'bn': 'ben', 
        'de': 'deu', 'ru': 'rus', 'mn': 'mon', 'kk': 'kaz',
        'ar': 'ara', 'vi': 'vie', 'ur': 'urd', 'nl': 'nld', 'it': 'ita'
    }
    
    langs_map_m2m = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
     'Russian': 'ru', 'Thai': 'th', 'Greek': 'el', 'Telugu': 'te',
     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
     'Vietnamese': 'vi', 'Hindi': 'hi', 'Dutch': 'nl', 'Urdu': 'ur', 'Mongolian': 'mn', 'Kazakh': 'kk'}

    langs_map_nllb = {
        'English': 'eng_Latn', 'Swahili': 'swh_Latn', 'Chinese': 'zho_Hans', 'Bengali': 'ben_Beng',
        'German': 'deu_Latn', 'Spanish': 'spa_Latn', 'French': 'fra_Latn', 'Japanese': 'jpn_Jpan',
        'Russian': 'rus_Cyrl', 'Thai': 'tha_Thai', 'Mongolian': 'khk_Cyrl', 'Kazakh': 'kaz_Cyrl',
        'Arabic': 'arb_Arab', 'Vietnamese': 'vie_Latn', 'Urdu': 'urd_Arab',
        'Dutch': 'nld_Latn', 'Italian': 'ita_Latn'
    }

    if 'nllb' in mt_tokenizer_path.lower():
        langs_map = langs_map_nllb
    else:
        langs_map = langs_map_m2m

    # load tokenizer
    tokenizer_mt = AutoTokenizer.from_pretrained(mt_tokenizer_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_tokenizer_path)
    if "llama3" in llm_tokenizer_path or "llama-3" in llm_tokenizer_path:
        tokenizer_llm.pad_token_id = 128002
    elif tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token_id = 0
    tokenizer_llm.padding_side = "left"
    
    # load model
    config = XBridgeConfig.from_pretrained(base_model)
    config.max_gen_len = max_new_tokens
    config.llm_only = True
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        len_tokenizer_llm=len(tokenizer_llm)
    )
    model.model_mt.lm_head.weight = model.model_mt.model.shared.weight
    # model.model_mt.lm_head._hf_hook.execution_device=model.model_mt.model.shared.weight.device.index

    model.eval()
    
    def mt_input_features(input_text_m2m, source_language, langs_map):
        tokenizer_mt.src_lang = langs_map[source_language]
        encoding_m2m = tokenizer_mt(
            input_text_m2m,
            padding=False,
            truncation=False,
            return_tensors=None,
            add_special_tokens=True
        )
        return encoding_m2m["input_ids"]

    def llm_input_features(input_texts_llm, add_special_tokens=True):
        encoding_llm = tokenizer_llm(
            input_texts_llm,
            padding=False,
            truncation=False,
            return_tensors=None,
            add_special_tokens=add_special_tokens
        )
        return encoding_llm["input_ids"]

    def pad_and_mask(input_ids_mt, input_ids_prompt, pad_token_id, device):
        input_ids = [seq1 + seq2 for seq1, seq2 in zip(input_ids_mt, input_ids_prompt)]
        
        max_len = max(len(seq) for seq in input_ids)
        augmentation = [[0] * (max_len - len(input_ids[i])) + [1] * len(input_ids_mt[i]) + [2] * len(input_ids_prompt[i]) for i in range(len(input_ids_mt))]
        attention_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in input_ids]
        input_ids = [[pad_token_id] * (max_len - len(seq)) + seq for seq in input_ids]

        return torch.tensor(input_ids, device=device), torch.tensor(attention_mask, device=device), torch.tensor(augmentation, device=device)
    
    def get_response(seq, skip_words=""):
        return seq.split(skip_words)[1].strip()
    
    def evaluate(
        instruction=None,
        input=None,
        src_lang="",
        tgt_lang="",
        temperature=0.0,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        input_ids_mt = mt_input_features(
            input_text_m2m=input, 
            source_language=src_lang,
            langs_map=langs_map
        )
        
        _head = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        _tail = "\n\n### Response: Let's think step by step."
        if no_src_in_prompt:
            _prompts = [f"{_head}{_tail}" for _ in input]
        elif wrong_src_in_prompt:
            _prompts = [f"{_head}Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?{_tail}" for _ in input]
        else:
            _prompts = [f"{_head}{seq}{_tail}" for seq in input]
        input_ids_prompt = llm_input_features(
            input_texts_llm=_prompts,
            add_special_tokens=False
        )

        input_ids, attention_mask, augmentation = pad_and_mask(
            input_ids_mt, input_ids_prompt, tokenizer_llm.pad_token_id, next(model.parameters()).device
        )

        if "nllb" in mt_tokenizer_path.lower():
            forced_decoder_start_token_id = [tokenizer_mt.convert_tokens_to_ids(langs_map[lang]) for lang in tgt_lang]
        else:
            forced_decoder_start_token_id = [tokenizer_mt.get_lang_id(langs_map[lang]) for lang in tgt_lang]

        model_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            augmentation=augmentation,
            forced_decoder_start_token_id=forced_decoder_start_token_id,
        )
        llm_generate_ids = model_out[0]
        llm_outputs = tokenizer_llm.batch_decode(llm_generate_ids, skip_special_tokens=True)

        return llm_outputs


    for lang in test_langs:
        file_out_llm = f"{output_dir}/mgsm_{lang}.en.llm"
        result_out = f"{output_dir}/accuracy"
        with open(f"{testset_dir}/mgsm_{lang}.json") as f:
            testset = json.load(f)
        
        lines = [d["question"] for d in testset]
        answers = [d["answer"] for d in testset]

        print("Evaluating: " + lang + ", lines: " + str(len(lines)))

        hit_llm = 0

        for i in range(0, len(lines), batch_size):
            r = (i + batch_size) if (i + batch_size <= len(lines)) else len(lines)

            # inference
            llm_outputs = evaluate(
                input=lines[i: r],
                src_lang=lang_map_mm2l[lang],
                max_new_tokens=max_new_tokens,
                tgt_lang=[lang_map_mm2l[lang]]
            )

            # write generation to files
            llm_outputs = [output.replace("\n", " ") for output in llm_outputs]
            text = "\n".join(llm_outputs)
            with open(file_out_llm, "a", encoding="utf-8") as f:
                f.write(text + "\n")

            # calculate acc
            ground_truths = [extract_last_num(text) for text in answers[i: r]]

            results_llm = [extract_last_num(text) for text in llm_outputs]
            for result_p, ground_truth in zip(results_llm, ground_truths):
                if float(result_p) == float(ground_truth):
                    hit_llm += 1

        acc_llm = round(hit_llm / len(lines) * 100, 4)
        print(f"Accuracy for gsm_8k_{lang}.en.llm: {acc_llm}")
        with open(result_out, "a+") as f:
            f.write(f"Accuracy for gsm_8k_{lang}.en.llm: {acc_llm}\n")


if __name__ == "__main__":
    fire.Fire(main)
