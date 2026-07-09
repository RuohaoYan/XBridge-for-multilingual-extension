import os
import sys
import ast

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
    

def main(
    load_8bit: bool = False,
    mt_tokenizer_path: str = "",
    llm_tokenizer_path: str = "",
    base_model: str = "",
    batch_size: int = "",
    max_new_tokens: int = 512,
    testset_dir: str = "",
    output_dir: str = "",
    error_file: str = "",
    trans_langs: str = "",
    test_langs: str = "",
    mode: str = "supervised"
):
    if not trans_langs and test_langs:
        trans_langs = test_langs

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(trans_langs, str):
        trans_langs = [x.strip() for x in trans_langs.split(",") if x.strip()]
    elif isinstance(trans_langs, tuple):
        trans_langs = list(trans_langs)
    else:
        trans_langs = list(trans_langs)

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

    def pad_and_mask(input_ids_mt, pad_token_id, device):
        input_ids = [seq for seq in input_ids_mt]
        
        max_len = max(len(seq) for seq in input_ids)
        augmentation = [[0] * (max_len - len(input_ids[i])) + [1] * len(input_ids_mt[i]) for i in range(len(input_ids_mt))]
        attention_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in input_ids]
        input_ids = [[pad_token_id] * (max_len - len(seq)) + seq for seq in input_ids]

        return torch.tensor(input_ids, device=device), torch.tensor(attention_mask, device=device), torch.tensor(augmentation, device=device)
    
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

        input_ids, attention_mask, augmentation = pad_and_mask(
            input_ids_mt, tokenizer_llm.pad_token_id, next(model.parameters()).device
        )

        if "nllb" in mt_tokenizer_path.lower():
            forced_decoder_start_token_id = tokenizer_mt.convert_tokens_to_ids([langs_map[lang] for lang in tgt_lang])
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
        mt_dec_outputs_list = []
        if len(model_out) > 1:
            mt_dec_outputs_list = model_out[1]
            if not isinstance(mt_dec_outputs_list, list):
                mt_dec_outputs_list = [mt_dec_outputs_list]
            mt_dec_outputs_list = [
                tokenizer_mt.batch_decode(mt_dec_generate_ids, skip_special_tokens=True)
                for mt_dec_generate_ids in mt_dec_outputs_list
            ]

        return llm_outputs, mt_dec_outputs_list


    for idx, lang in enumerate(trans_langs):
        file_in = f"{testset_dir}/{lang_map_flores2mm[lang]}.devtest"
        file_out_llm = f"{output_dir}/{mode}.{lang}-en.en.llm"

        with open(file_in, "r") as fin:
            lines = fin.readlines()
        
        lines = [line[: -1] for line in lines]
        
        print(f"Evaluating: {lang}-x...")
        
        for i in range(0, len(lines), batch_size):
            r = (i + batch_size) if (i + batch_size <= len(lines)) else len(lines)
            
            llm_outputs, mt_dec_outputs_list = evaluate(
                input=lines[i: r], 
                src_lang=lang_map_mm2l[lang],
                tgt_lang=[lang_map_mm2l[tgt_lang] for tgt_lang in trans_langs],
                max_new_tokens=max_new_tokens
            )
            
            llm_outputs = [output.replace("\n", " ") for output in llm_outputs]
            text = "\n".join(llm_outputs)
            with open(file_out_llm, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            
            if mt_dec_outputs_list:
                for mt_dec_outputs, tgt_lang in zip(mt_dec_outputs_list, trans_langs):
                    file_out_mt = f"{output_dir}/{mode}.{lang}-{tgt_lang}.{tgt_lang}.mt"
                    mt_dec_outputs = [output.replace("\n", " ") for output in mt_dec_outputs]
                    text = "\n".join(mt_dec_outputs)
                    with open(file_out_mt, "a", encoding="utf-8") as f:
                        f.write(text + "\n")

if __name__ == "__main__":
    fire.Fire(main)
