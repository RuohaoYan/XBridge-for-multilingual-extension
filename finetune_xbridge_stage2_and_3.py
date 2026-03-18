import os
import sys
from typing import List, Optional, Union

import fire
import torch
import transformers
from datasets import load_dataset

from transformers import AutoTokenizer

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

from trainer_with_augmentation import TrainerWithAugmentation
from data_collator_with_additional_keys import DataCollatorWithAdditionalKeys

def train(
    # model/data params
    mt_path: str = "",
    llm_path: str = "",  # the only required argument
    mt_tokenizer_path: str = "",
    llm_tokenizer_path: str = "",
    data_path: str = "",
    output_dir: str = "",
    freeze_enc: bool = True,
    freeze_llm: bool = False,
    freeze_dec: bool = True,
    freeze_mapping_enc2llm: bool = False,
    freeze_mapping_llm2dec: bool = False,
    # training hyperparams
    task: str = "math",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    dec_lambda: float = 0.2,
    ot_lambda: float = 1.0,
    max_seq_len: int = 512,
    max_gen_len: int = 128,
    val_set_size: int = 2000,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: Optional[Union[str, bool]] = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    output_hidden_states: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"llm_path: {llm_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"freeze_enc: {freeze_enc}\n"
            f"freeze_llm: {freeze_llm}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"max_seq_len: {max_seq_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert llm_path, "Please specify a --llm_path, e.g. --llm_path='huggyllama/llama-7b'"
    
    langs_map_m2m = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
     'Russian': 'ru', 'Thai': 'th', 'Greek': 'el', 'Telugu': 'te',
     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
     'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur', 'Mongolian': 'mn', 'Kazakh': 'kk'}

    langs_map_nllb = {
        'English': 'eng_Latn', 'Swahili': 'swh_Latn', 'Chinese': 'zho_Hans', 'Bengali': 'ben_Beng',
        'German': 'deu_Latn', 'Spanish': 'spa_Latn', 'French': 'fra_Latn', 'Japanese': 'jpn_Jpan',
        'Russian': 'rus_Cyrl', 'Thai': 'tha_Thai', 'Mongolian': 'khk_Cyrl', 'Kazakh': 'kaz_Cyrl'
    }

    if 'nllb' in mt_path.lower():
        langs_map = langs_map_nllb
    else:
        langs_map = langs_map_m2m
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    
    # prepare tokenizer
    tokenizer_mt = AutoTokenizer.from_pretrained(mt_tokenizer_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_tokenizer_path)
    if "llama3" in llm_path or "llama-3" in llm_path:
        tokenizer_llm.pad_token_id = 128002
    elif tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token_id = 0
    tokenizer_llm.padding_side = "left"  # Allow batched inference
    
    def mt_input_features(input_text_m2m, source_language, langs_map, max_length=-1):
        tokenizer_mt.src_lang = langs_map[source_language]
        encoding_m2m = tokenizer_mt(
            input_text_m2m,
            padding=False,
            truncation=False if max_length == -1 else True,
            max_length=None if max_length == -1 else max_length,
            return_tensors=None,
            add_special_tokens=True
        )
        input_ids_m2m = encoding_m2m.input_ids
        attention_mask_m2m = encoding_m2m.attention_mask
        return input_ids_m2m

    def llm_input_features(input_texts_llm, max_length=-1, add_special_tokens=True):
        encoding_llm = tokenizer_llm(
            input_texts_llm,
            padding=False,
            truncation=False if max_length == -1 else True,
            max_length=None if max_length == -1 else max_length,
            return_tensors=None,
            add_special_tokens=add_special_tokens
        )
        return encoding_llm["input_ids"]

    def generate_inputs(data_point, task):
        if task == "math":
            mt_input = data_point['instruction_non_en'] + ("\n\n" + data_point['input_non_en'] if data_point['instruction_non_en'] and data_point['input_non_en'] else data_point['input_non_en'])
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{mt_input}\n\n### Response: Let's think step by step."
        else:
            raise ValueError("Please specify task prompt!")
        return mt_input, prompt
    
    def generate_and_tokenize_prompt(data_point):
        result = {}

        mt_input, prompt = generate_inputs(data_point, task = task)
        
        if task == "translation":
            enc_lang = data_point["input_lang"]
            dec_lang = data_point["output_lang"]
        else:
            enc_lang = dec_lang = data_point["non_en_lang"]

        # input sentence
        input_ids_mt = mt_input_features(
            input_text_m2m=mt_input,
            source_language=enc_lang,
            langs_map=langs_map,
            max_length=max_seq_len
        )
        
        # instruction
        input_ids_prompt = llm_input_features(
            input_texts_llm=prompt, 
            add_special_tokens=False,
            max_length=max_seq_len
        )
        
        # output sentence
        llm_labels = llm_input_features(
            input_texts_llm=f"{data_point['output_en']}{tokenizer_llm.eos_token}", 
            add_special_tokens=False,
            max_length=max_seq_len,
        )
        
        mt_labels = mt_input_features(
            input_text_m2m=data_point["output_en"], 
            source_language="English",
            langs_map=langs_map,
            max_length=max_seq_len,
        )
        
        decoder_labels = mt_input_features(
            input_text_m2m=data_point["output_non_en"], 
            source_language=dec_lang,
            langs_map=langs_map,
            max_length=max_seq_len,
        )

        result["input_ids"] = input_ids_mt + input_ids_prompt + llm_labels
        result["attention_mask"] = [1] * len(result["input_ids"])
        result["augmentation"] = [1] * len(input_ids_mt) + [2] * len(input_ids_prompt) + [3] * len(llm_labels)
        result["decoder_input_ids"] = [tokenizer_mt.eos_token_id] + decoder_labels
        result["mt_labels"] = mt_labels

        result["labels"] = result["input_ids"].copy()
        result["decoder_labels"] = result["decoder_input_ids"][1:] + [-100]

        return result


    def prepare_model_for_training(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}"
        )
        
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)
        return model
    
    # load_dataset
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].map(generate_and_tokenize_prompt, num_proc=8).shuffle()
        )
        val_data = (
            train_val["test"].map(generate_and_tokenize_prompt, num_proc=8).shuffle()
        )
    else:
        train_data = data["train"].map(generate_and_tokenize_prompt, num_proc=8).shuffle()
        val_data = None

    if llm_path != llm_tokenizer_path:
        config = XBridgeConfig.from_pretrained(llm_path)
        config.freeze_enc = freeze_enc
        config.freeze_llm = freeze_llm
        config.freeze_dec = freeze_dec
        config.dec_lambda = dec_lambda
        config.ot_lambda = ot_lambda
        config.freeze_mapping_enc2llm = freeze_mapping_enc2llm
        config.freeze_mapping_llm2dec = freeze_mapping_llm2dec
        
        model = LlamaForCasualLMWithXBridge.from_pretrained(
            llm_path,
            config=config,
            torch_dtype=torch.float32,
            device_map="auto",
            len_tokenizer_llm=len(tokenizer_llm)
        )
        model.model_mt.model.shared.weight.requires_grad = False
        model.model_mt.lm_head.weight = model.model_mt.model.shared.weight
        model.model_mt.lm_head._hf_hook.execution_device=model.model_mt.model.shared.weight.device.index
    else:
        model_config = XBridgeConfig(
            mt_path=mt_path, 
            llm_path=llm_path,
            dec_lambda=dec_lambda,
            ot_lambda=ot_lambda,
            max_gen_len=max_gen_len, 
            mt_pad_token_id=tokenizer_mt.pad_token_id,
            mt_eos_token_id=tokenizer_mt.eos_token_id,
            llm_bos_token_id=tokenizer_llm.bos_token_id,
            llm_eos_token_id=tokenizer_llm.eos_token_id,
            llm_pad_token_id=tokenizer_llm.pad_token_id,
            freeze_enc=freeze_enc,
            freeze_llm=freeze_llm,
            freeze_dec=freeze_dec,
            freeze_mapping_enc2llm=freeze_mapping_enc2llm,
            freeze_mapping_llm2dec=freeze_mapping_llm2dec,
        )
        model = LlamaForCasualLMWithXBridge(model_config, is_training=True, len_tokenizer_llm=len(tokenizer_llm))
    
    model.model_llm.config.output_hidden_states = output_hidden_states
    model = prepare_model_for_training(model)
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = TrainerWithAugmentation(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            # torch_empty_cache_steps=10,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorWithAdditionalKeys(
            tokenizer_llm, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    model.model_llm.config.use_cache = False
    model.model_mt.config.use_cache = False
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # model = model.half()
    model.config.use_cache = True
    model.model_llm.config.use_cache = True
    model.model_mt.config.use_cache = True
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
