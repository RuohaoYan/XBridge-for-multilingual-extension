import fire
import gradio as gr

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

lang_map_mm2l = {
    "af":"Afrikaans","am":"Amharic","ar":"Arabic","hy":"Armenian","as":"Assamese",
    "ast":"Asturian","az":"Azerbaijani","be":"Belarusian","bn":"Bengali","bs":"Bosnian",
    "bg":"Bulgarian","my":"Burmese","ca":"Catalan","ceb":"Cebuano","zh":"Chinese",
    "hr":"Croatian","cs":"Czech","da":"Danish","nl":"Dutch","en":"English",
    "et":"Estonian","tl":"Filipino","fi":"Finnish","fr":"French","ff":"Fulah",
    "gl":"Galician","lg":"Ganda","ka":"Georgian","de":"German","el":"Greek",
    "gu":"Gujarati","ha":"Hausa","he":"Hebrew","hi":"Hindi","hu":"Hungarian",
    "is":"Icelandic","ig":"Igbo","id":"Indonesian","ga":"Irish","it":"Italian",
    "ja":"Japanese","jv":"Javanese","kea":"Kabuverdianu","kam":"Kamba","kn":"Kannada",
    "kk":"Kazakh","km":"Khmer","ko":"Korean","ky":"Kyrgyz","lo":"Lao",
    "lv":"Latvian","ln":"Lingala","lt":"Lithuanian","luo":"Luo","lb":"Luxembourgish",
    "mk":"Macedonian","ms":"Malay","ml":"Malayalam","mt":"Maltese","mi":"Maori",
    "mr":"Marathi","mn":"Mongolian","ne":"Nepali","ns":"Northern Sotho","no":"Norwegian",
    "ny":"Nyanja","oc":"Occitan","or":"Oriya","om":"Oromo","ps":"Pashto",
    "fa":"Persian","pl":"Polish","pt":"Portuguese","pa":"Punjabi","ro":"Romanian",
    "ru":"Russian","sr":"Serbian","sn":"Shona","sd":"Sindhi","sk":"Slovak",
    "sl":"Slovenian","so":"Somali","ku":"Sorani Kurdish","es":"Spanish","sw":"Swahili",
    "sv":"Swedish","tg":"Tajik","ta":"Tamil","te":"Telugu","th":"Thai",
    "tr":"Turkish","uk":"Ukrainian","umb":"Umbundu","ur":"Urdu","uz":"Uzbek",
    "vi":"Vietnamese","cy":"Welsh","wo":"Wolof","xh":"Xhosa","yo":"Yoruba","zu":"Zulu"
}

lang_map_flores2mm = {
    "af":"afr","am":"amh","ar":"ara","hy":"hye","as":"asm",
    "ast":"ast","az":"azj","be":"bel","bn":"ben","bs":"bos",
    "bg":"bul","my":"mya","ca":"cat","ceb":"ceb","zh":"zho_simpl",
    "hr":"hrv","cs":"ces","da":"dan","nl":"nld","en":"eng",
    "et":"est","tl":"tgl","fi":"fin","fr":"fra","ff":"ful",
    "gl":"glg","lg":"lug","ka":"kat","de":"deu","el":"ell",
    "gu":"guj","ha":"hau","he":"heb","hi":"hin","hu":"hun",
    "is":"isl","ig":"ibo","id":"ind","ga":"gle","it":"ita",
    "ja":"jpn","jv":"jav","kea":"kea","kam":"kam","kn":"kan",
    "kk":"kaz","km":"khm","ko":"kor","ky":"kir","lo":"lao",
    "lv":"lav","ln":"lin","lt":"lit","luo":"luo","lb":"ltz",
    "mk":"mkd","ms":"msa","ml":"mal","mt":"mlt","mi":"mri",
    "mr":"mar","mn":"mon","ne":"npi","ns":"nso","no":"nob",
    "ny":"nya","oc":"oci","or":"ory","om":"orm","ps":"pus",
    "fa":"fas","pl":"pol","pt":"por","pa":"pan","ro":"ron",
    "ru":"rus","sr":"srp","sn":"sna","sd":"snd","sk":"slk",
    "sl":"slv","so":"som","ku":"ckb","es":"spa","sw":"swh",
    "sv":"swe","tg":"tgk","ta":"tam","te":"tel","th":"tha",
    "tr":"tur","uk":"ukr","umb":"umb","ur":"urd","uz":"uzb",
    "vi":"vie","cy":"cym","wo":"wol","xh":"xho","yo":"yor","zu":"zul"
}

langs_map_nllb = {
    "Afrikaans":"afr_Latn", "Amharic":"amh_Ethi", "Arabic":"arb_Arab", "Armenian":"hye_Armn", "Assamese":"asm_Beng", 
    "Asturian":"ast_Latn", "Azerbaijani":"azj_Latn", "Belarusian":"bel_Cyrl", "Bengali":"ben_Beng", "Bosnian":"bos_Latn", 
    "Bulgarian":"bul_Cyrl", "Burmese":"mya_Mymr", "Catalan":"cat_Latn", "Cebuano":"ceb_Latn", "Chinese":"zho_Hans", 
    "Croatian":"hrv_Latn", "Czech":"ces_Latn", "Danish":"dan_Latn", "Dutch":"nld_Latn", "English":"eng_Latn", 
    "Estonian":"est_Latn", "Filipino":"tgl_Latn", "Finnish":"fin_Latn", "French":"fra_Latn", "Fulah":"ful_Latn", 
    "Galician":"glg_Latn", "Ganda":"lug_Latn", "Georgian":"kat_Geor", "German":"deu_Latn", "Greek":"ell_Grek", 
    "Gujarati":"guj_Gujr", "Hausa":"hau_Latn", "Hebrew":"heb_Hebr", "Hindi":"hin_Deva", "Hungarian":"hun_Latn", 
    "Icelandic":"isl_Latn", "Igbo":"ibo_Latn", "Indonesian":"ind_Latn", "Irish":"gle_Latn", "Italian":"ita_Latn", 
    "Japanese":"jpn_Jpan", "Javanese":"jav_Latn", "Kabuverdianu":"kea_Latn", "Kamba":"kam_Latn", "Kannada":"kan_Knda", 
    "Kazakh":"kaz_Cyrl", "Khmer":"khm_Khmr", "Korean":"kor_Hang", "Kyrgyz":"kir_Cyrl", "Lao":"lao_Laoo", "Latvian":"lav_Latn", 
    "Lingala":"lin_Latn", "Lithuanian":"lit_Latn", "Luo":"luo_Latn", "Luxembourgish":"ltz_Latn", "Macedonian":"mkd_Cyrl", 
    "Malay":"zsm_Latn", "Malayalam":"mal_Mlym", "Maltese":"mlt_Latn", "Maori":"mri_Latn", "Marathi":"mar_Deva", 
    "Mongolian":"khk_Cyrl", "Nepali":"npi_Deva", "Northern Sotho":"nso_Latn", "Norwegian":"nob_Latn", "Nyanja":"nya_Latn", 
    "Occitan":"oci_Latn", "Oriya":"ory_Orya", "Oromo":"gaz_Latn", "Pashto":"pbt_Arab", "Persian":"pes_Arab", 
    "Polish":"pol_Latn", "Portuguese":"por_Latn", "Punjabi":"pan_Guru", "Romanian":"ron_Latn", "Russian":"rus_Cyrl", 
    "Serbian":"srp_Cyrl", "Shona":"sna_Latn", "Sindhi":"snd_Arab", "Slovak":"slk_Latn", "Slovenian":"slv_Latn", 
    "Somali":"som_Latn", "Sorani Kurdish":"ckb_Arab", "Spanish":"spa_Latn", "Swahili":"swh_Latn", "Swedish":"swe_Latn", 
    "Tajik":"tgk_Cyrl", "Tamil":"tam_Taml", "Telugu":"tel_Telu", "Thai":"tha_Thai", "Turkish":"tur_Latn", 
    "Ukrainian":"ukr_Cyrl", "Umbundu":"umb_Latn", "Urdu":"urd_Arab", "Uzbek":"uzn_Latn", "Vietnamese":"vie_Latn", 
    "Welsh":"cym_Latn", "Wolof":"wol_Latn", "Xhosa":"xho_Latn", "Yoruba":"yor_Latn", "Zulu":"zul_Latn"
}


def main(
    model_path: str,
    mt_tokenizer_path: str,
    llm_tokenizer_path: str,
    max_gen_len: int = 256
):
    langs = ['Afrikaans', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Asturian', 'Azerbaijani', 'Belarusian', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', 'Catalan', 'Cebuano', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Filipino', 'Finnish', 'French', 'Fulah', 'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Gujarati', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kabuverdianu', 'Kamba', 'Kannada', 'Kazakh', 'Khmer', 'Kyrgyz', 'Lao', 'Latvian', 'Lingala', 'Lithuanian', 'Luo', 'Luxembourgish', 'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Nepali', 'Northern Sotho', 'Norwegian', 'Nyanja', 'Occitan', 'Oriya', 'Oromo', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Shona', 'Sindhi', 'Slovak', 'Slovenian', 'Somali', 'Sorani Kurdish', 'Spanish', 'Swahili', 'Swedish', 'Tajik', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', 'Vietnamese', 'Welsh', 'Wolof', 'Xhosa', 'Yoruba', 'Zulu']

    langs_map = langs_map_nllb

    # load tokenizer
    tokenizer_mt = AutoTokenizer.from_pretrained(mt_tokenizer_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_tokenizer_path)
    if "llama3" in llm_tokenizer_path or "llama-3" in llm_tokenizer_path:
        tokenizer_llm.pad_token_id = 128002
    else:
        tokenizer_llm.pad_token_id = 0
    tokenizer_llm.padding_side = "left"

    # load model
    config = XBridgeConfig.from_pretrained(model_path)
    config.max_gen_len = max_gen_len
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        len_tokenizer_llm=len(tokenizer_llm)
    )
    model.model_mt.lm_head.weight = model.model_mt.model.shared.weight
    # model.model_mt.lm_head._hf_hook.execution_device=model.model_mt.model.shared.weight.device.index

    model.eval()

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
        return encoding_m2m["input_ids"]

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


    def pad_and_mask(input_ids_mt, input_ids_prompt, pad_token_id):
        input_ids = [mt + prompt for mt, prompt in zip(input_ids_mt, input_ids_prompt)]
        
        max_len = max(len(seq) for seq in input_ids)
        augmentation = [[0] * (max_len - len(input_ids[i])) + [1] * len(input_ids_mt[i]) + [2] * len(input_ids_prompt[i]) for i in range(len(input_ids_mt))]
        attention_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in input_ids]
        input_ids = [[pad_token_id] * (max_len - len(seq)) + seq for seq in input_ids]

        return torch.tensor(input_ids).cuda(), torch.tensor(attention_mask).cuda(), torch.tensor(augmentation).cuda()


    def interact_with_model(input_text, src_lang, tgt_lang):
        mt_input = input_text
        
        input_ids_mt = mt_input_features(
            input_text_m2m=[mt_input], 
            source_language=src_lang,
            langs_map=langs_map
        )

        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{mt_input}\n\n### Response:\n"
            
        input_ids_prompt = llm_input_features(
            input_texts_llm=[prompt], 
            add_special_tokens=False
        )

        input_ids, attention_mask, augmentation = pad_and_mask(input_ids_mt, input_ids_prompt, tokenizer_llm.pad_token_id)

        llm_generate_ids, mt_dec_generate_ids = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            augmentation=augmentation,
            forced_decoder_start_token_id=[tokenizer_mt.convert_tokens_to_ids(langs_map[tgt_lang])]
        )
        llm_outputs = tokenizer_llm.batch_decode(llm_generate_ids, skip_special_tokens=True)
        mt_dec_outputs = tokenizer_mt.batch_decode(mt_dec_generate_ids, skip_special_tokens=True)

        return llm_outputs[0], mt_dec_outputs[0]


    with gr.Blocks(title="Custom Model Interface") as demo:
        gr.Markdown("## 🌐 Custom Instruction-following XBridge Interface")

        with gr.Row():
            src_lang = gr.Dropdown(
                choices = langs, label="Source Language"
            )
            tgt_lang = gr.Dropdown(
                choices = langs, label="Target Language"
            )

        input_text = gr.Textbox(label="Input Text", lines=4, placeholder="Enter the input text here")

        with gr.Row():
            output_lr = gr.Textbox(label="Output (Low-resource Language)", lines=4)
            output_en = gr.Textbox(label="Output (English Reference)", lines=4)

        submit_btn = gr.Button("Run Model")

        submit_btn.click(
            fn=interact_with_model,
            inputs=[input_text, src_lang, tgt_lang],
            outputs=[output_en, output_lr]
        )

    demo.launch(server_name="0.0.0.0", server_port=6654, share=True)

if __name__ == "__main__":
    fire.Fire(main)