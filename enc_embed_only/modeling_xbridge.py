import math
from math import ceil, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GenerationConfig
from transformers import M2M100ForConditionalGeneration, MT5ForConditionalGeneration, LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.utils import ModelOutput
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from transformers.modeling_outputs import BaseModelOutput

from transformers.activations import ACT2FN

class XBridgeConfig(LlamaConfig):
    def __init__(
        self, 
        mt_path = None, 
        llm_path = None,
        dec_lambda = 0.2,
        ot_lambda = 1.0,
        max_gen_len = 128, 
        mt_pad_token_id = 0,
        mt_eos_token_id = 2,
        llm_bos_token_id = 1, 
        llm_eos_token_id = 2,
        llm_pad_token_id = 0, 
        freeze_enc = True,
        freeze_llm = True,
        freeze_dec = True,
        freeze_mapping_enc2llm = False,
        freeze_mapping_llm2dec = False,
        llm_only = True,
        train_device_map = "auto",
        use_embed_fusion = False,
        mt_vocab_size = 32000,
        **kwargs):
        super().__init__(**kwargs)
        self.mt_path = mt_path
        self.llm_path = llm_path
        self.dec_lambda = dec_lambda
        self.ot_lambda = ot_lambda
        self.max_gen_len = max_gen_len
        self.mt_pad_token_id = mt_pad_token_id
        self.mt_eos_token_id = mt_eos_token_id
        self.llm_bos_token_id = llm_bos_token_id
        self.llm_eos_token_id = llm_eos_token_id
        self.llm_pad_token_id = llm_pad_token_id
        self.freeze_enc = freeze_enc
        self.freeze_llm = freeze_llm
        self.freeze_dec = freeze_dec
        self.freeze_mapping_enc2llm = freeze_mapping_enc2llm
        self.freeze_mapping_llm2dec = freeze_mapping_llm2dec
        self.llm_only = llm_only
        self.train_device_map = train_device_map
        self.use_embed_fusion = use_embed_fusion
        self.mt_vocab_size = mt_vocab_size

class MLP(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, hidden_act, rms_norm_eps):
        super(MLP, self).__init__()
        self.input_layernorm = LlamaRMSNorm(dim_in, eps=rms_norm_eps)
        self.linear1 = nn.Linear(dim_in, dim_mid)
        self.act_fn = ACT2FN[hidden_act]
        self.linear2 = nn.Linear(dim_mid, dim_out)

    def forward(self, x):
        x = self.input_layernorm(x)
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

class Mapping(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, layer_num, hidden_act="silu", rms_norm_eps=1e-6):
        super(Mapping, self).__init__()
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, dim_out), requires_grad=True
        )
        
        assert layer_num >= 1, "layer_num should be >= 1"
        layers = []
        if layer_num == 1:
            layers = [MLP(dim_in, dim_mid, dim_out, hidden_act, rms_norm_eps)]
        else:
            layers.append(MLP(dim_in, dim_mid, dim_mid, hidden_act, rms_norm_eps))
            for _ in range(layer_num - 2):
                layers.append(MLP(dim_mid, dim_mid, dim_mid, hidden_act, rms_norm_eps))
            layers.append(MLP(dim_mid, dim_mid, dim_out, hidden_act, rms_norm_eps))

        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states):
        return self.layers(hidden_states)

    def get_embed(self):
        return self.end_boundary


class LlamaForCasualLMWithXBridge(LlamaForCausalLM):
    config_class = XBridgeConfig
    def __init__(self, config: XBridgeConfig, is_training=False, len_tokenizer_llm=None):
        LlamaPreTrainedModel.__init__(self, config)

        self.config = config
        
        self.mt_pad_token_id = config.mt_pad_token_id
        self.mt_eos_token_id = config.mt_eos_token_id
        self.llm_pad_token_id = config.llm_pad_token_id
        self.llm_bos_token_id = config.llm_bos_token_id
        self.llm_eos_token_id = config.llm_eos_token_id
        
        self.dec_lambda = config.dec_lambda
        self.ot_lambda = config.ot_lambda
        
        self.config_mt = AutoConfig.from_pretrained(config.mt_path)
        self.config_llm = AutoConfig.from_pretrained(config.llm_path)
        
        # load in model weights at training stage, otherwise just create an empty model and load from from_pretrained()
        if is_training:
            device_map = getattr(config, "train_device_map", "auto")
            load_dtype = torch.float16
            self.model_mt = AutoModelForSeq2SeqLM.from_pretrained(
                config.mt_path, device_map=device_map, torch_dtype=load_dtype
            )
            self.model_llm = AutoModelForCausalLM.from_pretrained(
                config.llm_path, device_map=device_map, torch_dtype=load_dtype
            )
            if config.llm_only:
                # Encoder-only training does not need NLLB decoder weights.
                if hasattr(self.model_mt, "model") and hasattr(self.model_mt.model, "decoder"):
                    del self.model_mt.model.decoder
                if hasattr(self.model_mt, "lm_head"):
                    del self.model_mt.lm_head
                torch.cuda.empty_cache()
        else:
            if "nllb" in self.config.mt_path.lower():
                self.model_mt = M2M100ForConditionalGeneration(self.config_mt)
            else:
                self.model_mt = MT5ForConditionalGeneration(self.config_mt)
            self.model_llm = LlamaForCausalLM(self.config_llm)

        if len_tokenizer_llm and len_tokenizer_llm > self.model_llm.vocab_size:
            self.model_llm.resize_token_embeddings(len_tokenizer_llm)

        # mt model message
        self.config.mt_vocab_size = self.model_mt.config.vocab_size

        if config.freeze_enc:     # freeze encoder
            for name, parameter in self.model_mt.get_encoder().named_parameters():
                parameter.requires_grad = False
        if config.freeze_dec:   # freeze decoder
            decoder = getattr(getattr(self.model_mt, "model", None), "decoder", None)
            if decoder is not None:
                for name, parameter in decoder.named_parameters():
                    parameter.requires_grad = False
                    if 'encoder_attn' in name:      # train decoder cross-attention
                        parameter.requires_grad = True
                        print(f"Unfroze: {name}")
        
        print('MT model size:', sum(param.numel() for param in self.model_mt.parameters()) / 1000000, 'MB')
        
        # llm message
        if config.freeze_llm:     # freeze llm
            for name, parameter in self.model_llm.named_parameters():
                parameter.requires_grad = False
        print('llm size:', sum(param.numel() for param in self.model_llm.parameters()) / 1000000, 'MB')

        # dimension mapping
        if 'bert' in config.mt_path or 'Qwen' in config.mt_path:
            self.mt_hidden_size = self.config_mt.hidden_size
        elif 'GPT' in config.mt_path:
            self.mt_hidden_size = self.config_mt.n_embd
        else:
            self.mt_hidden_size = self.config_mt.d_model
        
        self.mapping_enc2llm = Mapping(self.mt_hidden_size, self.mt_hidden_size*4, self.config_llm.hidden_size, 1, hidden_act=config.hidden_act, rms_norm_eps=config.rms_norm_eps).to(self.model_mt.device)      # trainable
        # Embedding-fusion branch: maps raw NLLB token embeddings (d_model) to LLM hidden,
        # added (residual) to mapping_enc2llm(Enc(x)) when config.use_embed_fusion=True.
        self.mapping_embed = nn.Sequential(
            MLP(self.mt_hidden_size, self.mt_hidden_size*4, self.config_llm.hidden_size, config.hidden_act, config.rms_norm_eps)
        ).to(self.model_mt.device)
        self.mapping_llm2dec = Mapping(self.config_llm.hidden_size, self.config_llm.hidden_size*2, self.mt_hidden_size, 2, hidden_act=config.hidden_act, rms_norm_eps=config.rms_norm_eps)
        if is_training and config.llm_only:
            self.mapping_llm2dec = self.mapping_llm2dec.to("cpu")
        else:
            self.mapping_llm2dec = self.mapping_llm2dec.to(self.model_mt.device)

        if config.freeze_mapping_enc2llm:     # freeze enc-llm mapping
            for name, parameter in self.mapping_enc2llm.named_parameters():
                parameter.requires_grad = False
        
        if config.freeze_mapping_llm2dec:     # freeze llm-dec mapping
            for name, parameter in self.mapping_llm2dec.named_parameters():
                parameter.requires_grad = False
        
        print('mapping_enc2llm layer size:', sum(param.numel() for param in self.mapping_enc2llm.parameters()) / 1000000, 'MB')
        print('mapping_llm2dec layer size:', sum(param.numel() for param in self.mapping_llm2dec.parameters()) / 1000000, 'MB')
        
        

    
    def squeeze_pad(self, hidden_states, masks):
        x_01 = (masks != 0).long()

        seq_num_len = x_01.size(1)
        offset = torch.tensor([(i + 1) for i in range(seq_num_len)], dtype=torch.long).to(x_01.device)
        offset = offset.unsqueeze(dim=0).expand_as(x_01)
        x_01 *= offset
        _, idx = x_01.sort(1, descending=False)

        masks = masks.gather(1, idx)
        idx = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states.gather(1, idx)

        bs, seq_len, dim = hidden_states.size()
        masks_sum = torch.sum(masks, dim=0)
        idx = masks_sum > 0
        idx = idx.unsqueeze(dim=0).expand_as(masks)
        masks = masks[idx]
        idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states[idx_ex]
        hidden_states = hidden_states.view(bs, -1, dim)
        masks = masks.view(bs, -1)

        return hidden_states, masks, idx

    def forward(
        self,
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        augmentation: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mt_labels: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_labels: Optional[torch.LongTensor] = None,
        forced_decoder_start_token_id: Optional[Union[int, List]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        bs = input_ids.size(0)
        llm_embedding_layer = self.model_llm.get_input_embeddings()
        bos = torch.full((bs, 1), self.llm_bos_token_id, dtype=torch.long, device=input_ids.device)
        bos_embedding = llm_embedding_layer(bos).to(input_ids.device)
        bos_mask = torch.ones([bs, 1], dtype=torch.long, device=input_ids.device)    
        
        end_boundary = self.mapping_enc2llm.get_embed()
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)]).to(input_ids.device)
        boundary_mask = torch.ones([bs, 1], dtype=torch.long).to(input_ids.device)
        
        mt_mask = (augmentation == 1)
        prompt_mask = (augmentation == 2)
        label_mask = (augmentation == 3)
        
        # padding
        def pad_sequences(sequences, max_len, pad_token_id, padding_side="left"):
            if max_len == 0:
                return torch.full((len(sequences), 1), pad_token_id, dtype=torch.long, device=input_ids.device)
            if padding_side == "right":
                return torch.tensor(
                    [(seq + [pad_token_id] * (max_len - len(seq))) if seq else [pad_token_id] * max_len for seq in sequences],
                    dtype=torch.long,
                    device=input_ids.device
                )
            else:
                return torch.tensor(
                    [([pad_token_id] * (max_len - len(seq)) + seq) if seq else [pad_token_id] * max_len for seq in sequences],
                    dtype=torch.long,
                    device=input_ids.device
                )
        
        def left_to_right_padding(matrix, origin_pad_token_id, new_pad_token_id):
            bsz, seq_len = matrix.shape
            result = torch.full_like(matrix, fill_value=new_pad_token_id)

            for i in range(bsz):
                ex = matrix[i]
                non_pad = ex[ex != origin_pad_token_id]
                result[i, :len(non_pad)] = non_pad

            return result
        
        def embedding_tokens(input_ids_all, mask_all, pad_token_id, padding_side="left", encoder=None, mapping=True):
            if mask_all.any():
                input_ids_selected = [input_ids_all[i][mask_all[i]].tolist() for i in range(bs)]
                max_len = max(len(seq) for seq in input_ids_selected) if input_ids_selected else 0
                input_ids_selected = pad_sequences(input_ids_selected, max_len, pad_token_id, padding_side)
                mask_selected = (input_ids_selected != pad_token_id).long()
                if encoder is not None:
                    if mapping:
                        # Embed-only ablation: skip NLLB encoder, source feature = mapping_embed(E(x)).
                        nllb_emb = self.model_mt.get_input_embeddings()(input_ids_selected)
                        nllb_emb = nllb_emb.to(self.mapping_embed[0].linear1.weight.device)
                        nllb_emb = nllb_emb * mask_selected.unsqueeze(-1).to(nllb_emb.dtype)
                        embedding = self.mapping_embed(nllb_emb)
                    else:
                        # mapping=False: raw encoder output for ot_loss path (unused when llm_only/ot_lambda=0).
                        mt_encoder_outputs = encoder(
                            input_ids=input_ids_selected,
                            attention_mask=mask_selected,
                            output_hidden_states=True
                        )
                        embedding = mt_encoder_outputs[0].to(self.mapping_enc2llm.layers[0].linear1.weight.device)
                else:
                    embedding = llm_embedding_layer(input_ids_selected)
            else:
                embedding = torch.zeros(bs, 0, self.model_llm.config.hidden_size, device=self.model_llm.device)
                mask_selected = torch.zeros(bs, 0, device=self.model_llm.device)
                input_ids_selected = torch.zeros(bs, 0, device=self.model_llm.device)
            
            return embedding, mask_selected, input_ids_selected

        embedding_prompt, mask_prompt, input_ids_prompt = embedding_tokens(input_ids, prompt_mask, self.llm_pad_token_id)
        embedding_labels, mask_label, input_ids_labels = embedding_tokens(input_ids, label_mask, self.llm_pad_token_id)
        mt_hidden_state, attention_mask_mt, input_ids_mt = embedding_tokens(input_ids, mt_mask, self.mt_pad_token_id, "right", self.model_mt.get_encoder())

        # concat for such seq: (<bos> // mt_hidden_state (src) // sep // prompt (instruction) // label_embedding (tgt))
        llm_input_embedding = torch.cat([bos_embedding, mt_hidden_state.to(input_ids.device), end_boundary, embedding_prompt.to(input_ids.device), embedding_labels.to(input_ids.device)], dim=1)
        llm_input_mask = torch.cat([bos_mask, attention_mask_mt.to(input_ids.device), boundary_mask, mask_prompt.to(input_ids.device), mask_label.to(input_ids.device)], dim=1)

        llm_input_embedding, llm_input_mask, _ = self.squeeze_pad(llm_input_embedding, llm_input_mask.to(llm_input_embedding.device))

        def cost(x, y, type="l2"):
            bsz, len1, dim = x.size(0), x.size(1), x.size(2)
            len2 = y.size(-2)
            tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
            ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)

            if type == "l2":
                res = torch.linalg.norm(tx - ty, dim=-1)
            else:
                f_simi = torch.nn.CosineSimilarity(dim=-1)
                res = 1.0 - f_simi(tx, ty)
            return res

        def compute_op_distance_min(x, x_mask, y, y_mask, pad_val=4e6, eps=1e-8):
            x_valid = (x_mask == 1).to(torch.bool)
            y_valid = (y_mask == 1).to(torch.bool)

            # calculate cost
            C = cost(x, y, "cosin")
            C = C.masked_fill((~x_valid).unsqueeze(-1), pad_val).masked_fill((~y_valid).unsqueeze(-2), pad_val)

            # move source to target
            d_min = C.min(dim=-1)[0]
            
            # calculate weight
            norms = torch.linalg.norm(x, dim=-1)
            norms = norms * x_valid.to(norms.dtype)
            denom = norms.sum(dim=-1, keepdim=True).clamp(min=eps)
            weight = norms / denom
            
            # calculate loss
            y_has_token = y_valid.any(dim=-1)
            sample_loss = (d_min * weight.detach()).sum(dim=-1)
            sample_loss = sample_loss * y_has_token.to(sample_loss.dtype) 

            if y_has_token.any():
                loss = sample_loss.sum() / y_has_token.sum()
            else:
                loss = torch.tensor(0., device=x.device)

            return loss


        # process labels according to the mask during training
        if labels is not None:
            # llm loss
            pad_len = max(llm_input_mask.size(1) - input_ids_labels.size(1), 0)
            pad_labels = torch.full((bs, pad_len), -100, dtype=torch.long, device=input_ids_labels.device)
            labels_llm = input_ids_labels * mask_label - 100 * (1 - mask_label)
            labels_llm = torch.cat([pad_labels, labels_llm], dim=1)
    
            assert llm_input_mask.shape == labels_llm.shape, "labels and attention_mask should have the same dimension."
            
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=False)
            
            # forward llm
            llm_outputs = self.model_llm(
                inputs_embeds=llm_input_embedding,
                attention_mask=llm_input_mask,
                labels=labels_llm,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            if not self.config.freeze_enc or not self.config.freeze_mapping_enc2llm or not self.config.freeze_llm:
                loss = llm_outputs.loss

            if (
                not self.config.llm_only
                and (not self.config.freeze_mapping_llm2dec or not self.config.freeze_dec)
            ):
                # decoder loss
                cross_attention_mask = (labels_llm != -100).long()
                # shift cross_attention
                cross_attention_mask = cross_attention_mask[:, 1:]
                llm_hidden_states = llm_outputs.hidden_states[-2][:, : -1]

                keep_mask = cross_attention_mask.any(dim=0)
                first_nonzero_idx = torch.argmax(keep_mask.int()).item()
                cross_attention_mask = cross_attention_mask[:, first_nonzero_idx:]
                llm_hidden_states = llm_hidden_states[:, first_nonzero_idx:]

                cross_attention_kv = self.mapping_llm2dec(llm_hidden_states.to(self.mapping_llm2dec.layers[0].linear1.weight.device))

                encoder_outputs = BaseModelOutput(
                    last_hidden_state=cross_attention_kv
                )
                
                decoder_input_ids = left_to_right_padding(decoder_input_ids, 0, self.mt_pad_token_id)
                decoder_labels = left_to_right_padding(decoder_labels, -100, -100)

                decoder_outputs = self.model_mt(
                    input_ids=None,
                    attention_mask=cross_attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    labels=decoder_labels,
                )
                
                loss = loss + self.dec_lambda * decoder_outputs.loss.to(loss.device)

                # compute ot loss: encoder(labels) ~ hidden_states of labels (cross_attention_kv, cross_attention_mask)
                mt_labels_mask = (mt_labels != 0)
                mt_label_encoded, mt_label_attention_mask, input_ids_mt_label = embedding_tokens(mt_labels, mt_labels_mask, self.mt_pad_token_id, "right", encoder=self.model_mt.get_encoder(), mapping=False)

                if abs(self.ot_lambda) > 1e-8:
                    ot_loss2 = compute_op_distance_min(mt_label_encoded, mt_label_attention_mask.to(mt_label_encoded.device), cross_attention_kv.to(mt_label_encoded.device), cross_attention_mask.to(mt_label_encoded.device))
                    loss = loss + self.ot_lambda * ot_loss2.to(loss.device)

            return (loss, )
        else:
            need_decoder = (
                not self.config.llm_only
                and forced_decoder_start_token_id is not None
            )
            generate_ids = self.model_llm.generate(
                inputs_embeds=llm_input_embedding.to(self.model_llm.dtype),
                attention_mask=llm_input_mask,
                max_new_tokens=self.config.max_gen_len,
                pad_token_id=self.llm_pad_token_id,
                eos_token_id=self.llm_eos_token_id if self.llm_eos_token_id is not None else [2, 128001, 128009],
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=need_decoder,
            )

            if not need_decoder:
                return (generate_ids.sequences,)

            def generate_cross_attention_mask(sequences, eos_token_id):
                bsz, max_len = sequences.shape
                eos_mask = (sequences == eos_token_id)

                first_eos_pos = torch.where(
                    eos_mask.any(dim=1),
                    eos_mask.int().argmax(dim=1),
                    torch.full((bsz,), max_len - 1, device=sequences.device)
                )
                valid_lens = first_eos_pos + 1
                range_row = torch.arange(max_len, device=sequences.device).unsqueeze(0)
                cross_attention_mask = (range_row < valid_lens.unsqueeze(1)).long()
                
                return cross_attention_mask
            
            def rightpad_to_leftpad(hidden_states, attention_mask):
                bsz, length, dim = hidden_states.size()
                
                valid_counts = attention_mask.sum(dim=1)  # (bsz,)

                new_mask = torch.zeros_like(attention_mask)
                for i in range(bsz):
                    cnt = valid_counts[i]
                    if cnt > 0:
                        new_mask[i, length - cnt:] = 1

                new_hidden = torch.zeros_like(hidden_states)

                for i in range(bsz):
                    cnt = valid_counts[i]
                    if cnt == 0:
                        continue
                    valid_h = hidden_states[i][attention_mask[i] == 1]
                    new_hidden[i, length - cnt:] = valid_h

                return new_hidden, new_mask
            
            
            # prepare for decoder inference
            prompt_hiddens = generate_ids.hidden_states[0][-2]
            gen_seq_states = [hidden_state[-2] for hidden_state in generate_ids.hidden_states[1:]]
            gen_seq_states = torch.cat(gen_seq_states, dim=1)
            llm_hidden_states = torch.cat([prompt_hiddens[:, -1:, :], gen_seq_states], dim=1)
            
            cross_attention_mask = generate_cross_attention_mask(generate_ids.sequences, self.llm_eos_token_id)

            llm_hidden_states, cross_attention_mask = rightpad_to_leftpad(llm_hidden_states, cross_attention_mask.to(llm_hidden_states.device))
            
            cross_attention_kv = self.mapping_llm2dec(llm_hidden_states)
            encoder_outputs = BaseModelOutput(
                last_hidden_state=cross_attention_kv
            )
            
            decoder_input_ids = torch.full((bs, 1), self.mt_eos_token_id, dtype=torch.long).to(self.model_mt.device)
            
            decoder_generate_ids_list = []
            
            if forced_decoder_start_token_id is not None:
                if isinstance(forced_decoder_start_token_id, int):
                    forced_decoder_start_token_id = [forced_decoder_start_token_id]
                
                for decoder_start_token_id in forced_decoder_start_token_id:
                    decoder_generate_config = GenerationConfig(
                        decoder_start_token_id=self.mt_eos_token_id,
                        forced_bos_token_id=decoder_start_token_id,
                        pad_token_id=self.mt_pad_token_id,
                        eos_token_id=self.mt_eos_token_id,
                    )

                    decoder_generate_ids = self.model_mt.generate(
                        input_ids=decoder_input_ids,
                        generation_config=decoder_generate_config,
                        encoder_outputs=encoder_outputs,
                        attention_mask=cross_attention_mask,
                        max_new_tokens=self.config.max_gen_len
                    )

                    decoder_generate_ids_list.append(decoder_generate_ids)

            if len(decoder_generate_ids_list) == 1:
                return (generate_ids.sequences, decoder_generate_ids_list[0])
            else:
                return (generate_ids.sequences, decoder_generate_ids_list)


