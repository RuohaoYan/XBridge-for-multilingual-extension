"""Chat-template helpers for encoder-only XBridge inference."""

from __future__ import annotations

from typing import List, Sequence, Tuple

ENC_PLACEHOLDER = "<<<XBRIDGE_ENC>>>"
TRANSLATE_INSTRUCTION = "Translate the following text to English."
MGSM_INSTRUCTION = "Solve the following math problem step by step."


def supports_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


def build_chat_inject_parts(tokenizer, user_instruction: str) -> Tuple[List[int], List[int]]:
    """Split chat template into prefix + suffix around an encoder content slot.

    Resulting layout after model assembly (Plan C):
      prefix(aug=2) + boundary + Enc(x)(aug=1) + suffix(aug=2)
    i.e. <|user|> instruction ... [Enc(x)] <|eot|><|assistant|>
    """
    if not supports_chat_template(tokenizer):
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} has no chat_template; "
            "use an Instruct model or disable --use_chat_template."
        )
    user_content = f"{user_instruction}\n{ENC_PLACEHOLDER}"
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    ph_ids = tokenizer.encode(ENC_PLACEHOLDER, add_special_tokens=False)
    ph_start = _find_subsequence(full_ids, ph_ids)
    if ph_start < 0:
        raise ValueError(
            "Failed to locate encoder placeholder in chat template; "
            f"instruction={user_instruction!r}"
        )
    prefix_ids = full_ids[:ph_start]
    suffix_ids = full_ids[ph_start + len(ph_ids) :]
    return prefix_ids, suffix_ids


def build_translate_inject_parts(tokenizer) -> Tuple[List[int], List[int]]:
    return build_chat_inject_parts(tokenizer, TRANSLATE_INSTRUCTION)


def build_mgsm_inject_parts(tokenizer) -> Tuple[List[int], List[int]]:
    return build_chat_inject_parts(tokenizer, MGSM_INSTRUCTION)


def pad_encoder_chat_inject_batch(
    ids_mt_list: Sequence[Sequence[int]],
    prefix_list: Sequence[Sequence[int]],
    suffix_list: Sequence[Sequence[int]],
    pad_token_id: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Left-pad batched prefix(2) + mt(1) + suffix(2) sequences."""
    seqs = [
        list(prefix) + list(mt) + list(suffix)
        for prefix, mt, suffix in zip(prefix_list, ids_mt_list, suffix_list)
    ]
    max_len = max(len(seq) for seq in seqs)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    augmentation: List[List[int]] = []
    for prefix, mt_ids, suffix, seq in zip(prefix_list, ids_mt_list, suffix_list, seqs):
        pad_len = max_len - len(seq)
        input_ids.append([pad_token_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
        augmentation.append(
            [0] * pad_len
            + [2] * len(prefix)
            + [1] * len(mt_ids)
            + [2] * len(suffix)
        )
    return input_ids, attention_mask, augmentation

# Legacy Plan A helpers (Enc prefix + full chat user content with duplicated text)
TRANSLATE_USER_TEMPLATE = "Translate the following text to English:\n{text}"
MGSM_USER_TEMPLATE = "Solve the following math problem step by step.\n\n{question}"


def build_chat_prompt_ids(tokenizer, user_content: str) -> List[int]:
    if not supports_chat_template(tokenizer):
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} has no chat_template; "
            "use an Instruct model or disable --use_chat_template."
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )


def translate_user_content(src: str) -> str:
    return TRANSLATE_USER_TEMPLATE.format(text=src)


def mgsm_user_content(question: str) -> str:
    return MGSM_USER_TEMPLATE.format(question=question)


def pad_encoder_prompt_batch(
    ids_mt_list: Sequence[Sequence[int]],
    ids_prompt_list: Sequence[Sequence[int]],
    pad_token_id: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Left-pad batched Enc(x) + chat-prompt sequences with aug 1/2 (Plan A)."""
    seqs = [list(mt) + list(prompt) for mt, prompt in zip(ids_mt_list, ids_prompt_list)]
    max_len = max(len(seq) for seq in seqs)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    augmentation: List[List[int]] = []
    for mt_ids, prompt_ids, seq in zip(ids_mt_list, ids_prompt_list, seqs):
        pad_len = max_len - len(seq)
        input_ids.append([pad_token_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
        augmentation.append(
            [0] * pad_len + [1] * len(mt_ids) + [2] * len(prompt_ids)
        )
    return input_ids, attention_mask, augmentation
