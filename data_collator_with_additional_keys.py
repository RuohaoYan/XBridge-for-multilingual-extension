from transformers import DataCollatorForSeq2Seq
import numpy as np

class DataCollatorWithAdditionalKeys(DataCollatorForSeq2Seq):
    
    additional_keys = []
    official_keys = ["input_ids", "attention_mask", "labels"]
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # pad additional keys
        for key in features[0].keys():
            if key not in self.official_keys and key not in self.additional_keys:
                self.additional_keys.append(key)

        
        for key in self.additional_keys:
            key_lengths = [len(feature[key]) for feature in features if isinstance(feature[key], (list, np.ndarray))]
            if not key_lengths:
                continue
            max_key_len = max(key_lengths)
            if self.pad_to_multiple_of is not None:
                max_key_len = (
                    (max_key_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
                )

            for feature in features:
                value = feature[key]
                pad_val = -100 if key == "decoder_labels" else 0
                remainder = [pad_val] * (max_key_len - len(value))

                if isinstance(value, list):
                    feature[key] = value + remainder if padding_side == "right" else remainder + value
                else:
                    feature[key] = (
                        np.concatenate([value, remainder]) if padding_side == "right"
                        else np.concatenate([remainder, value])
                    ).astype(np.int64)

        # convert to tensors
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features