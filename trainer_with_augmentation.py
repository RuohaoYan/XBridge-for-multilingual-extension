import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from packaging import version

from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.utils import (
    is_datasets_available,
    is_peft_available,
    is_torch_tpu_available,
    logging
)

if is_datasets_available():
    import datasets

if is_peft_available():
    from peft import PeftModel


logger = logging.get_logger(__name__)

class TrainerWithAugmentation(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _remove_unused_columns(
        self, 
        dataset: "datasets.Dataset", 
        description: Optional[str] = None
    ):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if "augmentation" in ignored_columns:
            ignored_columns.remove("augmentation")
            columns.append("augmentation")
        
        if "encoder_input_ids" in ignored_columns:
            ignored_columns.remove("encoder_input_ids")
            columns.append("encoder_input_ids")
            
        if "llm_labels" in ignored_columns:
            ignored_columns.remove("llm_labels")
            columns.append("llm_labels")
            
        if "decoder_input_ids" in ignored_columns:
            ignored_columns.remove("decoder_input_ids")
            columns.append("decoder_input_ids")
            
        if "decoder_labels" in ignored_columns:
            ignored_columns.remove("decoder_labels")
            columns.append("decoder_labels")
        
        if "mt_labels" in ignored_columns:
            ignored_columns.remove("mt_labels")
            columns.append("mt_labels")
        
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
    

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False

