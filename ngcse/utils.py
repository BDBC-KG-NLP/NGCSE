import pdb
import torch

from transformers import MODEL_FOR_MASKED_LM_MAPPING, TrainingArguments
from transformers.utils import logging
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

from typing import Union, List, Optional, Dict
from dataclasses import dataclass, field

# logger = logging.getLogger(__name__)
logger = logging.get_logger()

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# tools args



# train args
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_type: Optional[str] = field(
        # default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES) +\
        ". Also used to determine backend model."
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # NGCSE's arguments
    use_native_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to calculate contrastive loss for aigens and others together."
        }
    )
    use_loss3: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Hierarchical Triplet loss."
        }
    )
    use_loss4: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Recall Knowledge loss."
        }
    )
    temp: Optional[float] = field(
        default=None,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    margins: Optional[str] = field(
        default=None,
        metadata={
            "help": "Margins for Hierarchical Triplet loss."
        }
    )
    beta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Weight of Hierarchical Triplet loss."
        }
    )
    gamma: Optional[float] = field(
        default=None,
        metadata={
            "help": "Weight of Recall Knowledge loss."
        }
    )
    pooler_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "What kind of pooler to use.",
            "choices": ["cls", "avg", "mask"]
        }
    )
    feature_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Embedding size for feature space."
        }
    )
    num_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of training layers in Bert."
        }
    )
    optional: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional settings."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # NGCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.jsonl)."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The validation data file (.jsonl)."}
    )
    only_aigen: bool = field(
        default=False,
        metadata={
            "help": "Whether to train with `aigen` data only. "
        },
    )
    aigen_batch_size: int = field(
        default=48,
        metadata={
            "help": "The minimun number of aigen data in a batch size."
        },
    )
    combine_training: bool = field(
        default=True,
        metadata={
            "help": "Whether to combine aigen and other data in a batch."
        },
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json(l)."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Training

    dump_embeddings_num: int = field(
        default=1000,
        metadata={"help": "Number of Embeddings to be dumped after training"}
    )

    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    # overload
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

# Prepare features
@dataclass
class PrepareFeatures:

    column_names: List[str]
    tokenizer: PreTrainedTokenizerBase
    data_args: DataTrainingArguments
    training_args: OurTrainingArguments
    
    def __call__(self, examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.

        total = len(examples[self.column_names[0]])

        sentences = []
        if not hasattr(self, 'aigen_keys'): # check which key to use
            self.aigen_keys = []
        if not hasattr(self, 'other_keys'):
            self.other_keys = []

        for idx in range(total):
            if examples['split'][idx] == 'aigen':
                if len(self.aigen_keys) == 0: # check which key to use
                    for key in ['sentence', '5', '4', '3', '2', '1', '0']:
                        if examples[key][idx] != '':
                            self.aigen_keys.append(key)
                for key in self.aigen_keys:
                    sentences.append(examples[key][idx] if examples[key][idx] is not None else ' ')
        assert len(sentences) == 0 or len(sentences) % len(self.aigen_keys) == 0
        aigen_group_num = (0 if len(sentences) == 0 else len(sentences) // len(self.aigen_keys))
        
        for idx in range(total):
            if examples['split'][idx] == 'other':
                if len(self.other_keys) == 0:
                    for key in ['sentence', '5', '4', '3', '2', '1', '0']:
                        if examples[key][idx] != '':
                            self.other_keys.append(key)
                for key in self.other_keys:
                    sentences.append(examples[key][idx] if examples[key][idx] is not None else ' ')
        assert len(sentences) - aigen_group_num * len(self.aigen_keys) == 0 or \
            (len(sentences) - aigen_group_num * len(self.aigen_keys)) % len(self.other_keys) == 0
        other_group_num = 0 if len(sentences) - aigen_group_num * len(self.aigen_keys) == 0 else\
            (len(sentences) - aigen_group_num * len(self.aigen_keys)) // len(self.other_keys)

        sent_features = self.tokenizer(
            sentences,
            max_length=self.data_args.max_seq_length,
            truncation=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        features = {key: [] for key in sent_features.keys()}
        features['split'] = []
        for idx in range(aigen_group_num):
            for key in features.keys():
                if key == 'split':
                    features[key].append('aigen')
                else:
                    features[key].append(sent_features[key][idx * len(self.aigen_keys): (idx + 1) * len(self.aigen_keys)])
        for idx in range(other_group_num):
            for key in features.keys():
                if key == 'split':
                    features[key].append('other')
                else:
                    features[key].append(sent_features[key][
                        aigen_group_num * len(self.aigen_keys) + idx * len(self.other_keys): 
                        aigen_group_num * len(self.aigen_keys) + (idx + 1) * len(self.other_keys)
                    ])
                    if len(self.other_keys) == 1:
                        features[key][-1] = features[key][-1] * 2
        
        return features
   
# Data collator
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    aigen_sent_num: int = 0
    other_sent_num: int = 0
    aigen_batch_size: int = 64
    combine_training: bool = False
    aigen_features_cache = []
    other_features_cache = []

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        drop_keys = ['loss_pair', 'split']

        batch_size = len(features)
        for feature in features:
            if feature['split'] == 'aigen':
                self.aigen_features_cache.append(feature)
            else:
                self.other_features_cache.append(feature)
        '''new
        if len(self.aigen_features_cache) >= self.aigen_batch_size:
            features = self.aigen_features_cache[:self.aigen_batch_size]
            self.aigen_features_cache = self.aigen_features_cache[self.aigen_batch_size: ]
            if self.combine_training and self.aigen_batch_size < batch_size:
                other_batch_size = batch_size - self.aigen_batch_size
                features += self.other_features_cache[:other_batch_size]
                self.other_features_cache = self.other_features_cache[other_batch_size: ]
        else:
            if self.combine_training:
                features = self.other_features_cache[:batch_size]
                self.other_features_cache = self.other_features_cache[batch_size:]
            else:
                features = self.other_features_cache
                self.other_features_cache = []
        '''

        # old: when testing old version, change to this
        if len(self.aigen_features_cache) >= self.aigen_batch_size:
            features = self.aigen_features_cache
            self.aigen_features_cache = []
            if self.combine_training:
                features += self.other_features_cache
                self.other_features_cache = []
        else:
            features = self.other_features_cache
            self.other_features_cache = []

        flat_features = []
        aigen_num = 0
        other_num = 0
        for feature in features:
            if feature['split'] == 'aigen':
                aigen_num += 1
                for i in range(self.aigen_sent_num):
                    flat_features.append({k: (v[i] if k in special_keys else v) 
                                            for k, v in feature.items() if k not in drop_keys})
        for feature in features:                             
            if feature['split'] == 'other':
                other_num += 1
                for i in range(self.other_sent_num):
                    flat_features.append({k: (v[i] if k in special_keys else v) 
                                            for k, v in feature.items() if k not in drop_keys})
        
        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # input_ids, attention_mask, token_type_ids：[Tensor of shape(abs, 7, seq_len), Tensor of shape(obs, 2, seq_len)]
        batch = {k : [v[: aigen_num * self.aigen_sent_num], v[aigen_num * self.aigen_sent_num: ]] for k, v in batch.items()}
        for i, sen_num in zip(range(2), [self.aigen_sent_num, self.other_sent_num]):
            for k, v in batch.items():
                if k in special_keys:
                    v[i] = v[i].reshape((v[i].shape[0] // sen_num if sen_num else 0), sen_num, v[i].shape[1])
                else:
                    v[i] = v[i].reshape((v[i].shape[0] // sen_num if sen_num else 0), sen_num, v[i].shape[1])[:, 0]
        
        return batch
    
def print_example(example):
    for key in ['sentence'] + [str(i) for i in range(5, 0, -1)]:
        if example[key]:
            print(example[key] + '\n')