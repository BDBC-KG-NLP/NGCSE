import os
import sys
import pdb
import json

from typing import cast

from datasets import load_dataset

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.utils import logging
from transformers.trainer_utils import is_main_process

from ngcse.models import NGBert, NGBertConfig, NGRoberta, NGRobertaConfig
from ngcse.trainers import NGTrainer, GenerateEmbeddingCallback
from ngcse.utils import (
    ModelArguments, DataTrainingArguments, OurTrainingArguments,
    PrepareFeatures, OurDataCollatorWithPadding
)

# logger = logging.getLogger(__name__) # logging
logger = logging.get_logger() # transformers.utils.logging
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataTrainingArguments, data_args)
    training_args = cast(OurTrainingArguments, training_args)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    os.makedirs(training_args.output_dir, exist_ok=training_args.overwrite_output_dir)

    # Save configuration
    all_args = dict()
    for tmp_args in [model_args, data_args, training_args]:
        all_args.update({k: v for k, v in tmp_args.__dict__.items() if \
            type(v) in {int, float, str, list, dict, tuple}})
    with open(os.path.join(training_args.output_dir, 'all_args.config'), 'w', encoding='utf8') as fo:
        json.dump(all_args, fo, indent=4)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load dataset
    assert data_args.train_file and data_args.train_file.split(".")[-1] in ['json', 'jsonl']
    datasets = load_dataset("json", 
        data_files={"train": data_args.train_file}, 
        cache_dir="./data/"
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.use_native_loss:
        config_kwargs['use_native_loss'] = model_args.use_native_loss
    if model_args.use_loss3:
        config_kwargs['use_loss3'] = model_args.use_loss3
    if model_args.use_loss4:
        config_kwargs['use_loss4'] = model_args.use_loss4
    if model_args.temp:
        config_kwargs['temp'] = model_args.temp
    if model_args.margins:
        config_kwargs['margins'] = eval(model_args.margins)
    if model_args.beta:
        config_kwargs['beta'] = model_args.beta
    if model_args.gamma:
        config_kwargs['gamma'] = model_args.gamma
    if model_args.pooler_type:
        config_kwargs['pooler_type'] = model_args.pooler_type
    if model_args.feature_size:
        config_kwargs['feature_size'] = model_args.feature_size
    if model_args.num_layers:
        config_kwargs['num_layers'] = model_args.num_layers
    if model_args.optional:
        config_kwargs['optional'] = model_args.optional

    assert model_args.model_type in MODEL_TYPES, f'Determine model_type within {MODEL_TYPES}'
    config_class = NGBertConfig if model_args.model_type == 'bert' else NGRobertaConfig
    if model_args.config_name:
        config = config_class.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = config_class()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    assert model_args.model_name_or_path, "Requires model_name_or_path."
    model_type = NGBert if model_args.model_type == 'bert' else NGRoberta
    model, loading_info = model_type.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        output_loading_info=True
    )
    if 'mlp.bias' in loading_info['missing_keys']:
        model.custom_param_init()

    model.resize_token_embeddings(len(tokenizer))
    
    column_names = datasets["train"].column_names
    # if training_args.do_train:
    prepare_features = PrepareFeatures(column_names, tokenizer, data_args, training_args)
    train_dataset = datasets["train"]
    if data_args.only_aigen:
        train_dataset = train_dataset.filter(lambda raw: raw['split'] == 'aigen')
    train_dataset = train_dataset.map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[column_name for column_name in column_names if column_name != 'split'],
        load_from_cache_file=not data_args.overwrite_cache,
    )
    
    data_collator = default_data_collator if data_args.pad_to_max_length \
        else OurDataCollatorWithPadding(tokenizer,
                                        aigen_sent_num=len(prepare_features.aigen_keys), 
                                        other_sent_num=max(2, len(prepare_features.other_keys)),
                                        aigen_batch_size=data_args.aigen_batch_size, 
                                        combine_training=data_args.combine_training)

    trainer = NGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[GenerateEmbeddingCallback()]
    )

    # Training
    if training_args.do_train:
        # resume_from_checkpoint = (
        #     model_args.model_name_or_path
        #     if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
        #     else training_args.resume_from_checkpoint
        # )
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        model.display_loss(125) # display loss log.

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
