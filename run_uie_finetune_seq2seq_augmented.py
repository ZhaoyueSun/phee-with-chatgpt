#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/prj/eventnlu/zhaoyue/.cache/huggingface/'


import pdb
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import datasets
from datasets import load_dataset, concatenate_datasets
datasets.disable_caching()
datasets.disable_progress_bar()

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process,PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from uie.extraction import constants
from uie.extraction.record_schema import RecordSchema
from uie.extraction.extraction_metrics import get_extract_metrics
from uie.extraction.noiser.ordered_spot_asoc_noiser import OrderedSpotAsocNoiser
from uie.seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainingArguments, ConstraintSeq2SeqTrainer
from uie.seq2seq.features import BasicFeature
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2seq.trainer_arguments import ModelArguments, DataTrainingArguments
from uie.extraction.constants import BaseStructureMarker 
from uie.extraction.utils import convert_to_record_function

logger = logging.getLogger(__name__)

@dataclass
class DataArguments(DataTrainingArguments):
    input_format: str = field(
        default="raw", metadata={"help": "prompt format, can be raw/prompt/instruction"}
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to lora finetuning."
        },
    )

    max_augment_samples: int = field(
        default=None,
        metadata={
            "help": "Max number of augmented data to use."
        },
    )

    augment_file: str = field(
        default=None,
        metadata={
            "help": "path to augment data file."
        },
    )


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
    


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, ConstraintSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # pdb.set_trace()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # logger.setLevel(logging.ERROR)

    logger.info("Options:")
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        # transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `record_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            # extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            # extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            # extension = data_args.test_file.split(".")[-1]
        if data_args.augment_file is not None:
            data_files["augment"] = data_args.augment_file
            # extension = data_args.augment_file.split(".")[-1]
    logger.info(data_files)
    datasets = load_dataset("uie_json.py", data_files=data_files, download_mode="force_redownload")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    logger.info(datasets)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    logger.info("Load Config: %s" % model_args.config_name if model_args.config_name else model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.max_length = data_args.max_target_length

    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if 'char' in tokenizer_name:
        tokenizer = T5BertTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    if not data_args.use_lora:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            mirror='tuna'
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            mirror='tuna',
            load_in_8bit=True,
            device_map='auto'
        )
                    
        # Define LoRA Config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info(tokenizer)

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if data_args.record_schema and os.path.exists(data_args.record_schema):
        record_schema = RecordSchema.read_from_file(data_args.record_schema)
    else:
        record_schema = None

    instruction = "Extract the event information in the text."
    prompt = "The events included in the text are:"

    spot_asoc_nosier = OrderedSpotAsocNoiser(
        schema=record_schema,
        spot_noise_ratio=data_args.spot_noise,  # noise add to record(target) during training
        asoc_noise_ratio=data_args.asoc_noise,  # add null span 
        null_span=constants.null_span,
        ordered_prompt=data_args.ordered_prompt
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).

    text_column = data_args.text_column
    record_column = data_args.record_column
    logger.info('Using src: %s and tgt: %s' % (text_column, record_column))

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def regenerate_spot_asoc(record):
        # Inject rejection noise
        spot_label_list = record_schema.type_list.copy()
        asoc_label_list = record_schema.role_list.copy()
        for spot in record:
            if spot['label'] in spot_label_list:
                spot_label_list.remove(spot['label'])

            for asoc in spot['asoc']:
                if asoc[0] in asoc_label_list:
                    asoc_label_list.remove(asoc[0])

        if spot_asoc_nosier is not None:
            record = spot_asoc_nosier.add_noise(record, spot_label_list=spot_label_list, asoc_label_list=asoc_label_list)
        # Generate new record
        record = convert_to_record_function[data_args.decoding_format](
            record,
            structure_maker=BaseStructureMarker()
        )

        return record

    def preprocess_function(example):
        inputs = example[text_column]
        inputs = preprocess_text(inputs)

        if data_args.input_format == 'prompt':
            inputs = inputs + " " + prompt
        elif data_args.input_format == 'instruction':
            evt_types = record_schema.type_list
            arg_types = record_schema.role_list
            schema_prompt = "Event type: "+", ".join(evt_types)+" Argument type: "+", ".join(arg_types)
            inputs = instruction + " " + schema_prompt + " Sentence: " + inputs + " Output:"

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # regenerate label (reject mechanism)
        if 'spot_asoc' in example:
            record = regenerate_spot_asoc(example['spot_asoc'])                
        else:
            record = example[record_column]
        record = preprocess_text(record)
        # # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(record, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) for _label in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_text(x_str):
        x_str = x_str.replace('[', '&(').replace(']', ')&')
        x_str = x_str.replace('<extra_id_0>', '[').replace('<extra_id_1>', ']').replace('<extra_id_5>', ':').replace('<extra_id_6>', 'null').replace('<extra_id_7>', 'null')
        
        return x_str
    
    def postprocess_text(x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        
        # replace span split symbol and null symbol to be consistent with prompting inference
        x_str = x_str.replace('null', '<extra_id_6>').replace(':', '<extra_id_5>').replace(']', '<extra_id_1>').replace('[','<extra_id_0>')
        x_str = x_str.replace('&(', '[').replace(')&', ']')
        
        return x_str.strip()


    logger.info("Start Data Preprocessing ...")

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        
        if "augment" in datasets:
            augment_dataset = datasets["augment"]
            if data_args.max_augment_samples is not None:
                augment_dataset = augment_dataset.select(range(data_args.max_augment_samples))

            train_dataset = concatenate_datasets([train_dataset, augment_dataset])
            

        train_dataset = train_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=BasicFeature,
        )


    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=BasicFeature,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=BasicFeature,
        )

    logger.info("End Data Preprocessing ...")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [postprocess_text(x) for x in decoded_preds]
        decoded_labels = [postprocess_text(x) for x in decoded_labels]
        # Overall-F1 = spot-F1 + asoc-F1
        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=record_schema,
            decoding_format=data_args.decoding_format,
        )

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = ConstraintSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        decoding_type_schema=record_schema,
        decoding_format=data_args.decoding_format,
        source_prefix="",
        task=data_args.task,
    )
    trainer.add_callback(EarlyStoppingCallback(5))
    # trainer.add_callback(EarlyStoppingCallback(2))
    if data_args.use_lora:
        trainer.add_callback(SavePeftModelCallback())

    # Training
    if training_args.do_train:
        if model_args.from_checkpoint:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path
        checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if data_args.use_lora:
            model.save_pretrained(training_args.output_dir) 
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
        
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate(max_length=data_args.val_max_target_length, num_beams=data_args.num_beams)
        results = {k: round(v, 4) for k, v in results.items()}

        eval_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                eval_preds = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                eval_preds = [postprocess_text(pred) for pred in eval_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "eval_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(eval_preds))

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)

        output_test_result_file = os.path.join(training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                test_preds = [postprocess_text(pred) for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":

    sys.argv.extend(['train_script_config/flan_t5_large_finetune_phee_augment.json'])
    main()

