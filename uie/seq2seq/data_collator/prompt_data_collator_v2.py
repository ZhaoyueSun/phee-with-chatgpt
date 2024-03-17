#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from dataclasses import dataclass
import torch
import logging
import random
import math
from typing import Optional, Union
from collections import OrderedDict, defaultdict
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy

from uie.extraction.record_schema import RecordSchema
from uie.extraction.constants import BaseStructureMarker 
from uie.extraction.utils import convert_to_record_function
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser


logger = logging.getLogger("__main__")

class PromptSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, schema: RecordSchema, positive_rate=1, negative=5, ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample, Negative: {self.negative}, Ordered Prompt: {self.ordered_prompt}")

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in torch.randperm(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot
        """
        negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=self.negative)
        positive_spot = random.sample(positive, math.floor(len(positive) * self.positive_rate))

        return positive_spot, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc
        """
        negative_asoc = self.sample_negative(postive=positive, candidates=self.asoc_list, k=self.negative)

        return negative_asoc

    
@dataclass
class PromptDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of target sequence length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    negative_sampler: PromptSSIGenerator
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None
    decoding_format: str = 'spotasoc'
    prompt_strategy: str = 'random'
    
    demo_num_counter = defaultdict(int)


    def regenerate_spot_asoc(self, spots, asocs, spot_asocs):
        positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=spots)
        negative_asoc = self.negative_sampler.sample_asoc(positive=asocs)

        # Deleted positive example Spot in Target that was not sampled by Prefix
        record = [spot_asoc for spot_asoc in spot_asocs if spot_asoc["label"] in positive_spot]

        # Inject rejection noise
        if self.spot_asoc_nosier is not None:
            if isinstance(self.spot_asoc_nosier, SpotAsocNoiser):
                record = self.spot_asoc_nosier.add_noise(
                    record,
                    spot_label_list=negative_spot,
                    asoc_label_list=negative_asoc,
                )
            else:
                raise NotImplementedError(f'{self.spot_asoc_nosier} is not implemented.')

        # Generate new record
        record = convert_to_record_function[self.decoding_format](
            record,
            structure_maker=BaseStructureMarker()
        )

        return record


    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """
        prefix = 'text: '
        suffix = ' output:'

        for feature in features:

            # Dynamic generating spot-asoc during training
            if 'spot_asoc' in feature:
                spots = feature.get('spots', [])
                asocs = feature.get('asocs', [])
                record = self.regenerate_spot_asoc(spots, asocs, feature['spot_asoc'])                
                # record.replace('<extra_id_0>', '[').replace('<extra_id_1>', ']')
                feature["labels"] = self.tokenizer.encode(record)

            feature['input_ids'] = feature['input_ids']
            input_len = len(feature['input_ids'])
            # truncate `input_ids` to max length
            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]

            add_demos = 0

            if 'demos' in feature:
                for demo_instance in feature['demos']:
                    spots = demo_instance.get('spot', [])
                    asocs = demo_instance.get('asoc', [])
                    record = self.regenerate_spot_asoc(spots, asocs, demo_instance['spot_asoc'])                
                    record.replace('<extra_id_0>', '[').replace('<extra_id_1>', ']')
                    demo_context = prefix + demo_instance['text'] + suffix + record
                    demo_tokens = self.tokenizer.encode(demo_context)[:-1] # drop the last </s>
                    if len(demo_tokens) + input_len <= self.max_target_length:
                        input_len += len(demo_tokens)
                        feature['input_ids'] = demo_tokens + feature['input_ids']
                        add_demos += 1
                    
            self.demo_num_counter[add_demos] += 1

            feature['attention_mask'] = [1] * len(feature['input_ids'])
            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature.pop('spots') if 'spots' in feature else None
            feature.pop('asocs') if 'asocs' in feature else None
            feature.pop('demos') if 'demos' in feature else None
            
            
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
