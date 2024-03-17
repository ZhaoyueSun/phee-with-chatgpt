#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

from collections import Counter
import argparse
import logging
import json
import re
from nltk.tree import ParentedTree

from uie.sel2record.sel2record import SEL2Record, proprocessing_graph_record
from uie.extraction.predict_parser.spotasoc_predict_parser import add_bracket, convert_bracket, clean_text, check_well_form
from uie.extraction.predict_parser.spotasoc_predict_parser import get_tree_str, resplit_label_span
from uie.extraction.constants import (
    null_span,
    type_start,
    type_end,
    span_start,
)

logger = logging.getLogger(__name__)


def get_record_list(sel_tree, schema_dict):
    """ Convert single sel expression to extraction records
    Args:
        sel_tree (Tree): sel tree
        text (str, optional): _description_. Defaults to None.
    Returns:
        spot_list: list of (spot_type: str, spot_span: str)
        asoc_list: list of (spot_type: str, asoc_label: str, asoc_text: str)
        record_list: list of {'asocs': list(), 'type': spot_type, 'spot': spot_text}
    """

    spot_list = list()
    asoc_list = list()
    record_list = list()

    for spot_tree in sel_tree:
        # Drop incomplete tree
        if isinstance(spot_tree, str) or len(spot_tree) == 0:
            continue

        spot_type = spot_tree.label()
        spot_text = get_tree_str(spot_tree)
        spot_type, spot_text = resplit_label_span(
            spot_type, spot_text)

        # Drop empty generated span
        if spot_text is None or spot_text == null_span:
            continue
        # Drop empty generated type
        if spot_type is None:
            continue
        # Drop invalid spot type
        if schema_dict['record'].type_list and spot_type not in schema_dict['record'].type_list:
            continue

        record = {'asocs': list(),
                    'type': spot_type,
                    'spot': spot_text}

        for asoc_tree in spot_tree:
            if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                continue

            asoc_label = asoc_tree.label()
            asoc_text = get_tree_str(asoc_tree)
            asoc_label, asoc_text = resplit_label_span(
                asoc_label, asoc_text)

            # Drop empty generated span
            if asoc_text is None or asoc_text == null_span:
                continue
            # Drop empty generated type
            if asoc_label is None:
                continue
            # Drop invalid spot type
            if schema_dict['record'].role_list and asoc_label not in schema_dict['record'].role_list:
                continue

            asoc_list += [(spot_type, asoc_label, asoc_text)]
            record['asocs'] += [(asoc_label, asoc_text)]

        spot_list += [(spot_type, spot_text)]
        record_list += [record]

    return spot_list, asoc_list, record_list

def decode(pred, schema_dict):
    left_bracket = '【'
    right_bracket = '】'
    brackets = left_bracket + right_bracket

    pred = convert_bracket(pred)
    pred = clean_text(pred)
    try:
        if not check_well_form(pred):
            pred = add_bracket(pred)
            counter.update(['fixed_pred'])

        pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
        counter.update(['pred_tree' for _ in pred_tree])
        counter.update(['well-formed'])

    except ValueError:
        counter.update(['ill-formed'])
        logger.debug('ill-formed', pred)
        pred_tree = ParentedTree.fromstring(
            left_bracket + right_bracket,
            brackets=brackets
        )

    instance = {}
    instance['pred_spot'], instance['pred_asoc'], instance['pred_record'] = get_record_list(
                sel_tree=pred_tree,
                schema_dict=schema_dict
            )

    return instance


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', dest='gold_folder', help='Gold Folder')
    parser.add_argument('-p', dest='pred_folder', nargs='+', help='Pred Folder')

    parser.add_argument('-c', '--config', dest='map_config', help='Offset Mapping Config')
    parser.add_argument('-d', dest='decoding', default='spotasoc')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', help='More details information.')
    options = parser.parse_args()

    schema_dict = SEL2Record.load_schema_dict(options.gold_folder)

    global counter
    counter = Counter()


    data_dict = {
        'eval': ['eval_preds_seq2seq.txt', 'val.json', 'eval_preds_record.txt'],
        'test': ['test_preds_seq2seq.txt', 'test.json', 'test_preds_record.txt'],
    }

    for pred_folder in options.pred_folder:

        for data_key, (generation, gold_file, record_file) in data_dict.items():

            pred_filename = os.path.join(pred_folder, generation)

            if not os.path.exists(pred_filename):
                logger.warning("%s not found.\n" % pred_filename)
                continue

            pred_list = [line.strip() for line in open(pred_filename).readlines()]

            pred_records = list()
            for pred in pred_list:
                pred_instance = decode(pred, schema_dict)
                pred_record = proprocessing_graph_record(
                    pred_instance,
                    schema_dict
                )
                out_record = {}
                for key, value in pred_record.items():
                    out_record[key] = {"offset": [], "string": []}
                    for data in value:
                        out_record[key]['string'].append(data)
                pred_records.append(out_record)

            with open(os.path.join(pred_folder, record_file), 'w') as output:
                for record in pred_records:
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(counter)

if __name__ == "__main__":

    sys.argv.extend(['-g', 'data/converted_data/text2spotasoc/event/phee_2/'])  # gold folder
    sys.argv.extend(['-p', 'hf_models/flan_t5_large_instruction_finetune_phee2aug_asoc_noise_1_order_pos_stage2_r3']) # pred folder
    sys.argv.extend(['-c', 'config/offset_map/first_offset_en.yaml']) # offset mapping strategy config
    sys.argv.extend(['-v'])

    main()
