#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import argparse
import logging
import json

from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', dest='gold_folder', help='Gold Folder')
    parser.add_argument('-p', dest='pred_folder', nargs='+', help='Pred Folder')

    parser.add_argument('-c', '--config', dest='map_config', help='Offset Mapping Config')
    parser.add_argument('-d', dest='decoding', default='spotasoc')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', help='More details information.')
    options = parser.parse_args()

    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict(options.gold_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
    )

    data_dict = {
        'eval': ['eval_preds_seq2seq.txt', 'val.json', 'eval_preds_record.txt'],
        'test': ['test_preds_seq2seq.txt', 'test.json', 'test_preds_record.txt'],
    }

    for pred_folder in options.pred_folder:
        gold_folder = options.gold_folder

        for data_key, (generation, gold_file, record_file) in data_dict.items():

            pred_filename = os.path.join(pred_folder, generation)

            if not os.path.exists(pred_filename):
                logger.warning("%s not found.\n" % pred_filename)
                continue

            gold_filename = os.path.join(gold_folder, gold_file)

            print("pred:", pred_filename) if options.verbose else None
            print("gold:", gold_filename) if options.verbose else None

            # Only using text and tokens in Gold file
            gold_list = [json.loads(line) for line in open(gold_filename)]
            gold_text_list = [gold['text'] for gold in gold_list]
            # gold_text_list = []
            # for gold in gold_list:
            #     gold_text = gold['text']
            #     if '<query_start>' in gold_text:
            #         gold_text = gold_text.split('<query_start> ',1)[1]
            #         gold_text = gold_text.split(' <query_end>',1)[0]
            #     gold_text_list.append(gold_text)

            gold_token_list = [gold['tokens'] for gold in gold_list]

            pred_list = [line.strip() for line in open(pred_filename).readlines()]

            pred_records = list()
            for pred, text, tokens in zip(pred_list, gold_text_list, gold_token_list):
                pred_record = sel2record.sel2record(pred, text, tokens)
                pred_records += [pred_record]

            with open(os.path.join(pred_folder, record_file), 'w') as output:
                for record in pred_records:
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == "__main__":

    sys.argv.extend(['-g', 'data/converted_data/text2spotasoc/event/casie'])  # gold folder
    sys.argv.extend(['-p', 'hf_models/uie_base_pipeline_casie/num_beam_5_top_p_1']) # pred folder
    sys.argv.extend(['-c', 'config/offset_map/first_offset_en.yaml']) # offset mapping strategy config
    sys.argv.extend(['-v'])

    main()
