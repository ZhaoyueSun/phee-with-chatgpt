"""
Evaluate ChatGPT zero-shot result. 
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import argparse
import logging
import json
from easydict import EasyDict as edict
from collections import defaultdict, Counter
import yaml
from scripts.eval_phee.phee_metric import compute_metric


def read_gold_result(gold_file):
    PARENT_ARGS = ['Subject', "Effect", 'Treatment'] # TODO: discard this Disorder after cleaning the data
    PARENT_TO_CHILD = {
        "Subject": ["Race", "Age", "Gender", "Population", "Disorder"],
        "Treatment": ["Duration", "Time_elapsed","Route","Freq","Dosage", "Disorder", "Drug"],
        "Effect": [],
    }
    outputs = []
    with open(gold_file, 'r') as f:
        for line in f.readlines():

            data = json.loads(line)
            data = edict(data)

            annotation = data.annotations[0]

            instance = defaultdict(list)
            instance['context'] = data.context
            instance['id'].append(data.id)
            instance['is_mult'] = len(annotation.events) > 1

            for event in annotation.events:
                # Convert trigger
                ev_type = event.event_type
                trigger_text = event.Trigger.text[0][0]
                instance[ev_type+".Trigger"].append(trigger_text)

                # Convert arguments
                for role in PARENT_ARGS:
                    if role in event: # not appeared arguments are not stored
                        argument = event[role]
                        for entities in argument.text: # for each span in a multi-span argument
                            for t in entities: # for each discontinuous part of a argument span
                                instance[ev_type+"."+role].append(t)
                        # extract sub_arguments information
                        for key in argument.keys():
                            if key in PARENT_TO_CHILD[role]:
                                sub_arg = argument[key]
                                for entities in sub_arg.text: # for each span in a multi-span argument
                                    for t in entities: # for each discontinuous part of a argument span
                                        instance[ev_type+"."+role+"."+key].append(t)

                        # extraction combination.drug information
                        if role == 'Treatment' and 'Combination' in argument:
                            for comb in argument.Combination:
                                if "Drug" in comb:
                                    for entities in comb.Drug.text:
                                        for t in entities:
                                            instance[ev_type+".Combination.Drug"].append(t)

            outputs.append(instance)

    return outputs

def filter_span(span):
    span = span.lower()
    if "n/a" in span:
        return True
    if "null" in span or "none" in span:
        return True
    if "not mentioned" in span:
        return True
    return False

def read_pred_results(pred_folder, gold_instances):

    evt_type_map = {
        'adverse event': 'Adverse_event',
        'potential therapeutic event': 'Potential_therapeutic_event',
        'adverse_event': 'Adverse_event',
        'potential_therapeutic_event': 'Potential_therapeutic_event'
    }

    arg_type_map = {
        "subject": "Subject",
        "treatment": "Treatment",
        "effect": "Effect",
        "age": "Subject.Age",
        "gender": "Subject.Gender",
        "race": "Subject.Race",
        "population": "Subject.Population",
        "subject_disorder": "Subject.Disorder",
        "drug": "Treatment.Drug",
        "dosage": "Treatment.Dosage",
        "route": "Treatment.Route",
        "duration": "Treatment.Duration",
        "frequency": "Treatment.Freq",
        "time_elapsed": "Treatment.Time_elapsed",
        "indication": "Treatment.Disorder",
        "combination_drug": "Combination.Drug"
    }

    outputs = []
    wrong_event_type = 0
    wrong_arg_type = 0
    break_format = 0
    for gold in gold_instances:
        sample_id = gold['id'][0]
        pred_file = os.path.join(pred_folder, "%s.json"%sample_id)
        if not os.path.exists(pred_file):
            raise Exception("Predict file not found!")
        with open(pred_file, 'r') as f:
            try:
                pred_evts = json.load(f) 
            except:
                break_format += 1
                pred_evts = []

        # start parsing the result
        instance = defaultdict(list) 
        for evt in pred_evts:
            event_type = evt["event_type"]
            if event_type not in evt_type_map:
                wrong_event_type += 1
                event_type = 'Adverse_event'
            else:
                event_type = evt_type_map[event_type]

            arguments = evt["arguments"]
            for arg in arguments:
                arg_type = arg["argument_type"] if "argument_type" in arg else None
                if arg_type not in arg_type_map:
                    wrong_arg_type += 1
                    continue
                else:
                    arg_type = arg_type_map[arg_type]

                arg_span = arg["argument_span"] if "argument_span" in arg else None
                if arg_span is None: arg_span = 'N/A'
                arg_span = arg_span.strip()

                if filter_span(arg_span): 
                    continue
                
                if arg_type == 'Treatment.Drug' or arg_type == 'Combination':
                    arg_span = arg_span.replace(',', ';')
                    arg_span = arg_span.replace('and', ';')

                arg_type = event_type + "." + arg_type
                arg_spans = arg_span.split(';')
                instance[arg_type] += arg_spans

        outputs.append(instance)

    print("break format cases:", break_format)
    print("wrong event type cases:", wrong_event_type)
    print("matching wrong types to Adverse_event")
    print("wrong argument type cases:", wrong_arg_type)
    return outputs


def get_label_mapper(config_file):
    with open(config_file) as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    mapper = {}
    for key, value in config['mapper'].items():
        mapper[value] = key

    return mapper
    
    
def main():

    split = "test"
    gold_file = "data/datasets/phee2/source/%s.json"%split
    pred_folder = "chatgpt_zero_shot/output/sub_args/%s"%split
    out_file = "chatgpt_zero_shot/output/sub_args/%s_result.json"%split

    gold_instances = read_gold_result(gold_file)
    pred_instances = read_pred_results(pred_folder, gold_instances)

    assert(len(gold_instances) == len(pred_instances))

    instances = []
    # ev_tp = 0
    # ev_pred_n = 0
    # ev_gold_n = 0

    # counts = defaultdict(int)

    for preds, golds in zip(pred_instances, gold_instances):
        instance_id = golds['id'][0]
        # if eval_mult and not golds['is_mult']: continue
        # if eval_single and golds['is_mult']: continue
        question_types = list(set(list(preds.keys())+list(golds.keys())))
        question_types.remove('id')
        question_types.remove('is_mult')
        question_types.remove('context')
        for qtype in question_types:
            if qtype == 'id': continue
            instance = {
                'id': instance_id,
                'type': qtype,
                'predictions':[],
                'golds':[]
            }
            if qtype in preds:
                instance['predictions'] = preds[qtype]
            if qtype in golds:
                instance['golds'] = golds[qtype]
            # if eval_event_type:
            #     if eval_event_type in qtype:
            #         instances.append(instance)
            #         if qtype in golds:
            #             counts[qtype] +=1
            # else:
            instances.append(instance)

    #     # for event classification evaluation
    #     pred_evs = []
    #     gold_evs = []
    #     for k in preds:
    #         if 'Trigger' in k:
    #             ev_type = k.split('.')[0]
    #             pred_evs.append(ev_type)

    #     for k in golds:
    #         if 'Trigger' in k:
    #             ev_type = k.split('.')[0]
    #             for _ in golds[k]:
    #                 gold_evs.append(ev_type)

    #     common = Counter(pred_evs) & Counter(gold_evs)
    #     num_same = sum(common.values())
    #     ev_tp += num_same
    #     ev_pred_n += len(pred_evs)
    #     ev_gold_n += len(gold_evs)

    # ev_p = 1.0 * ev_tp / ev_pred_n
    # ev_r = 1.0 * ev_tp / ev_gold_n
    # if ev_p == 0 or ev_r == 0: ev_f1 = 0
    # else:
    #     ev_f1 = 2*ev_p*ev_r/(ev_p+ev_r)

    # print("event detection f1: ", ev_f1)
    # print(counts)

    result = compute_metric(instances=instances)
    # result['EVENT_F1'] = ev_f1

    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4)





if __name__ == '__main__':
    main()