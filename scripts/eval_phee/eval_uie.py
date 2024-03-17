"""
Evaluate UIE model result for PHEE. 
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

def read_pred_results(pred_file, lb_mapper):
    outputs = []
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            data = edict(data)
            instance = defaultdict(list)

            for event in data.event.string:
                ev_type = lb_mapper[event.type]
                if 'trigger' in event:
                    instance[ev_type+".Trigger"].append(event.trigger)
                if 'roles' in event:
                    for arg_type, arg_text in event.roles:
                         instance[ev_type+"."+lb_mapper[arg_type]].append(arg_text)
            
            outputs.append(instance)
    return outputs


def get_label_mapper(config_file):
    with open(config_file) as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    mapper = {}
    for key, value in config['mapper'].items():
        mapper[value] = key

    return mapper
    
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', dest='gold_file', help='Gold File: Use PHEE original json rather than UIE converted one.')
    parser.add_argument('-p', dest='pred_file', help='Pred File')
    parser.add_argument('-o', dest='out_file', help='Output File')
    parser.add_argument('-c', dest='config_file', help='Label Mapper Config File')
    parser.add_argument('-m', dest='eval_mult', action="store_true", help='Only evaluate multi-event cases.')
    parser.add_argument('-s', dest='eval_single', action="store_true", help='Only evaluate single-event cases.')
    parser.add_argument('-e', dest='eval_event_type', help='Only evaluate such event type', default=None)
    options = parser.parse_args()

    if options.eval_mult and options.eval_single:
        raise Exception("cannot simultaneously set -m and -s")

    if not os.path.exists(options.gold_file):
        raise Exception("Gold file path does not exist!")
    if not os.path.exists(options.pred_file):
        raise Exception("Pred file path does not exist!")

    
    lb_mapper = get_label_mapper(options.config_file)
    gold_instances = read_gold_result(options.gold_file)
    pred_instances = read_pred_results(options.pred_file, lb_mapper)

    assert(len(gold_instances) == len(pred_instances))

    instances = []
    ev_tp = 0
    ev_pred_n = 0
    ev_gold_n = 0

    counts = defaultdict(int)

    for preds, golds in zip(pred_instances, gold_instances):
        instance_id = golds['id'][0]
        if options.eval_mult and not golds['is_mult']: continue
        if options.eval_single and golds['is_mult']: continue
        question_types = list(set(list(preds.keys())+list(golds.keys())))
        question_types.remove('id')
        question_types.remove('is_mult')
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
            if options.eval_event_type:
                if options.eval_event_type in qtype:
                    instances.append(instance)
                    if qtype in golds:
                        counts[qtype] +=1
            else:
                instances.append(instance)

        # for event classification evaluation
        pred_evs = []
        gold_evs = []
        for k in preds:
            if 'Trigger' in k:
                ev_type = k.split('.')[0]
                pred_evs.append(ev_type)

        for k in golds:
            if 'Trigger' in k:
                ev_type = k.split('.')[0]
                for _ in golds[k]:
                    gold_evs.append(ev_type)

        common = Counter(pred_evs) & Counter(gold_evs)
        num_same = sum(common.values())
        ev_tp += num_same
        ev_pred_n += len(pred_evs)
        ev_gold_n += len(gold_evs)

    ev_p = 1.0 * ev_tp / ev_pred_n
    ev_r = 1.0 * ev_tp / ev_gold_n
    if ev_p == 0 or ev_r == 0: ev_f1 = 0
    else:
        ev_f1 = 2*ev_p*ev_r/(ev_p+ev_r)

    print("event detection f1: ", ev_f1)
    print(counts)

    result = compute_metric(instances=instances)
    result['EVENT_F1'] = ev_f1

    with open(options.out_file, 'w') as f:
        json.dump(result, f, indent=4)





if __name__ == '__main__':
    model_name = 'mistral-7b'
    # sys.argv.extend(['-c', 'dataset_processing/data_config/event/phee.yaml']) 
    sys.argv.extend(['-c', 'dataset_processing/data_config/event/phee2.yaml']) 
    
    # # sys.argv.extend(['-g', 'data/datasets/phee/phee2.0/mod_disorder/test.json'])
    sys.argv.extend(['-p', 'hf_models/%s/test_preds_record.txt'%model_name])
    sys.argv.extend(['-g', 'data/datasets/phee2/source/test.json'])
    sys.argv.extend(['-o', 'hf_models/%s/test_result.json'%model_name])
    # sys.argv.extend(['-o', 'hf_models/%s/test_result_mult_event.json'%model_name])
    
    # sys.argv.extend(['-m'])
    # sys.argv.extend(['-s'])
    # sys.argv.extend(['-e', 'Potential_therapeutic_event'])
    # sys.argv.extend(['-p', 'hf_models/%s/eval_preds_record.txt'%model_name])
    # sys.argv.extend(['-g', 'data/datasets/phee2/source/dev.json'])
    # sys.argv.extend(['-o', 'hf_models/%s/eval_result.json'%model_name])


    main()