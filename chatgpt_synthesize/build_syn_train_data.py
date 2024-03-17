import os
import json
from collections import defaultdict
import re

def filter_span(span):
    span = span.lower()
    if "n/a" in span:
        return True
    if "null" in span or "none" in span:
        return True
    if "not mentioned" in span:
        return True
    return False

def transfer_list_arguments(arguments):
    out = {}
    try:
        for argument in arguments:
            out[argument["argument_type"]] = argument["argument_span"]
    except:
        return None
    return out


def dispatch_cross_valid_data():
    src_folder = "chatgpt_synthesize/output_mix_strategy_run2"
    output_folder = "data/datasets/phee2/cross/"
    

    for i in range(0, 5):
        aug_data = []
        out_file = os.path.join(output_folder,"cross_split_%d"%(i+1), "train_aug_r2.json")
        # print(out_file)
        for j in [i%5, (i+1)%5, (i+2)%5]:
            part_file = os.path.join(src_folder, "partition%d.json"%(j+1))
            # print(part_file)
            with open(part_file, 'r') as f:
                aug_data += f.readlines()

        with open(out_file, 'w') as f:
            f.writelines(aug_data)

            


def main():

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
        "combination_drug": "Treatment.Combination.Drug"
    }

    PARENT_TO_CHILD = {
        "Subject": ["Race", "Age", "Gender", "Population", "Disorder"],
        "Treatment": ["Duration", "Time_elapsed","Route","Freq","Dosage", "Disorder", "Drug", "Combination.Drug"],
        "Effect": [],
    }

    out_data = []
    gen_folder = "chatgpt_synthesize/output_4_constrain_drug_and_effect"
    output_file = "chatgpt_synthesize/train_aug_constrain_drug_and_effect.json"
    illegal_case = 0
    wrong_event_type = 0
    wrong_arg_type = 0
    unfound_trigger = 0
    unfound_arg = 0
    filter_cases = 0

    for file_name in os.listdir(gen_folder):
        if 'constraint_drugs' in file_name or 'fail_cases' in file_name:
            continue
        instance = {}
        sample_id = file_name.split('.')[0] + "_0"
        instance["id"] = sample_id
        to_filter = False
        with open(os.path.join(gen_folder, file_name), 'r') as f:
            try:
                line = f.read()
                out_str = re.search("\{.*\}", line).group()
                gpt_dict = json.loads(out_str)
                sentence = gpt_dict["sentence"]
                events = gpt_dict["output"]
            except:
                illegal_case += 1
                print("illegal generation: %s"%file_name)
                continue

            instance["context"] = sentence
            if len(events) > 1:
                instance["is_mult_event"] = True
            instance["annotations"] = [{"events": []}]
            ent_id = 1
            evt_id = 1
            for evt in events:
                oevt = {}
                oevt["event_id"] = "E%d"%(evt_id)
                evt_id += 1
                event_type = evt["event_type"]
                if event_type not in evt_type_map:
                    wrong_event_type += 1
                    to_filter = True
                    continue
                else:
                    event_type = evt_type_map[event_type]
                oevt["event_type"] = event_type
                trigger = evt["event_trigger"].strip()
                trigger_match = re.search(re.escape(trigger), sentence, flags=re.I)
                if not trigger_match: 
                    unfound_trigger += 1
                    print("unfound trigger: %s, %s"%(trigger, file_name))
                    to_filter = True
                oevt["Trigger"] = {"text": [[trigger_match.group() if trigger_match else trigger]], "start": [[trigger_match.start() if trigger_match else -1]], "entity_id": ["T%d"%(ent_id)]}
                ent_id += 1

                arguments = defaultdict(list)
                if type(evt["arguments"]) == type([]):
                    evt["arguments"] = transfer_list_arguments(evt["arguments"])
                    if not evt["arguments"]:
                        illegal_case += 1
                        print("unformated arguments: %s"%file_name)
                        continue
                for arg_type, arg_span in evt["arguments"].items():
                    if filter_span(arg_span): continue
                    if arg_type not in arg_type_map:
                        wrong_arg_type += 1
                        to_filter = True
                        continue
                    arguments[arg_type_map[arg_type]] += arg_span.split(";")

                for main_type in PARENT_TO_CHILD:
                    if main_type in arguments:
                        if main_type not in oevt:
                            oevt[main_type] = {"text": [], "start": [], "entity_id": []}
                        for span in arguments[main_type]:
                            span = span.strip()
                            span_match = re.search(re.escape(span), sentence, flags=re.I)
                            if not span_match:
                                unfound_arg += 1
                                print("unfound arg: %s, %s"%(span, file_name))
                                to_filter = True
                            oevt[main_type]["text"].append([span_match.group() if span_match else span])
                            oevt[main_type]["start"].append([span_match.start() if span_match else -1])
                            oevt[main_type]["entity_id"].append("T%d"%(ent_id))
                            ent_id += 1

                    for sub_type in PARENT_TO_CHILD[main_type]:
                        full_sub_type = main_type + "." + sub_type
                        if full_sub_type in arguments:
                            if main_type not in oevt:
                                oevt[main_type] = {"text": [], "start": [], "entity_id": []}
                            if full_sub_type == 'Treatment.Combination.Drug':
                                comb_evt = {"event_id": "E%d"%evt_id, 
                                            "event_type": "Combination", 
                                            "Trigger": {"text": [["and"]], "start": [[-1]], "entity_id":["T%d"%ent_id]},
                                            "Drug": {"text": [], "start": [], "entity_id":[]}
                                            } 
                                evt_id += 1
                                ent_id += 1
                                for span in arguments[full_sub_type]:
                                    span = span.strip()
                                    span_match = re.search(re.escape(span), sentence, flags=re.I)
                                    if not span_match:
                                        unfound_arg += 1
                                        print("unfound arg: %s, %s"%(span, file_name))
                                        to_filter = True
                                    comb_evt["Drug"]["text"].append([span_match.group() if span_match else span])
                                    comb_evt["Drug"]["start"].append([span_match.start() if span_match else -1])
                                    comb_evt["Drug"]["entity_id"].append("T%d"%(ent_id))
                                    ent_id += 1
                                oevt["Treatment"]["Combination"] = [comb_evt]
                            else:
                                if sub_type not in oevt[main_type]:
                                    oevt[main_type][sub_type] = {"text": [], "start": [], "entity_id":[]}
                                for span in arguments[full_sub_type]:
                                    span = span.strip()
                                    span_match = re.search(re.escape(span), sentence, flags=re.I)

                                    if not span_match:
                                        unfound_arg += 1
                                        to_filter = True
                                        print("unfound arg: %s, %s"%(span, file_name))
                                    oevt[main_type][sub_type]["text"].append([span_match.group() if span_match else span])
                                    oevt[main_type][sub_type]["start"].append([span_match.start() if span_match else -1])
                                    oevt[main_type][sub_type]["entity_id"].append("T%d"%(ent_id))
                                    ent_id += 1

                instance["annotations"][0]["events"].append(oevt)
        if to_filter:
            filter_cases += 1
        else:
            out_data.append(instance)

    with open(output_file, 'w') as f:
        for data in out_data:
            f.write(json.dumps(data)+"\n")

    print("illegal_case:", illegal_case)
    print("wrong_event_type:", wrong_event_type)
    print("wrong_arg_type:", wrong_arg_type)
    print("unfound_trigger:", unfound_trigger)
    print("unfound_arg:", unfound_arg)
    print("filter_cases:", filter_cases)
    print(len(out_data))




if __name__ == '__main__':
    main()
    # dispatch_cross_valid_data()