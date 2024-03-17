import json
import os
import shutil
from collections import defaultdict
import random 

def get_phee_ebm_model_test_data():
    src_file = "/Data/projects/PV_UIE/data/converted_data/text2spotasoc/event/phee/test.json"
    target_file = "/Data/projects/PV_UIE/data/converted_data/text2spotasoc/event/phee/ebm-sent-test.json"

    with open(src_file, "r") as f:
        with open(target_file, "w") as fout:
            lines = f.readlines()
            for lid, line in enumerate(lines):
                data = json.loads(line)
                data['text_id'] = str(lid)
                new_line = json.dumps(data)
                fout.write(new_line+"\n")

def get_mix_training_data():
    phee_folder = "data/converted_data/text2spotasoc/event/phee"
    ebm_folder = "data/converted_data/text2spotasoc/entity/ebm-nlp-sent"
    n2c2_folder = "data/converted_data/text2spotasoc/relation/n2c2"
    tgt_folder = "data/converted_data/text2spotasoc/event/phee-ebm-n2c2"

    # merge entity schema
    entities = []
    for folder in [phee_folder, ebm_folder, n2c2_folder]:
        with open(os.path.join(folder, "entity.schema"), 'r') as f:
            ents = json.loads(f.readlines()[0])
            entities += ents
    entities = list(set(entities))
    with open(os.path.join(tgt_folder, "entity.schema"), 'w') as f:
        f.write(json.dumps(entities)+"\n")
        f.write(json.dumps([])+"\n")
        f.write(json.dumps({})+"\n")

    # merge relation schema
    shutil.copy(os.path.join(n2c2_folder, "relation.schema"), os.path.join(tgt_folder, "relation.schema"))

    # merge event schema
    shutil.copy(os.path.join(phee_folder, "event.schema"), os.path.join(tgt_folder, "event.schema"))

    # merge record schema
    spots = []
    asocs = []
    spot2asoc = defaultdict(list)
    for folder in [phee_folder, ebm_folder, n2c2_folder]:
        with open(os.path.join(folder, "record.schema"), 'r') as f:
            lines = f.readlines()
            spots += json.loads(lines[0])
            asocs += json.loads(lines[1])
            for k, v in json.loads(lines[2]).items():
                spot2asoc[k] += v
    spots = list(set(spots))
    asocs = list(set(asocs))
    for k in spot2asoc:
        spot2asoc[k] = list(set(spot2asoc[k]))

    with open(os.path.join(tgt_folder, "record.schema"), 'w') as f:
        f.write(json.dumps(spots)+"\n")
        f.write(json.dumps(asocs)+"\n")
        f.write(json.dumps(spot2asoc)+"\n")

    # copy dev and test data
    shutil.copy(os.path.join(phee_folder, "test.json"), os.path.join(tgt_folder, "test.json"))
    shutil.copy(os.path.join(phee_folder, "val.json"), os.path.join(tgt_folder, "val.json"))

    # merge train data
    merge_lines = []
    with open(os.path.join(phee_folder, "train.json"), 'r') as f:
        merge_lines = f.readlines()

    for folder in [n2c2_folder, ebm_folder]:
        cand_lines = []
        with open(os.path.join(folder, "train.json"), 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                if 'text_id' in data:
                    data.pop('text_id')
                if data["spot_asoc"]:
                    cand_lines.append(json.dumps(data)+"\n")
        print(len(cand_lines))
        random.shuffle(cand_lines)
        merge_lines += cand_lines[:2898]

    with open(os.path.join(tgt_folder, "train.json"), 'w') as f:
        f.write("".join(merge_lines))


    
def copy_n2c2_to_brat():
    src_gold_folder = "data/datasets/n2c2/source/test"
    tgt_gold_folder = "brat/brat-v1.3_Crunchy_Frog/data/n2c2_gold"
    src_pred_folder = "hf_models/n2c2-uie-base-en/test_pred"
    tgt_pred_folder = "brat/brat-v1.3_Crunchy_Frog/data/n2c2_pred"

    for file_name in os.listdir(src_gold_folder):
        shutil.copy(os.path.join(src_gold_folder, file_name), os.path.join(tgt_gold_folder, file_name))
        if file_name.endswith(".txt"):
            shutil.copy(os.path.join(src_gold_folder, file_name), os.path.join(tgt_pred_folder, file_name))
    
    for file_name in os.listdir(src_pred_folder):
        shutil.copy(os.path.join(src_pred_folder, file_name), os.path.join(tgt_pred_folder, file_name))



if __name__ == '__main__':
    # get_mix_training_data()
    # copy_n2c2_to_brat()