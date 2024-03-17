import os
import json
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
import tqdm
from chatgpt_few_shot.prompter import Prompter


def main():
    data_folder = "data/converted_data/text2spotasoc/event/phee_2/"
    demo_strategy = 'BM25-Type'
    # demo_num = 1
    for demo_num in [5]:
        for split in ['test']:
            instruction_template = "chatgpt_few_shot/instruction_templates/explanation.json"
            query_template = "chatgpt_few_shot/query_templates/explanation.json"
            prompter = Prompter(data_folder, demo_num=demo_num, demo_strategy=demo_strategy, instruction_template=instruction_template, query_template=query_template, random_seed=42)

            output_folder = "chatgpt_few_shot/output/%s_demo_%d/%s"%(demo_strategy, demo_num, split)
            if not os.path.exists(output_folder): 
                os.makedirs(output_folder)
            log_file = os.path.join(output_folder, 'fail_cases.json')
            if not os.path.exists(log_file):
                print("No fail cases:%s"%output_folder)
                continue
            with open(log_file, 'r') as f:
                fail_cases = json.load(f)

            new_log_file = os.path.join(output_folder, 'fail_cases_2.json')
            new_fail_cases = []
            case_list = prompter.test_list if split == 'test' else prompter.dev_list
            for sample_id in tqdm.tqdm(case_list):
                if sample_id not in fail_cases:
                    continue
                try:
                    test_instance = prompter.get_an_instance(sample_id)
                    result = prompter.get_result(test_instance)
                except:
                    new_fail_cases.append(sample_id)
                    with open(new_log_file, 'w') as f:
                        json.dump(new_fail_cases, f)
                    print("fail case: %s"%sample_id)
                    continue

                with open(os.path.join(output_folder, sample_id+".json"), 'w') as f:
                    json.dump(result, f)
        
        



if __name__ == '__main__':
    main()