import os
import openai
import random
import json
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
from chatgpt_zero_shot.utils import read_data
import tabulate
import time


class GPTConnector:
    def __init__(self, model:str="gpt-3.5-turbo-0301"):
        openai.organization = "org-G1AdYymvZPETTLDmtFln1REi"
        with open('chatgpt_zero_shot/open-ai-key.txt', 'r') as f:
            openai.api_key = f.readline().strip()
        openai.Model.list()

        self.model = model

    def get_chatgpt_response(self, prompts, temperature=0):
        response = openai.ChatCompletion.create(
        model=self.model,
        messages=prompts,
        temperature=temperature
        )
        answer = response['choices'][0]['message']['content']

        return answer



def main():

    
    # get chatgpt indication response
    prompt_type = "main_args"
    with open("chatgpt_zero_shot/templates/%s.json"%prompt_type, 'r') as f:
        prompt = json.load(f)
    instruction = prompt[0]['content']

    gpt = GPTConnector()

    src_folder =  "data/datasets/phee2/source"
    splits = ['dev', 'test']
    for split in splits:
        start_time = time.time()
        output_folder = "chatgpt_zero_shot/output/%s/%s"%(prompt_type, split)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        log_file = os.path.join(output_folder,"fail_cases.json")
        fail_cases = []
        data = read_data(os.path.join(src_folder, "%s.json"%split))
        
        count = 0
        for sample_id, sample in data.items():
            count += 1

            if count % 50 == 0:
                print("Processed %s set: %d/%d cases."%(split, count, len(data)))
            try:
                sentence = sample['context']
                prompt = [{"role": "user", "content": instruction.replace("<sentence>", sentence)}]
                answer = gpt.get_chatgpt_response(prompt)
            except:
                fail_cases.append(sample_id)
                print("Failed to get chatgpt answer: %s"%sample_id)
                with open(log_file, 'w') as f:
                    json.dump(fail_cases, f)
                    f.flush()
                continue

            output_file = os.path.join(output_folder, sample_id+".json")
            with open(output_file, 'w') as f:
                f.write(answer)

        print("--- time for running the %s set: %s seconds ---" % (split, time.time() - start_time))





if __name__ == '__main__':
    main()