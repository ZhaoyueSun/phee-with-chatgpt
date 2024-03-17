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
    prompt_type = "sub_args"
    with open("chatgpt_zero_shot/templates/%s.json"%prompt_type, 'r') as f:
        qa_dict = json.load(f)

    instruction = "Answer the question related to the given sentence and given event information. The answer should be a span exactly extracted from the sentence. If no answer can be found from the sentence, return N/A. Sentence: <sentence> Event: Event type: <event_type> Subject: <subject> Treatment: <treatment> Effect: <effect>. <question>"

    gpt = GPTConnector()

    src_folder =  "data/datasets/phee2/source"
    # splits = ['dev', 'test']
    splits = ['test']
    for split in splits:
        start_time = time.time()
        output_folder = "chatgpt_zero_shot/output/%s/%s"%(prompt_type, split)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        main_ans_folder = "chatgpt_zero_shot/output/%s/%s"%("main_args", split)

        log_file = os.path.join(output_folder,"fail_cases.json")
        with open(log_file, 'r') as f:
            fail_cases = json.load(f)

        new_log_file = os.path.join(output_folder,"fail_cases_2.json")
        new_fail_cases = []

        data = read_data(os.path.join(src_folder, "%s.json"%split))
        

        for sample_id, sample in data.items():
            if sample_id in fail_cases:
                print("%sset: %s"%(split, sample_id))
                
                sentence = sample['context']
                # get first step answer
                main_ans_file = os.path.join(main_ans_folder, sample_id+".json")
                output = []
                with open(main_ans_file, 'r') as f:
                    events = json.load(f)
                for evt in events:
                    evt_type = evt["event_type"] if "event_type" in evt else "N/A"
                    subject = evt["subject"] if "subject" in evt and evt["subject"] is not None else "N/A"
                    treatment = evt["treatment"] if "treatment" in evt and evt["treatment"] is not None else "N/A"
                    effect = evt["effect"] if "effect" and evt["effect"] is not None in evt else "N/A"
                    out_evt = {
                        "event_type": evt_type,
                        "arguments": [
                            {"argument_type": "subject", "argument_span": subject},
                            {"argument_type": "treatment", "argument_span": treatment},
                            {"argument_type": "effect", "argument_span": effect}
                        ]
                    }
                    for sub_arg, question in qa_dict.items():
                        try:
                            prompt = instruction.replace('<sentence>', sentence).replace('<event_type>', evt_type).replace('<subject>', subject).replace('<treatment>', treatment).replace('<effect>', effect).replace('<question>', question)
                            prompt = [{"role": "user", "content": prompt}]
                            answer = gpt.get_chatgpt_response(prompt)
                            out_evt['arguments'].append({
                                "argument_type": sub_arg,
                                "argument_span": answer
                            })
                            time.sleep(1)
                        except:
                            if sample_id not in new_fail_cases:
                                new_fail_cases.append(sample_id)
                                print("Failed to get chatgpt answer: %s"%sample_id)
                                with open(new_log_file, 'w') as f:
                                    json.dump(new_fail_cases, f)
                                    f.flush()
                            continue
                    output.append(out_evt)

                output_file = os.path.join(output_folder, sample_id+".json")
                with open(output_file, 'w') as f:
                    json.dump(output, f)

    
        print("--- time for running the %s set: %s seconds ---" % (split, time.time() - start_time))





if __name__ == '__main__':
    main()