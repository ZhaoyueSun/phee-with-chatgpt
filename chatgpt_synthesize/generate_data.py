import os
import openai
import json
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
import time
from collections import OrderedDict
import tqdm


class GPTConnector:
    def __init__(self, model:str="gpt-3.5-turbo-0301"):
        openai.organization = "org-G1AdYymvZPETTLDmtFln1REi"
        with open('chatgpt_synthesize/open-ai-key.txt', 'r') as f:
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

def read_data(data_file):

    arg_type_map = OrderedDict({
        "subject disorder": "subject_disorder",
        "time elapsed":"time_elapsed",
        "treatment disorder":"indication",
        "combination drug":"combination_drug",
    })

    data = OrderedDict()

    with open(data_file, "r") as f:
        for line in f.readlines():
            sample = {}
            instance = json.loads(line)
            sample["id"] = instance["text_id"]
            sample["sentence"] = instance["text"]
            sample["event_types"] = []
            answer = []
            events = instance['event']
            for event in events:
                sample["event_types"].append(event["type"])
                evt = {
                    "event_type": event["type"],
                    "event_trigger": event["text"],
                    "arguments": OrderedDict({
                        "subject":[],
                        "treatment":[],
                        "effect":[],
                        "age":[],
                        "gender":[],
                        "race":[],
                        "population":[],
                        "subject_disorder":[],
                        "drug":[],
                        "dosage":[],
                        "route":[],
                        "duration":[],
                        "frequency":[],
                        "time_elapsed":[],
                        "indication":[],
                        "combination_drug":[]
                    })
                }

                for argument in event['args']:
                    arg_type = argument['type']
                    arg_type = arg_type_map[arg_type] if arg_type in arg_type_map else arg_type
                    arg_span = argument['text']
                    evt["arguments"][arg_type].append(arg_span)
                for arg_type in evt["arguments"]:
                    if not evt["arguments"][arg_type]:
                        evt["arguments"][arg_type] = "N/A"
                    else:
                        evt["arguments"][arg_type] = "; ".join(evt["arguments"][arg_type])
                answer.append(evt)

            sample["answer"] = json.dumps(answer)
            data[sample["id"]] = sample

    return data

def main():

    
    # get chatgpt indication response
    # with open("chatgpt_synthesize/templates/extraction_instruction.json", 'r') as f:
    #     extraction_instruction = json.load(f)[0]['content']
    with open("chatgpt_synthesize/templates/synthesize_instruction3.json", 'r') as f:
        synthesize_instruction = json.load(f)[0]['content']

    gpt = GPTConnector()

    train_file = "data/converted_data/text2spotasoc/event/phee_2/train.json"
    train_data = read_data(train_file)

    start_time = time.time()
    output_folder = "chatgpt_synthesize/output_2/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_file = os.path.join(output_folder,"fail_cases.json")
    fail_cases = []
    
    count = 0
    for sample_id, sample in tqdm.tqdm(train_data.items()):
        count += 1
        if count <5: continue
        # if count > 5: break
        if count % 50 == 0:
            print("Processed %d/%d cases."%(count, len(train_data)))
        try:
            sentence = sample['sentence']
            # prompt = [{"role": "user", "content": extraction_instruction.replace("<sentence>", sentence)}]
            # extraction = sample['answer']
            # prompt += [{"role": "assistant", "content": extraction}]
            syn_prompt = synthesize_instruction.replace("<sentence>", sentence).replace("<output>", sample['answer'])
            prompt = [{"role": "user", "content": syn_prompt}]
            # prompt += [{"role": "user", "content": synthesize_instruction}]
            answer = gpt.get_chatgpt_response(prompt)
            # print(sample_id)
            # print(sentence)
            # print(sample['answer'])
            # print(answer)
            
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

    print("--- time for running: %s seconds ---" % (time.time() - start_time))





if __name__ == '__main__':
    main()