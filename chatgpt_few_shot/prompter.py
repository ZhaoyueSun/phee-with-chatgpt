import os
import openai
import random
import json
from collections import OrderedDict, defaultdict
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from chatgpt_few_shot.tree_kernel import compute_tree_similarity
import spacy
import torch
import gzip

class Prompter:
    def __init__(self, data_folder, demo_strategy='random', demo_num=1, instruction_template=None, query_template=None, random_seed=None):

        if random_seed == None:
            random_seed = random.randint(0, 1000)
        random.seed(random_seed)

        openai.organization = "org-G1AdYymvZPETTLDmtFln1REi"
        with open('chatgpt_few_shot/open-ai-key.txt', 'r') as f:
            openai.api_key = f.readline().strip()
        openai.Model.list()

        self.demo_strategy = demo_strategy
        self.demo_num = demo_num

        self.data_folder = data_folder
        self.instruction_template = instruction_template
        self.query_template = query_template
        self.train_list = []
        self.dev_list = []
        self.test_list = []
        self.data = {}
        self.read_data()

        self.id_list = {
            'train': self.train_list,
            'dev': self.dev_list,
            'test': self.test_list
        }

        if demo_strategy == 'SBert':
            # self.model = SentenceTransformer('allenai/scibert_scivocab_uncased')
            self.model = SentenceTransformer('all-mpnet-base-v2')
            sentences = []
            for sample_id in self.train_list:
                instance = self.data[sample_id]
                sentence = instance['sentence']
                sentences.append(sentence)
            self.embeddings = self.model.encode(sentences, convert_to_tensor=True)
        elif demo_strategy == 'SBert-Type':
            # self.model = SentenceTransformer('allenai/scibert_scivocab_uncased')
            self.model = SentenceTransformer('all-mpnet-base-v2')
            ade_sents = []
            pte_sents = []
            self.ade_list = []
            self.pte_list = []
            for sample_id in self.train_list:
                instance = self.data[sample_id]
                sentence = instance['sentence']
                if 'adverse event' in instance['event_types']:
                    ade_sents.append(sentence)
                    self.ade_list.append(sample_id)
                if 'potential therapeutic event' in instance['event_types']:
                    pte_sents.append(sentence)
                    self.pte_list.append(sample_id)
            self.ade_embeddings = self.model.encode(ade_sents, convert_to_tensor=True)
            self.pte_embeddings = self.model.encode(pte_sents, convert_to_tensor=True)

        elif self.demo_strategy == 'BM25':
            tokenized_sentences = []
            for sample_id in self.train_list:
                instance = self.data[sample_id]
                sentence = instance['sentence']
                tokenized_sentences.append(sentence.split(" "))
            self.bm25 = BM25Okapi(tokenized_sentences)

        elif self.demo_strategy == 'BM25-Type':
            ade_sents = []
            pte_sents = []
            self.ade_list = []
            self.pte_list = []
            for sample_id in self.train_list:
                instance = self.data[sample_id]
                sentence = instance['sentence']
                if 'adverse event' in instance['event_types']:
                    ade_sents.append(sentence.split(" "))
                    self.ade_list.append(sample_id)
                if 'potential therapeutic event' in instance['event_types']:
                    pte_sents.append(sentence.split(" "))
                    self.pte_list.append(sample_id)
                
            self.ade_bm25 = BM25Okapi(ade_sents)
            self.pte_bm25 = BM25Okapi(pte_sents)

        elif self.demo_strategy == 'TreeKernel' or self.demo_strategy == 'TreeKernel-Type':
            self.nlp = spacy.load("en_core_web_sm")

        elif self.demo_strategy == 'InfoDist-Type': # information distance
            ade_sents = []
            pte_sents = []
            self.ade_list = []
            self.pte_list = []
            for sample_id in self.train_list:
                instance = self.data[sample_id]
                sentence = instance['sentence']
                if 'adverse event' in instance['event_types']:
                    ade_sents.append(sentence.split(" "))
                    self.ade_list.append(sample_id)
                if 'potential therapeutic event' in instance['event_types']:
                    pte_sents.append(sentence.split(" "))
                    self.pte_list.append(sample_id)


    def read_data(self):

        arg_type_map = OrderedDict({
            "subject disorder": "subject_disorder",
            "time elapsed":"time_elapsed",
            "treatment disorder":"indication",
            "combination drug":"combination_drug",
        })
        splits = ["train", "val", "test"]
        for split in splits:
            data_file = os.path.join(self.data_folder, split+".json")
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
                    self.data[sample["id"]] = sample
                    if split == 'train': self.train_list.append(sample["id"])
                    elif split == 'val': self.dev_list.append(sample["id"])
                    elif split == 'test': self.test_list.append(sample["id"])
                    else: raise Exception("No this data split: %s!"%split)
    

    def get_data_size(self, split=None):
        if split:
            return len(self.id_list[split])
        else:
            return len(self.train_list)+len(self.dev_list)+len(self.test_list)
        
    def sample_an_id(self, split):
        id_list = self.id_list[split]
        idx = random.randint(0, len(id_list)-1)
        return id_list[idx]
    
    def get_an_instance(self, sample_id):
        return self.data[sample_id]
    
    def get_instruction_prompt(self):

        if os.path.exists(self.instruction_template):
            with open(self.instruction_template, 'r') as f:
                instruction = json.load(f)
                return instruction
        else:
            raise Exception("File does not exists: instruction template!")
        

    def get_an_example_prompt(self, sample_id):
        instance = self.data[sample_id]
        prompt = self.get_query_prompt(instance)
        answer = instance['answer']
        prompt += [{"role": "assistant", "content": answer}]
        return prompt
    
    def get_query_prompt(self, instance):
        sentence = instance['sentence']
        if os.path.exists(self.query_template):
            with open(self.query_template, 'r') as f:
                prompt = json.load(f)
                prompt[0]['content'] = prompt[0]['content'].replace('<sentence>', sentence)
                return prompt
        else:
            raise Exception("File does not exists: query template!")
    

    def get_chatgpt_response(self, prompts):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=prompts,
        temperature=0
        )

        answer = response['choices'][0]['message']['content']

        return answer
    
    def get_prompts(self, instance):
        prompt = []
        prompt += self.get_instruction_prompt()
        prompt += self.get_demostrations_prompt(instance)
        prompt += self.get_query_prompt(instance)
        return prompt
   
    def get_result(self, instance):
        prompts = self.get_prompts(instance)
        answer = self.get_chatgpt_response(prompts)
        result = {'prompt': prompts,
        'answer': answer}

        return result
    
    def get_demostrations_prompt(self, instance):
        if self.demo_strategy == 'random':
            demo_ids = {
                "adverse event":[],
                "potential therapeutic event": []
            }
            while True:
                brk = True
                for k in demo_ids:
                    if len(demo_ids[k]) < self.demo_num:
                        brk = False
                if brk: break

                sid = self.sample_an_id('train')
                sample = self.get_an_instance(sid)
                if len(sample['event_types']) > 1:
                    continue
                if len(demo_ids[sample['event_types'][0]]) < self.demo_num:
                    demo_ids[sample['event_types'][0]].append(sid)
            demo_ids = [sid for sid_list in demo_ids.values() for sid in sid_list]

        elif self.demo_strategy == 'SBert':
            demo_ids = []
            top_k = min(self.demo_num, len(self.embeddings))
            query = instance['sentence']
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            for score, idx in zip(top_results[0], top_results[1]):
                demo_ids.append(self.train_list[idx])
        
        elif self.demo_strategy == 'SBert-Type':
            demo_ids = []
            query = instance['sentence']
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            top_k = min(self.demo_num, len(self.ade_embeddings), len(self.pte_embeddings))

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            ade_scores = util.cos_sim(query_embedding, self.ade_embeddings)[0]
            ade_top = torch.topk(ade_scores, k=top_k)
            for score, idx in zip(ade_top[0], ade_top[1]):
                demo_ids.append(self.ade_list[idx])

            pte_scores = util.cos_sim(query_embedding, self.pte_embeddings)[0]
            pte_top = torch.topk(pte_scores, k=2*top_k)
            for score, idx in zip(pte_top[0], pte_top[1]):
                if len(demo_ids) >= 2*top_k: break
                sid = self.pte_list[idx]
                if sid not in demo_ids:
                    demo_ids.append(sid)

        elif self.demo_strategy == 'BM25':
            demo_ids = []
            query = instance['sentence']
            tokenized_query = query.split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_results = torch.topk(torch.tensor(scores), k=self.demo_num)
            for score, idx in zip(top_results[0], top_results[1]):
                idx = idx.item()
                demo_ids.append(self.train_list[idx])

        elif self.demo_strategy == 'BM25-Type':
            demo_ids = []
            query = instance['sentence']
            tokenized_query = query.split(" ")
            ade_scores = self.ade_bm25.get_scores(tokenized_query)
            ade_tops = torch.topk(torch.tensor(ade_scores), k=self.demo_num)
            for score, idx in zip(ade_tops[0], ade_tops[1]):
                idx = idx.item()
                demo_ids.append(self.ade_list[idx])
            pte_scores = self.pte_bm25.get_scores(tokenized_query)
            pte_tops = torch.topk(torch.tensor(pte_scores), k=2*self.demo_num)
            for score, idx in zip(pte_tops[0], pte_tops[1]):
                if len(demo_ids) >= 2*self.demo_num: break
                idx = idx.item()
                sid = self.pte_list[idx]
                if sid not in demo_ids:
                    demo_ids.append(sid)

        elif self.demo_strategy == 'TreeKernel':
            scores = []
            demo_ids = []
            query = instance['sentence']
            for sample_id in self.train_list:
                sample = self.data[sample_id]
                sentence = sample['sentence']
                scores.append(compute_tree_similarity(query, sentence, self.nlp))
            top_results = torch.topk(torch.tensor(scores), k=self.demo_num)
            for score, idx in zip(top_results[0], top_results[1]):
                idx = idx.item()
                demo_ids.append(self.train_list[idx])

        elif self.demo_strategy == 'TreeKernel-Type':
            ade_scores = []
            pte_scores = []
            ade_list = []
            pte_list = []
            demo_ids = []
            query = instance['sentence']
            for sample_id in self.train_list:
                sample = self.data[sample_id]
                sentence = sample['sentence']
                score = compute_tree_similarity(query, sentence, self.nlp)
                if 'adverse event' in sample['event_types']:
                    ade_scores.append(score)
                    ade_list.append(sample_id)
                if 'potential therapeutic event' in sample['event_types']:
                    pte_scores.append(score)
                    pte_list.append(sample_id)
                
            ade_tops = torch.topk(torch.tensor(ade_scores), k=self.demo_num)
            for _, idx in zip(ade_tops[0], ade_tops[1]):
                idx = idx.item()
                demo_ids.append(ade_list[idx])

            pte_tops = torch.topk(torch.tensor(pte_scores), k=2*self.demo_num)
            for _, idx in zip(pte_tops[0], pte_tops[1]):
                if len(demo_ids) >= 2*self.demo_num: break
                idx = idx.item()
                sid = pte_list[idx]
                if sid not in demo_ids:
                    demo_ids.append(sid)

        elif self.demo_strategy == 'InfoDist-Type':
            demo_ids = []
            query = instance['sentence']
            tokenized_query = query.split(" ")
            Cquery = len(gzip.compress(query.encode()))
            ade_scores = []
            for sentence in self.ade_list:
                Csent = len(gzip.compress(sentence.encode()))
                concat = " ".join([query, sentence])
                Cconcat = len(gzip.compress(concat.encode()))
                ncd = (Cconcat - min(Cquery, Csent))/(max(Cquery, Csent))
                ade_scores.append(ncd)
            ade_tops = torch.topk(torch.tensor(ade_scores), k=self.demo_num)
            for score, idx in zip(ade_tops[0], ade_tops[1]):
                idx = idx.item()
                demo_ids.append(self.ade_list[idx])
            
            pte_scores = []
            for sentence in self.pte_list:
                Csent = len(gzip.compress(sentence.encode()))
                concat = " ".join([query, sentence])
                Cconcat = len(gzip.compress(concat.encode()))
                ncd = (Cconcat - min(Cquery, Csent))/(max(Cquery, Csent))
                pte_scores.append(ncd)
            pte_tops = torch.topk(torch.tensor(pte_scores), k=2*self.demo_num)
            for score, idx in zip(pte_tops[0], pte_tops[1]):
                if len(demo_ids) >= 2*self.demo_num: break
                idx = idx.item()
                sid = self.pte_list[idx]
                if sid not in demo_ids:
                    demo_ids.append(sid)

        

        else:
            raise Exception("Unsupported demo selection strategy: %s"%self.demo_strategy)
        
        prompts = []
        for sample_id in demo_ids:
            prompts += self.get_an_example_prompt(sample_id)

        return prompts


    



    

