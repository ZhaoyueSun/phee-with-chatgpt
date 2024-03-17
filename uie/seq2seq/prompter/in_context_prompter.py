import os
import random
import json
from collections import OrderedDict

import torch
from uie.seq2seq.features import DemoFeature

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

class InContextPrompter:
    def __init__(self, data, demo_num=0, prompt_strategy='random'):

        
        remove_columns = []
        for k in data.features.keys():
            if k not in DemoFeature:
                remove_columns.append(k)
        self.data = data.remove_columns(remove_columns)

        self.demo_num = demo_num
        self.prompt_strategy = prompt_strategy

        if prompt_strategy == 'SBert':
            # self.model = SentenceTransformer('allenai/scibert_scivocab_uncased')
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = self.data['text']
            self.embeddings = self.model.encode(sentences, convert_to_tensor=True)
        elif prompt_strategy == 'BM25':
            sentences = self.data['text']
            tokenized_sentences = [doc.split(" ") for doc in sentences]
            self.bm25 = BM25Okapi(tokenized_sentences)
            print(type(self.bm25))



    def get_in_context_examples(self, query):
        demos = []
        if self.prompt_strategy == 'random':
            idxs = random.sample(range(0, len(self.data)-1), self.demo_num)
            for idx in idxs:
                demos.append(self.data[idx])
        elif self.prompt_strategy == 'SBert':
            top_k = min(self.demo_num, len(self.embeddings))
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            for score, idx in zip(top_results[0], top_results[1]):
                idx = idx.item()
                demos.append(self.data[idx])
        elif self.prompt_strategy == 'BM25':
            tokenized_query = query.split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_results = torch.topk(torch.tensor(scores), k=self.demo_num)
            for score, idx in zip(top_results[0], top_results[1]):
                idx = idx.item()
                demos.append(self.data[idx])
        else:
            raise Exception("Unsupported demo selection strategy: %s"%self.demo_strategy)

        return demos



    
   






    

