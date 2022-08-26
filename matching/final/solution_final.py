import os
import json
import nltk
import torch
import faiss
import string
import langdetect
import numpy as np
from flask import (Flask,
                   request,)
from typing import (Dict,
                    List,
                    Tuple,)


app = Flask(__name__)


class Solution(object):
    def __init__(self):
        self.documents = None
        self.index = None
        torch.set_grad_enabled(False)
        
    def load_resources(self):
        self.vocab = json.load(open(os.environ['VOCAB_PATH'],
                               mode='r',
                               encoding='utf-8'))
        
        state_dict = torch.load(os.environ['EMB_PATH_KNRM'])
        self.emb_knrm_shape = state_dict['weight'].shape
        self.emb_knrm = torch.nn.Embedding.from_pretrained(state_dict['weight'],
                                                           freeze=True,
                                                           padding_idx=0)

#         self.emb_glove = list()
#         with open(os.environ['EMB_PATH_GLOVE'], mode='r') as file:
#             for line in file:
#                 self.emb_glove.append(line.split()[0])

        self.mlp_knrm = torch.load(os.environ['MLP_PATH'])
        
        global is_ready
        is_ready = True
    
    def _preprocess(self, input_str: str) -> str:
        table = str.maketrans(string.punctuation,
                              ' '*len(string.punctuation))
        return (input_str
                .translate(table)
                .lower())
    
    def _filter_glove_tokens(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t in self.emb_glove]
    
    def _get_tokens(self, input_str: str) -> List[str]:
        return nltk.word_tokenize(self._preprocess(input_str))
    
    def _get_tokens_ids(self,
                        input_str: str,
                        filter_glove: bool = False) -> List[int]:
        tokens = self._get_tokens(input_str)
        if filter_glove:
            tokens = self._filter_glove_tokens(tokens)
        return [self.vocab.get(t, self.vocab['OOV']) 
                for t in tokens]
        
    def update_index(self, documents: Dict[str, str]) -> int:
        self.documents = documents
        
        tokens_ids = list()
        for d in self.documents:
            ids = self._get_tokens_ids(self.documents[d],
                                       filter_glove=False)
            tokens_ids.append(ids)
            
        vectors = list()
        for ids in tokens_ids:
            embs = self.emb_knrm(torch.LongTensor(ids))
            vectors.append(embs
                           .mean(axis=0)
                           .numpy())
        vectors = np.array(vectors)
            
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(vectors,
                                np.array([int(i) for i in self.documents]))
        
        return self.index.ntotal
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, str]]:
        query_ids = self._get_tokens_ids(query, filter_glove=False)
        
        query_emb = self.emb_knrm(torch.LongTensor(query_ids)).mean(axis=0)
        query_emb = (query_emb
                     .numpy()
                     .reshape(-1, self.emb_knrm_shape[1]))
        
        _, document_ids = self.index.search(query_emb, k)
        
        return [(str(i), self.documents[str(i)])
                for i in document_ids.reshape(-1)]
        

solution = Solution()
is_ready = False


@app.route('/ping')
def ping():
    if not is_ready:
        return {'status': 'not ready'}
    return {'status': 'ok'}


@app.route('/query', methods=['POST'])
def query():
    if solution.index is None:
        return {'status': 'FAISS is not initialized!'}
    
    content = json.loads(request.json)
    queries = content['queries']
    
    results = list()
    for q in queries:
        if langdetect.detect(q) == 'en':
            candidates = solution.search(q)
            
            results.append(candidates)
        else:
            results.append(None)
    
    return {'lang_check': [True if r is not None else False 
                           for r in results],
            'suggestions': results,}


@app.route('/update_index', methods=['POST'])
def update_index():
    content = json.loads(request.json)
    documents = content['documents']
    
    return {'status': 'ok',
            'index_size': solution.update_index(documents)}


solution.load_resources()

