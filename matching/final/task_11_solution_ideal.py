from flask import Flask
from flask import jsonify, request

import os
import string
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import json
import torch
import torch.nn.functional as F
import faiss
from langdetect import detect


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )

class KNRM(torch.nn.Module):
    def __init__(self, 
                 emb_path: str, 
                 mlp_path: str,
                 kernel_num: int = 21,
                 sigma: float = 0.1, 
                 exact_sigma: float = 0.001,
                 out_layers: List[int] = []
                 ):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.load(emb_path)['weight'],
            freeze=True,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.out_layers = out_layers
        self.sigma = sigma
        self.exact_sigma = exact_sigma

        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(torch.load(mlp_path))
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out

app = Flask(__name__)
model_is_ready = False
index_is_ready = False

class Helper:
    def __init__(self):
        self.emb_path_glove = os.environ['EMB_PATH_GLOVE']
        self.vocab_path = os.environ['VOCAB_PATH']
        self.emb_path_knrm = os.environ['EMB_PATH_KNRM']
        self.mlp_path = os.environ['MLP_PATH']
        torch.set_grad_enabled(False)

    def prepare_model(self):
        self.model = KNRM(
            emb_path=self.emb_path_knrm,
            mlp_path=self.mlp_path
        )
        with open(self.vocab_path, 'r') as f_in:
            self.vocab = json.load(f_in)
        global model_is_ready
        model_is_ready = True

    def _hadle_punctuation(self, inp_str: str) -> str:
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str
    
    def _simple_preproc(self, inp_str: str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self._hadle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)

    def prepare_index(self, documents: Dict[str, str]):
        oov_val = self.vocab["OOV"]
        self.documents = documents
        idxs, docs = [], []
        for idx in documents:
            idxs.append(int(idx))
            docs.append(documents[idx])
        embeddings = []
        emb_layer = self.model.embeddings.state_dict()['weight']
        for d in docs:
            tmp_emb = [self.vocab.get(w, oov_val) for w in self._simple_preproc(d)]
            tmp_emb = emb_layer[tmp_emb].mean(dim = 0)
            embeddings.append(np.array(tmp_emb))          
        embeddings = np.array([embedding for embedding in embeddings]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(idxs))
        index_size = self.index.ntotal
        global index_is_ready
        index_is_ready = True
        
        return index_size

    def _text_to_token_ids(self, text_list: List[str]):
        tokenized = []
        for text in text_list:
            tokenized_text = self._simple_preproc(text)
            token_idxs = [self.vocab.get(i, self.vocab["OOV"]) for i in tokenized_text]
            tokenized.append(token_idxs)
        max_len = max(len(elem) for elem in tokenized)
        tokenized = [elem + [0] * (max_len - len(elem)) for elem in tokenized]
        tokenized = torch.LongTensor(tokenized)    
        return tokenized

    def get_suggestion(self, 
            query: str, ret_k: int = 10, 
            ann_k: int = 100) -> List[Tuple[str, str]]:
        q_tokens = self._simple_preproc(query)
        vector = [self.vocab.get(tok, self.vocab["OOV"]) for tok in q_tokens]
        emb_layer = self.model.embeddings.state_dict()['weight']
        q_emb = emb_layer[vector].mean(dim = 0).reshape(1, -1)
        q_emb = np.array(q_emb).astype(np.float32)
        _, I = self.index.search(q_emb, k = ann_k)
        cands = [(str(i), self.documents[str(i)]) for i in I[0] if i != -1]
        inputs = dict()
        inputs['query'] = self._text_to_token_ids([query] * len(cands))
        inputs['document'] = self._text_to_token_ids([cnd[1] for cnd in cands])
        scores = self.model(inputs)
        res_ids = scores.reshape(-1).argsort(descending=True)
        res_ids = res_ids[:ret_k]
        res = [cands[i] for i in res_ids.tolist()]
        return res

    def query_handler(self, inp):
        input_json = json.loads(inp.json)
        queries = input_json["queries"]
        lang_check = []
        suggestions = []
        for q in queries:
            is_en = detect(q) == "en"
            lang_check.append(is_en)
            if not is_en:
                suggestions.append(None)
                continue
            suggestion = self.get_suggestion(q)
            suggestions.append(suggestion)
        return suggestions, lang_check

    def index_handler(self, inp):
        input_json = json.loads(inp.json)
        documents = input_json["documents"]
        index_size = self.prepare_index(documents)
        return index_size

hlp = Helper()

@app.route('/ping')
def ping():
    if not model_is_ready:
        return jsonify(status="not ready")
    return jsonify(status="ok")

@app.route('/query', methods=['POST'])
def query():
    if not model_is_ready or not index_is_ready:
        return json.dumps({"status": "FAISS is not initialized!"})
    suggestions, lang_check = hlp.query_handler(request)

    return jsonify(suggestions=suggestions, lang_check=lang_check)

@app.route('/update_index', methods=['POST'])
def update_index():
    index_size = hlp.index_handler(request)

    return jsonify(status="ok", index_size=index_size)

hlp.prepare_model()

