import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F


glue_qqp_dir = 'data/QQP'
glove_path = 'data/glove.6B.50d.txt'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        sigma_ = torch.FloatTensor([self.sigma])
        return torch.exp(-torch.square(torch.sub(x, self.mu)) / (2*torch.square(sigma_)))


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        
        start = -1 + 1 / (self.kernel_num - 1)
        step = 2 / (self.kernel_num - 1)
        for mu in np.arange(start, 1, step):
            kernels.append(GaussianKernel(mu, self.sigma))
        kernels.append(GaussianKernel(sigma=self.exact_sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        if len(self.out_layers) == 0:
            return torch.nn.Sequential(
                torch.nn.Linear(self.kernel_num, 1)
            )
        else:            
            sizes = [self.kernel_num]
            sizes += self.out_layers
            sizes += [1]
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
                layers.append(torch.nn.ReLU())
            return torch.nn.Sequential(*layers)
        

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        q_emb = self.embeddings(query)
        d_emb = self.embeddings(doc)
               
        return torch.einsum('ijk,imk->ijm',
                            F.normalize(q_emb, dim=-1),
                            F.normalize(d_emb, dim=-1))

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']                
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        idxs = [
            self.vocab[word]
            if self.vocab.get(word) is not None
            else self.vocab[self.oov_val]
            for word in tokenized_text
        ]
        return idxs[:self.max_len]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        text = self.idx_to_text_mapping[idx]
        tokens = self.preproc_func(text)
        return self._tokenized_text_to_index(tokens)

    def __getitem__(self, idx: int):
        raise NotImplementedError


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        query, doc1, doc2, score = self.index_pairs_or_triplets[idx]
        return (
            {
                'query': self._convert_text_idx_to_token_idxs(query),
                'document': self._convert_text_idx_to_token_idxs(doc1),
            },
            {
                'query': self._convert_text_idx_to_token_idxs(query),
                'document': self._convert_text_idx_to_token_idxs(doc2),
            },
            score,
        )


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        query, doc, score = self.index_pairs_or_triplets[idx]
        return (
            {
                'query': self._convert_text_idx_to_token_idxs(query),
                'document': self._convert_text_idx_to_token_idxs(doc),
            },
            score,
        )


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.train_triplets = self.sample_data_for_train_iter(self.glue_train_df, random_seed)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)
        
        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg, 
              self.idx_to_text_mapping_dev, 
              vocab=self.vocab, oov_val=self.vocab['OOV'], 
              preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0, 
            collate_fn=collate_fn, shuffle=False)
        
        self.train_dataset = TrainTripletsDataset(self.train_triplets, 
              self.idx_to_text_mapping_train, 
              vocab=self.vocab, oov_val=self.vocab['OOV'], 
              preproc_func=self.simple_preproc)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.dataloader_bs, num_workers=0, 
            collate_fn=collate_fn, shuffle=True)
    
    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        return inp_str.translate(table)

    def simple_preproc(self, inp_str: str) -> List[str]:
        words = nltk.word_tokenize(self.hadle_punctuation(inp_str).lower())
        return words
    
    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {w: n for w, n in vocab.items() if n >= min_occurancies}
    
    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        texts = []
        for df in list_of_df:
            texts += [df['text_left'], df['text_right']]
        unique_texts = pd.concat(texts).drop_duplicates()
        
        words = []
        for text in unique_texts.values:
            words += self.simple_preproc(text)
        counter = Counter(words)
        
        del words, unique_texts, texts
        
        counter = self._filter_rare_words(counter, min_occurancies)
        counter = list(counter.keys())
        return counter
            

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        embeddings = {}
        with open(file_path) as file:
            for line in file:
                items = line.split()
                embeddings[items[0]] = items[1:]
        return embeddings       
        
    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        np.random.seed(random_seed)
        embeddings = self._read_glove_embeddings(file_path)
        embedding_size = 50
        matrix = [
            [0.0] * embedding_size,
            [1.0] * embedding_size,
        ]
        vocab = {'PAD': 0, 'OOV': 1, }
        unk_words = [
            'PAD',
            'OOV',
        ]
        for idx, token in enumerate(inner_keys):
            if embeddings.get(token) is not None:
                matrix.append(np.array(embeddings[token], dtype=np.float64))
            else:
                vector = np.random.uniform(low=-self.emb_rand_uni_bound,
                                           high=self.emb_rand_uni_bound,
                                           size=embedding_size)
                matrix.append(vector)
                unk_words.append(token)
            vocab[token] = idx + 2
        assert len(matrix) == len(vocab)
        return np.array(matrix), vocab, unk_words
                
        

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    
    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int
                                   ) -> List[List[Union[str, float]]]:
        np.random.seed(seed)
        
        id_left_groups = inp_df.groupby('id_left')
        
        triplets = []
        for id_left, group in id_left_groups:
            if group['label'].nunique() > 1:
                for label in group['label'].unique():
                    ids_same_labels = group[group['label'] == label]['id_right'].values
                    ids_less_labels = group[group['label'] < label]['id_right'].values
                    
                    if len(ids_same_labels) > 1:
                        ids_select = np.random.choice(ids_same_labels,
                                                      size=2,
                                                      replace=False)
                        triplets.append([id_left,
                                         ids_select[0],
                                         ids_select[1],
                                         0.5])
                    elif len(ids_less_labels) != 0:
                        id_pos = np.random.choice(ids_same_labels,
                                                  size=1,
                                                  replace=False)
                        id_neg = np.random.choice(ids_less_labels,
                                                  size=1,
                                                  replace=False)
                        triplets.append([id_left,
                                         id_pos[0],
                                         id_neg[0],
                                         1])
        return triplets
        

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def dcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        order = ys_pred.argsort()[::-1][:ndcg_top_k]
        index = np.arange(len(order)) + 1
        return ((2**ys_true[order] - 1) / np.log2(index + 1)).sum()
    
    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        try:
            return self.dcg_k(ys_true, ys_pred, ndcg_top_k) / self.dcg_k(ys_true, ys_true, ndcg_top_k)
        except:
            return np.nan

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])
        
        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds
        
        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
            
        for epoch in range(n_epochs):
            for i, batch in enumerate(self.train_dataloader):
                inp_1, inp_2, y = batch
                
                
                preds = self.model(inp_1, inp_2)
                loss = criterion(preds, y)
                
                loss.backward()
                opt.step()
            
            with torch.no_grad():
                ndcg_val = self.valid(self.model,
                                      self.val_dataloader)
                if ndcg_val >= 0.925:
                    break
    
