import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        self.trees = [
            DecisionTreeRegressor(
                max_depth=11,
                min_samples_leaf=75,
                random_state=i,
            ) 
            for i in np.arange(n_estimators)
        ]
        self.features_ids = []
        self.n_trees_used = n_estimators

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train).reshape(-1,1)
        self.ys_test = torch.FloatTensor(y_test).reshape(-1,1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for id_ in np.unique(inp_query_ids):
            mask = inp_query_ids == id_
            inp_feat_array[mask, :] = StandardScaler().fit_transform(inp_feat_array[mask])
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        np.random.seed(cur_tree_idx)
        
        lambdas = torch.zeros_like(train_preds)
        for query in np.unique(self.query_ids_train):
            mask = self.query_ids_train == query
            if self.ys_train[mask].sum() != 0:
                lambdas[mask] = self._compute_lambdas(self.ys_train[mask],
                                                      train_preds[mask])
        
        N_samples = self.X_train.shape[0]
        N_features = self.X_train.shape[1]
        samples_idx = np.random.permutation(N_samples)[:int(N_samples*self.subsample)]
        feaures_idx = np.random.permutation(N_features)[:int(N_features*self.colsample_bytree)]
        
        X_train = self.X_train[samples_idx][:,feaures_idx]
        y_train = -lambdas[samples_idx]
        
        tree = self.trees[cur_tree_idx]
        tree.fit(X_train, y_train)
        return tree, feaures_idx
        

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndcgs = []
        for query in np.unique(queries_list):
            mask = queries_list == query
            try:
                ndcgs.append(self._ndcg_k(true_labels[mask], preds[mask], self.ndcg_top_k))
            except Exception as ex:
                print(query, ex)
                ndcgs.append(0.0)
        return np.mean(ndcgs)


    def fit(self):
        np.random.seed(0)
        
        self.best_ndcg = 0.0
        self.ndcgs = []
        train_preds = torch.zeros_like(self.ys_train)
        test_preds = torch.zeros_like(self.ys_test)
        for ind in np.arange(self.n_estimators):
            tree, features_idx = self._train_one_tree(ind, train_preds)
            self.features_ids.append(features_idx)
                        
            train_preds += self.lr * tree.predict(self.X_train[:, features_idx]).reshape(-1, 1)
            test_preds += self.lr * tree.predict(self.X_test[:, features_idx]).reshape(-1, 1)
            
            self.ndcgs.append(self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds))
            if self.ndcgs[-1] > self.best_ndcg:
                self.best_ndcg = self.ndcgs[-1]
        
        self.n_trees_used = np.argmax(self.ndcgs) + 1
        self.trees = self.trees[:self.n_trees_used]
        self.features_ids = self.features_ids[:self.n_trees_used]
            
    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        pred = torch.zeros(data.shape[0], 1)
        for tree, features_idx in zip(self.trees, self.features_ids):
            pred += self.lr * tree.predict(data[:, features_idx]).reshape(-1, 1)
        return pred
            

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        N = 1 / self._dcg_k(y_true, y_true, self.ndcg_top_k)

        _, order = torch.sort(y_true, descending=True, axis=0)
        order += 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            rel_diff = y_true - y_true.t()
            pos_pairs = (rel_diff > 0).type(torch.float32)
            neg_pairs = (rel_diff < 0).type(torch.float32)
            Sij = pos_pairs - neg_pairs
            
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())

            decay_diff = (1.0 / torch.log2(order + 1.0)) - (1.0 / torch.log2(order.t() + 1.0))
            
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)

            lambdas =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambdas = torch.sum(lambdas, dim=1, keepdim=True)

            return lambdas

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
               ndcg_top_k: int) -> float:
        order = ys_pred.argsort(dim=0, descending=True)[:ndcg_top_k]
        order = order.reshape(-1)
        index = torch.arange(len(order), dtype=torch.float64).reshape(-1, 1)
        index += 1
        return ((torch.pow(2, ys_true[order]) - 1) / torch.log2(index + 1)).sum().item()
    
    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        dcg_val = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        dcg_best_val = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        return dcg_val / dcg_best_val

    def save_model(self, path: str):
        state = {
            'trees': self.trees,
            'features_ids': self.features_ids,
            'n_trees_used': self.n_trees_used,
        }
        with open(path, 'wb') as file:
            pickle.dump(state, file)

    def load_model(self, path: str):
        with open(path, 'rb') as file:
            state = pickle.load(file)
        self.trees = state['trees']
        self.features_ids = state['features_ids']
        self.n_trees_used = state['n_trees_used']
