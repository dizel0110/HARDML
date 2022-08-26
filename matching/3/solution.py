import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        (
            X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test,
        ) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for id_ in np.unique(inp_query_ids):
            mask = inp_query_ids == id_
            inp_feat_array[mask, :] = StandardScaler().fit_transform(inp_feat_array[mask])
        return inp_feat_array

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:        
        epochs = 5
        ndcgs = []
        for epoch in range(epochs):
            self._train_one_epoch()
            ndcgs.append(self._eval_test_set())
        return ndcgs

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_true = torch.softmax(batch_ys, dim=0)
        P_pred = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_true * torch.log(P_pred))

    def _train_one_epoch(self) -> None:
        self.model.train()
        
        queries = np.unique(self.query_ids_train)
        N_train = len(queries)
        batch_size = 20
        
        idx = torch.randperm(N_train)
        
        queries = queries[idx]
        
        cur_batch = 0
        for it in range(N_train // batch_size):
            batch_queries = queries[cur_batch:cur_batch+batch_size]
            for query in batch_queries:
                
                query_X = self.X_train[self.query_ids_train == query]
                query_ys = self.ys_train[self.query_ids_train == query]
                
                self.optimizer.zero_grad()
                if len(query_X) > 0:
                    query_pred = self.model(query_X).reshape(-1)
                    loss = self._calc_loss(query_ys, query_pred)
#                     if it % 10:
#                         print(f"Loss: {loss.item():.2f}")
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    
#             if it % 10 == 0:
#                 print(f"nDCG: {self._eval_test_set():.2f}")
            cur_batch += batch_size

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for query in np.unique(self.query_ids_test):
                try:
                    query_X = self.X_test[self.query_ids_test == query]
                    query_y = self.ys_test[self.query_ids_test == query]
                    query_pred = self.model(query_X).reshape(-1)
                    ndcgs.append(self._ndcg_k(query_y, query_pred, self.ndcg_top_k))
                except Exception as exp:
#                     print('Error in dcg: ', exp)
                    ndcgs.append(0.0)
            return np.mean(ndcgs)

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
               ndcg_top_k: int) -> float:
        order = ys_pred.argsort(descending=True)[:ndcg_top_k]
        index = torch.arange(len(order), dtype=torch.float64) + 1
        return ((2**ys_true[order] - 1) / torch.log2(index + 1)).sum().item()
        
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        
        dcg_val = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        dcg_best_val = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        return dcg_val / dcg_best_val
