import numpy as np
from typing import Iterable, Dict
from dataclasses import dataclass

@dataclass
class _Node(object):
    idxs: np.ndarray
    ate: float
    split_feat: int
    split_threshold: float

    def is_leaf(self):
        return self.split_feat is None 


class UpliftTreeRegressor(object):
    def __init__(self,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1000,
                 min_samples_leaf_treated: int = 300,
                 min_samples_leaf_control: int = 300):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.tree = [None] * (2 ** (self.max_depth + 1) - 1)

    @staticmethod
    def create_node(idxs: np.ndarray, 
                    treatment: np.ndarray,
                    y: np.ndarray) -> _Node:
        t_ = treatment[idxs]
        y_ = y[idxs]
        return _Node(idxs,
                     y_[t_ == 1].mean() - y_[t_ == 0].mean(),
                     None,
                     None)

    @staticmethod
    def get_thresholds(input_array: np.ndarray) -> np.ndarray:
        unique_values = np.unique(input_array)
        if len(unique_values) > 10:
            percentiles = np.percentile(input_array,
                                        [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        return np.unique(percentiles)

    def build_tree(self,
                   node: _Node,
                   ind: int,
                   X: np.ndarray, 
                   treatment: np.ndarray, 
                   y: np.ndarray) -> None:
        self.tree[ind] = node

        idxs = node.idxs
        n_samples, n_features = X[idxs].shape

        if np.log2(ind + 1) > self.max_depth:
            return

        features_and_thresholds = list()
        deltas = list()
        for i_feature in range(n_features):
            for threshold in self.get_thresholds(X[idxs, i_feature]):
                left_idxs = X[:, i_feature] <= threshold

                left_t = treatment[idxs & left_idxs]
                left_y = y[idxs & left_idxs]

                if (len(left_t) < self.min_samples_leaf
                    or (left_t == 1).sum() < self.min_samples_leaf_treated
                    or (left_t == 0).sum() < self.min_samples_leaf_control):
                    continue

                right_t = treatment[idxs & ~left_idxs]
                right_y = y[idxs & ~left_idxs]

                if (len(right_t) < self.min_samples_leaf
                    or (right_t == 1).sum() < self.min_samples_leaf_treated
                    or (right_t == 0).sum() < self.min_samples_leaf_control):
                    continue

                left_tau = ((left_y * left_t).sum() / left_t.sum()
                            - (left_y * (1 - left_t)).sum() / (1 - left_t).sum())
                right_tau = ((right_y * right_t).sum() / right_t.sum()
                             - (right_y * (1 - right_t)).sum() / (1 - right_t).sum())
                
                features_and_thresholds.append((i_feature, threshold))
                deltas.append(np.abs(left_tau - right_tau))
        
        if len(deltas) == 0:
            return

        deltas = np.array(deltas)

        i_feature, threshold = features_and_thresholds[deltas.argmax()]
        node.split_feat = i_feature
        node.split_threshold = threshold

        left_idxs = X[:, i_feature] <= threshold

        left_node = self.create_node(idxs & left_idxs, treatment, y)
        self.build_tree(left_node, 2 * ind + 1, X, treatment, y)

        right_node = self.create_node(idxs & ~left_idxs, treatment, y)
        self.build_tree(right_node, 2 * ind + 2, X, treatment, y)

    def fit(self,
            X: np.ndarray, 
            treatment: np.ndarray, 
            y: np.ndarray) -> None:        
        root_node = self.create_node([True] * X.shape[0], treatment, y)
        self.build_tree(root_node, 0, X, treatment, y)

    def build_paths(self, ind: int, mask: np.ndarray, out_paths: list, X: np.ndarray) -> None:
        node = self.tree[ind]
        if node.is_leaf():
            out_paths.append((mask, node.ate))
            return
        
        left_mask = X[:, node.split_feat] <= node.split_threshold
        self.build_paths(2 * ind + 1, mask & left_mask, out_paths, X)
        self.build_paths(2 * ind + 2, mask & ~left_mask, out_paths, X)

    def predict(self, X: np.ndarray) -> Iterable[float]:        
        paths = list()
        self.build_paths(0, [True] * len(X), paths, X)

        uplift = np.full(X.shape[0], np.nan)
        for mask, ate in paths:
            uplift[mask] = ate

        assert not np.any(np.isnan(uplift))
        return uplift
            
        
def _check(model_params: Dict[str, int],
           X: np.ndarray,
           treatment: np.ndarray,
           y: np.ndarray,
           uplift_true: np.ndarray) -> bool:
    assert X.shape[0] == treatment.shape[0]
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == uplift_true.shape[0]

    model = UpliftTreeRegressor(**model_params)
    model.fit(X, treatment, y)

    uplift_pred = np.array(model.predict(X)).reshape(len(X))

    assert uplift_pred.shape == uplift_true.shape
    assert not np.any(np.isnan(uplift_pred))
    return np.max(np.abs(uplift_pred - uplift_true)) < 1e-6


if __name__ == '__main__':
    model_params = {'max_depth': 3,
                    'min_samples_leaf': 6000, 
                    'min_samples_leaf_treated': 2500, 
                    'min_samples_leaf_control': 2500,}
    X = np.load('task2/example_X.npy')
    treatment = np.load('task2/example_treatment.npy')
    y = np.load('task2/example_y.npy')
    uplift_true = np.load('task2/example_preds.npy')

    print(f'X shape: {X.shape}', 
          f'treatment shape: {treatment.shape}',
          f'y shape: {y.shape}',
          f'uplift_true shape: {uplift_true.shape}',)

    print(_check(model_params, X, treatment, y, uplift_true))