import numpy as np
import pandas as pd

from typing import List, Tuple, Any


class Node(object):
    def __init__(
        self,
        ate: float = None,
        n_items: int = None,
        split_feat: Any = None,
        split_feat_idx: int = None,
        split_threshold: float = None,
        left=None,
        right=None,
    ):
        self.ate = ate
        self.n_items = n_items
        self.split_feat = split_feat
        self.split_threshold = split_threshold
        self.left = left
        self.right = right


class UpliftTreeRegressor(object):

    def __init__(
        self,
        max_depth: int = 3,
        criterion: str = 'delta_delta_p',
        min_samples_leaf: int = 1000,
        min_samples_leaf_treated: int = 300,
        min_samples_leaf_control: int = 300
    ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def fit(self, X, treatment, y) -> None:
        '''Обучение uplift-дерева.'''
        data, self.feat_columns = self._prepare_data(X, treatment, y)
        self.n_features_ = len(self.feat_columns)
        
        self.tree_ = self._grow_tree(data)

    def _prepare_data(self, X, treatment, y) -> Tuple[pd.DataFrame, List]:
        '''Приведение данных к нужному "внутреннему" формату.'''
        if isinstance(X, pd.DataFrame):
            _values = X.values
            feat_columns = list(X.columns)
        else:
            _values = X
            feat_columns = [f'feat{idx}' for idx in range(X.shape[1])]
        data = pd.DataFrame(data=_values, columns=feat_columns)

        data['_treatment'] = treatment
        data['_y'] = y
        
        return data, feat_columns

    def _grow_tree(self, data: pd.DataFrame, depth: int = 0) -> Node:
        '''Метод для рекурсивного построения дерева.

        Args:
            data: таблица с признаками, флагом воздействия '_treatment' и целевой переменной '_y'
            depth: уровень глубины текущей вершины. Для корня всего дерева равна 0.

        Returns:
            Корень поддерева для полученной части данных (data).
        '''
        node = Node(
            ate=self._compute_ate(data),
            n_items=len(data)
        )
        
        if depth >= self.max_depth:
            return node
        
        best_feat, best_threshold = self._find_best_split(data)
        if best_feat is not None:
            data_left, data_right = self.divide_dataset(data, best_feat, best_threshold)
            node.split_feat = best_feat
            node.split_feat_idx = self.feat_columns.index(best_feat)
            node.split_threshold = best_threshold
            node.left = self._grow_tree(data_left, depth + 1)
            node.right = self._grow_tree(data_right, depth + 1)
        
        return node

    def _find_best_split(self, data: pd.DataFrame) -> Tuple[Any, float]:
        '''Метод для поиска наилучшего разбиения данных.

        Args:
            data: таблица с признаками, флагом воздействия '_treatment' и целевой переменной '_y'

        Returns:
            Название признака и порог разбиения.
        '''
        best_gain = 0.0
        best_feat, best_threshold = None, None

        for feat in self.feat_columns:
            
            column_values = data.loc[:, feat]
            unique_values = np.unique(column_values)
            if len(unique_values) > 10:
                percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])
            threshold_options = np.unique(percentiles)
            
            for threshold in threshold_options:
                data_left, data_right = self.divide_dataset(data, feat, threshold)

                # check the split validity on min_samples_leaf
                if (
                    len(data_left) < self.min_samples_leaf or len(data_right) < self.min_samples_leaf
                    or len(data_left.query('_treatment == 1')) < self.min_samples_leaf_treated
                    or len(data_left.query('_treatment == 0')) < self.min_samples_leaf_control
                    or len(data_right.query('_treatment == 1')) < self.min_samples_leaf_treated
                    or len(data_right.query('_treatment == 0')) < self.min_samples_leaf_control
                ):
                    continue

                gain = self._compute_gain(data_left, data_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_threshold = threshold
                    
        return best_feat, best_threshold
    
    @staticmethod
    def divide_dataset(data: pd.DataFrame, feat: Any, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Разбивает датасет на две части по признаку и порогу.'''
        mask = data.loc[:, feat] <= threshold
        return data[mask], data[~mask]
    
    def _compute_gain(self, data_left: pd.DataFrame, data_right: pd.DataFrame) -> float:
        '''Расчет DeltaDeltaP критерия по данным в левой и правой вершинах соответственно'''
        ate_left = self._compute_ate(data_left)
        ate_right = self._compute_ate(data_right)
        return abs(ate_left - ate_right)
    
    @staticmethod
    def _compute_ate(data: pd.DataFrame) -> float:
        '''Расчет ATE для целевой переменной.'''
        tmp = data.groupby('_treatment')['_y'].mean()
        ate = tmp.loc[1] - tmp.loc[0]
        return ate
    
    def predict(self, X) -> List[float]:
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs) -> float:
        node = self.tree_
        while node.left:
            if inputs[node.split_feat_idx] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.ate


def print_tree(node: Node, name: str = 'Root', indent: int = 0):
    '''Функция для отрисовки дерева.'''
    print(indent * '\t' + name + (' <leaf>' if node.left is None else ''))
    print(indent * '\t' + f'n_items: {node.n_items}')
    print(indent * '\t' + f'ATE: {node.ate}')
    print(indent * '\t' + f'split_feat: {node.split_feat}')
    print(indent * '\t' + f'split_threshold: {node.split_threshold}')
    
    print()
    
    if node.left is not None:
        print_tree(node.left, name='Left', indent=indent + 1)
    if node.right is not None:
        print_tree(node.right, name='Right', indent=indent + 1)
    