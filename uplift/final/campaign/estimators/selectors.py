from typing import List
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class DummySelector(BaseEstimator, SelectorMixin):
    name = 'dummy_selector'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fitted = False
        self.mask = None

    def fit(self,
            X: DataFrame,
            y: Series = None,
            **fit_params):
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')

        self.mask = (X.nunique() > 1).to_numpy()

        self._fitted = True
        return self
    
    def _get_support_mask(self):
        return self.mask

    def transform(self, 
                  X: DataFrame,
                  y: Series = None) -> DataFrame:
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')
        check_is_fitted(self, '_fitted')

        return X.loc[:, self._get_support_mask()]        
        