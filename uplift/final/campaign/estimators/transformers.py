from typing import List, Union

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from category_encoders import LeaveOneOutEncoder


class LabelEncoder(BaseEstimator, TransformerMixin):
    name = 'label_encoder'

    def __init__(self,
                 columns: List[str],
                 copy: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.copy = copy
        # self._fitted = False

    def fit(self,
            X: DataFrame,
            y: Series = None,
            **fit_params):
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')

        self.columns = {c: None for c in self.columns}

        for column in self.columns:
            values = X.loc[X[column].notna(), column].unique()

            self.columns[column] = {values[i]: i 
                                    for i in range(len(values))}
        
        self._fitted = True
        return self

    def transform(self, 
                  X: DataFrame,
                  y: Series = None) -> DataFrame:
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')
        check_is_fitted(self, '_fitted')

        if self.copy:
            X_transformed = X.copy()
        else:
            X_transformed = X

        for column in self.columns:
            X_transformed.loc[:, column] = X[column].map(self.columns[column])

        return X_transformed


class Imputer(BaseEstimator, TransformerMixin):
    name = 'imputer'

    def __init__(self,
                 columns: List[str],
                 fill_values: List[Union[int, float, str]],
                 copy: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.fill_values = fill_values
        self.copy = copy
        self._fitted = False

    def fit(self,
            X: DataFrame,
            y: Series = None,
            **fit_params):
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')

        if self.columns == 'all':
            self.columns = X.columns.tolist()
            self.fill_values = [self.fill_values] * len(self.columns)

        for i, column in enumerate(self.columns):
            fill_value = self.fill_values[i]

            if type(fill_value) == str:
                if fill_value == 'mean':
                    self.fill_values[i] = X[column].mean()
                elif fill_value == 'freq':
                    self.fill_values[i] = X[column].mode()
                else:
                    raise ValueError
        
        self._fitted = True
        return self

    def transform(self, 
                  X: DataFrame,
                  y: Series = None) -> DataFrame:
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')
        check_is_fitted(self, '_fitted')

        if self.copy:
            X_transformed = X.copy()
        else:
            X_transformed = X

        for i, column in enumerate(self.columns):
            X_transformed.loc[:, column] = X[column].fillna(self.fill_values[i])

        return X_transformed


class LOOMeanTargetEncoder(LeaveOneOutEncoder):
    name = 'loo_mean_target_encoder'

    def __init__(self,
                 cols: List[str],
                 drop_invariant=False,
                 handle_unknown='value',
                 handle_missing='value',
                 random_state=None,
                 sigma=None):
        super().__init__(verbose=0,
                         cols=cols,
                         drop_invariant=drop_invariant,
                         return_df=True,
                         handle_unknown=handle_unknown,
                         handle_missing=handle_missing,
                         random_state=random_state,
                         sigma=sigma)
