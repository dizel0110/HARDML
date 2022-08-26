import numpy as np
from sklearn.base import BaseEstimator
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.inference.meta import (BaseSRegressor,
                                     BaseTRegressor,
                                     BaseXRegressor, )
from lightgbm import LGBMRegressor


class RandomForest(UpliftRandomForestClassifier, BaseEstimator):
    def __init__(self, 
                 n_estimators=10,
                 max_features=10,
                 random_state=2019,
                 max_depth=5,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 evaluationFunction=None,
                 control_name='0',
                 n_jobs=-1,
                 **kwargs):
        super().__init__(n_estimators,
                         max_features,
                         random_state,
                         max_depth,
                         min_samples_leaf,
                         min_samples_treatment,
                         n_reg,
                         evaluationFunction,
                         control_name,
                         n_jobs,
                         **kwargs)

    def fit(self, X, y, **fit_params):
        w = np.where(fit_params['w'] == 1, '1', '0')
        return super().fit(X.values, w, y.values)

    def predict(self, X):
        return super().predict(X.values, full_output=False).reshape(-1)


class MetaSRegressor(BaseSRegressor, BaseEstimator):
    def __init__(self,
                 model=LGBMRegressor(),
                 ate_alpha=0.05,
                 control_name=0):
        super().__init__(learner=model,
                         ate_alpha=ate_alpha,
                         control_name=control_name)
    
    def fit(self, X, y, **fit_params):
        return super().fit(X.values,
                           fit_params['w'].values,
                           y.values)

    def predict(self, X):
        return super().predict(X,
                               treatment=None,
                               y=None,
                               p=None,
                               return_components=False,
                               verbose=False).reshape(-1)


class MetaTRegressor(BaseTRegressor, BaseEstimator):
    def __init__(self,
                 model_c=LGBMRegressor(),
                 model_t=LGBMRegressor(),
                 ate_alpha=0.05,
                 control_name=0):
        super().__init__(learner=None,
                         control_learner=model_c,
                         treatment_learner=model_t,
                         ate_alpha=ate_alpha,
                         control_name=control_name)
    
    def fit(self, X, y, **fit_params):
        return super().fit(X.values,
                           fit_params['w'].values,
                           y.values)

    def predict(self, X):
        return super().predict(X,
                               treatment=None,
                               y=None,
                               p=None,
                               return_components=False,
                               verbose=False).reshape(-1)


class MetaXRegressor(BaseXRegressor, BaseEstimator):
    def __init__(self,
                 model_mu_c=LGBMRegressor(),
                 model_mu_t=LGBMRegressor(),
                 model_tau_c=LGBMRegressor(),
                 model_tau_t=LGBMRegressor(),
                 ate_alpha=0.05,
                 control_name=0,
                 p=None):
        self.p = p
        super().__init__(learner=None,
                         control_outcome_learner=model_mu_c,
                         treatment_outcome_learner=model_mu_t,
                         control_effect_learner=model_tau_c,
                         treatment_effect_learner=model_tau_t,
                         ate_alpha=ate_alpha,
                         control_name=control_name)
    
    def fit(self, X, y, **fit_params):
        p = self.p
        if type(p) in [int, float]:
            p = np.full(X.shape[0], p)
        return super().fit(X.values,
                           fit_params['w'].values,
                           y.values,
                           p)

    def predict(self, X):
        p = self.p
        if type(p) in [int, float]:
            p = np.full(X.shape[0], p)
        return super().predict(X,
                               treatment=None,
                               y=None,
                               p=p,
                               return_components=False,
                               verbose=False).reshape(-1)
