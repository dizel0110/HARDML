from inspect import ArgSpec
from typing import Dict, List
from sklearn.pipeline import Pipeline

from .transformers import *
from .selectors import *
from .models import *


_estimators = {}
for estimator in [LabelEncoder,
                  Imputer,
                  LOOMeanTargetEncoder,
                  DummySelector, ]:
    _estimators[estimator.name] = estimator
_estimators['random_forest'] = RandomForest
_estimators['meta_s'] = MetaSRegressor
_estimators['meta_t'] = MetaTRegressor
_estimators['meta_x'] = MetaXRegressor


def build_pipeline(config: List[Dict]) -> Pipeline:
    steps = list()
    for item in config:
        name, args = tuple(item.values())
        steps.append((name, _estimators[name](**args)))
    if len(steps) == 1:
        return steps[0][1]
    return Pipeline(steps)
