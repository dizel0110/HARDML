from typing import List, Dict
from dask import dataframe as dd
from .base import FeatureCalcer
from ..source import Engine


_calcers = dict()


def registerCalcer(class_calcer) -> None:
    _calcers[class_calcer.name] = class_calcer


def _createCalcer(name: str, **kwargs) -> FeatureCalcer:
    return _calcers[name](**kwargs)


def _join_tables(tables: List[dd.DataFrame], how: str) -> dd.DataFrame:
    result = tables[0]
    for table in tables[1: ]:
        result = result.join(table, how=how)
    return result


def compute_features(config: List[Dict], engine: Engine) -> dd.DataFrame:
    calcers = list()
    keys = None

    for calcer_config in config:
        name, args = tuple(calcer_config.values())
        args['engine'] = engine

        calcer = _createCalcer(name, **args)
        calcers.append(calcer)

    compute_results = list()
    for calcer in calcers:
        compute_results.append(calcer.compute())

    features_dd = _join_tables(compute_results, how='outer')
    
    return features_dd
