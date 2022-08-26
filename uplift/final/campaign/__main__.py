import os
import argparse
import pathlib
from typing import Dict, List

import yaml
import pickle
import numpy as np
import pandas as pd
from dask import dataframe as dd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *

from .source import Engine
from .featurise import compute_features
from .estimators import build_pipeline
from .metrics import uplift_at_k


_engine = Engine()
_datapath = None
_metricspath = None
_artifactspath = None
_submitspath = None
_random_state = 110894


def _init(workpath: str) -> None:
    global _datapath
    _datapath = os.path.join(workpath, 'data')

    global _metricspath
    _metricspath = os.path.join(workpath, 'metrics')

    global _artifactspath
    _artifactspath = os.path.join(workpath, 'artifacts')

    global _submitspath
    _submitspath = os.path.join(workpath, 'submits')

    for path in [_artifactspath,
                 _metricspath,
                 _submitspath, ]:
        if not os.path.exists(path):
            os.mkdir(path)

    _engine.registerTable('campaigns', dd.read_csv(os.path.join(_datapath, 'campaigns.csv')))
    _engine.registerTable('customers', dd.read_csv(os.path.join(_datapath, 'customers.csv')))
    _engine.registerTable('receipts', dd.read_parquet(os.path.join(_datapath, 'receipts.parquet')))


def featurize(name: str, config: List[Dict]) -> None:
    features_dd = compute_features(config['calcers'], _engine)
    features_dd.to_parquet(os.path.join(_datapath, f'{name}_features.parquet'))


def train(name: str, config: List[Dict]) -> None:
    featurise_name = config['featurise']
    features_dd = dd.read_parquet(os.path.join(_datapath, f'{featurise_name}_features.parquet'))
    features_df = (features_dd
                   .sample(frac=config['sample_frac'],
                           random_state=_random_state)
                   .compute())

    X = features_df.loc[:, features_df.columns[3:]]
    w = features_df.loc[:, 'target_group_flag'].fillna(0)
    y = (28 * features_df.loc[:, 'target_purchase_amt'].fillna(0)
         - features_df.loc[:, 'target_discount_sum'].fillna(0)
         - 1 * features_df.loc[:, 'target_group_flag'].fillna(0))

    pipeline = Pipeline([('transform', build_pipeline(config['transformers'])),
                         ('select', build_pipeline(config['selectors'])),
                         ('model', build_pipeline([config['model']]))])

    (X_train, X_test,
     y_train, y_test,
     w_train, w_test, ) = train_test_split(X, y, w,
                                           test_size=config['validation']['test_size'],
                                           random_state=_random_state,
                                           stratify=w)

    sampler = ParameterSampler(config['search']['param_distributions'],
                               config['search']['n_iter'],
                               random_state=_random_state)
    spliter = StratifiedKFold(config['search']['cv'],
                              shuffle=True,
                              random_state=_random_state)
    results = list()
    for params in list(sampler):
        for train_idx, test_idx in spliter.split(X_train, w_train):
            iter_pipeline = clone(pipeline,)
            iter_pipeline.set_params(**params)

            _ = iter_pipeline.fit(X_train.iloc[train_idx, :],
                                y_train.iloc[train_idx],
                                model__w=w_train.iloc[train_idx])
            uplift = iter_pipeline.predict(X_train.iloc[test_idx, :])

            score = uplift_at_k(uplift,
                                w_train.iloc[test_idx].values,
                                y_train.iloc[test_idx].values,
                                0.3)
            results.append((score, iter_pipeline))
    
    best_oof_score, best_estimator = list(sorted(results, key=lambda x: x[0]))[-1]
    oos_score = uplift_at_k(best_estimator.predict(X_test),
                            w_test.values,
                            y_test.values,
                            0.3)
    print(best_oof_score, oos_score, best_estimator)

    with open(os.path.join(_artifactspath, f'{name}_pipeline.pkl'), 'wb') as f:
        pickle.dump(best_estimator, f)

    uplift = best_estimator.predict(X_test)
    hist, edges = np.histogram(uplift, bins=config['evaluation']['bin_count'])
    edges = np.around(edges, 2)
    hist_ss = pd.Series(hist,
                        index=pd.IntervalIndex.from_breaks(edges,
                                                           closed='left'))

    cutoff_step = config['evaluation']['cutoff_step']
    cutoffs = np.arange(cutoff_step, 1, cutoff_step, dtype=np.float16)

    n_bootstraps = config['evaluation']['n_bootstraps']
    bootstrap_size = int(len(X_test) * config['evaluation']['bootstrap_size'])
    np.random.seed(_random_state)
    uplifts = list()
    for _ in range(n_bootstraps):
        idx = np.random.choice(X_test.index, bootstrap_size, replace=True)
        uplifts.append(best_estimator.predict(X_test.loc[idx, :]))
    metrics = list()
    for k in cutoffs:
        values = list()
        for i in range(n_bootstraps):
            values.append(uplift_at_k(uplifts[i],
                                      w_test[idx].values,
                                      y_test[idx].values,
                                      k))
        metrics.append((k, np.mean(values), np.std(values)))
    metrics_df = pd.DataFrame.from_records(metrics, columns=['k', 'mean', 'std'])
    metrics_df = metrics_df.set_index('k')

    example_df = X.copy()
    example_df['sample'] = np.nan
    example_df.loc[X_train.index, 'sample'] = 'train'
    example_df.loc[X_test.index, 'sample'] = 'test'
    example_df['w'] = w
    example_df['y'] = y
    example_df['uplift'] = best_estimator.predict(X)

    hist_ss.to_csv(os.path.join(_metricspath, f'{name}_hist.csv'))
    metrics_df.to_csv(os.path.join(_metricspath, f'{name}_metrics.csv'))
    example_df.to_csv(os.path.join(_metricspath, f'{name}_examples.csv'))


def inference(name: str, config: List[Dict]) -> None:
    featurise_name = config['featurise']
    features_dd = dd.read_parquet(os.path.join(_datapath, f'{featurise_name}_features.parquet'))
    features_dd = features_dd[features_dd.columns[3:]]

    with open(os.path.join(_artifactspath, f'{name}_pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)

    features_df = features_dd.compute()
    features_df['uplift'] = pipeline.predict(features_df)
    features_df = features_df.loc[features_df['uplift'] > 0.0, :]
    features_df = features_df.sort_values('uplift', ascending=False)

    N = features_df.shape[0]
    for cutoff in config['cutoffs']:
        n = int(N * cutoff)
        customers = features_df.index[:n].to_series()
        
        customers.to_csv(os.path.join(_submitspath, f'{name}_{cutoff}_submit.csv'), index=False)


_tasks = {'featurise': featurize, 
          'train': train,
          'inference': inference, }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name', type=str, required=True)
    ap.add_argument('-t', '--task', type=str, required=True)
    ap.add_argument('-c', '--config', type=str, required=True)
    ap.add_argument('-d', '--work-dir', type=pathlib.Path, required=False, default='.')

    args = vars(ap.parse_args())

    _init(args['work_dir'])
    
    with open(os.path.join(args['work_dir'],
                           'configs',
                           args['task'],
                           args['config'] + '.yaml'), 'r') as f:
        config = yaml.load(f, yaml.Loader)
    
    _tasks[args['task']](args['name'], config)
