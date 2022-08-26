import os
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t as t_dist
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
users_df = pd.read_csv(os.environ['PATH_DF_USERS'])
sales_df = pd.read_csv(os.environ['PATH_DF_SALES'])

# эксперимент проводился с 49 до 55 день включительно
pilot_mask = sales_df['day'].isin(np.arange(49, 55))

sales_pilot_df = (sales_df
                  .loc[pilot_mask, :]
                  .groupby('user_id')
                  .agg(metric_pilot=('sales', 'mean'))
                  .reset_index())

metric_df = (users_df
             .merge(sales_pilot_df,
                    how='left',
                    on='user_id')
             .fillna(0))

fields = ['gender', ]
groups = [((0, ), 0.570202),
          ((1, ), 0.429798), ]
for i, (values, weight) in enumerate(groups):
    mask = np.ones(len(metric_df)).astype(bool)
    for field, value in zip(fields, values):
        mask &= metric_df[field] == value
    metric_df.loc[mask, 'group'] = i
metric_df['group'] = metric_df['group'].astype(int)


app = Flask(__name__)

@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']

    # has_effect = _check_test_default(test)
    # has_effect = _check_test_outliers(test)
    has_effect = _check_test_poststratify(test)

    return jsonify(has_effect=int(has_effect))

def _check_test_default(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    sales_a = metric_df[
        metric_df['user_id'].isin(user_a)
    ][
        'metric_pilot'
    ].values

    sales_b = metric_df[
        metric_df['user_id'].isin(user_b)
    ][
        'metric_pilot'
    ].values

    return ttest_ind(sales_a, sales_b)[1] < 0.05


def _check_test_outliers(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    sales_a = metric_df[
        metric_df['user_id'].isin(user_a)
    ][
        'metric_pilot'
    ].clip(upper=1090.844079).values

    sales_b = metric_df[
        metric_df['user_id'].isin(user_b)
    ][
        'metric_pilot'
    ].clip(upper=1090.844079).values

    return ttest_ind(sales_a, sales_b)[1] < 0.05


def _check_test_poststratify(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    weights = [w for _, w in groups]
    def calc_mean_var(data):
        stats_by_groups = (data
                           .groupby('group')
                           .agg(mean=('metric_pilot', 'mean'),
                                var=('metric_pilot', 'var')))
        stats_by_groups = stats_by_groups * weights
        return stats_by_groups['mean'].sum(), stats_by_groups['var'].sum() / len(data)
    
    mean_a, var_a = calc_mean_var(metric_df[metric_df['user_id'].isin(user_a)])
    mean_b, var_b = calc_mean_var(metric_df[metric_df['user_id'].isin(user_b)])

    n_a = len(user_a)
    n_b = len(user_b)

    vn_a = var_a / n_a
    vn_b = var_b / n_b

    df = (vn_a + vn_b)**2 / (vn_a**2 / (n_a - 1) + vn_b**2 / (n_b - 1))
    df = np.where(np.isnan(df), 1, df)

    denom = np.sqrt(vn_a + vn_b)
    
    t_val = (mean_a - mean_b) / denom

    raise t_dist.sf(-t_val, df) < 0.05


def _check_test_cuped(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    raise NotImplementedError


def _check_test_poststratify_cuped(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    raise NotImplementedError

