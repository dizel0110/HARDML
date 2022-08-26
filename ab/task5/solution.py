import numpy as np
import pandas as pd
from pandas.core.indexes import period


def calculate_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name
):
    """Вычисляет значение метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит нужный
        полуинтервал, а дата окончание нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    mask = (df[user_id_name].isin(list_user_id)
            & (df[date_name] >= period['begin']) 
            & (df[date_name] < period['end']))

    metrics_df = (df
                  .loc[mask, :]
                  .groupby(user_id_name)[[value_name]].sum()
                  .rename(columns={value_name: metric_name})
                  .reset_index())
    
    metrics_df = (pd.DataFrame({user_id_name: list_user_id})
                  .merge(metrics_df, how='outer', on=user_id_name)
                  .fillna(0))
    return metrics_df


def calculate_metric_cuped(
    df, value_name, user_id_name, list_user_id, date_name, periods, metric_name
):
    """Вычисляет метрики во время пилота, коварианту и преобразованную метрику cuped.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    periods - dict, словарь с датами начала и конца периода пилота и препилота.
        Пример, {
            'prepilot': {'begin': '2020-01-01', 'end': '2020-01-08'},
            'pilot': {'begin': '2020-01-08', 'end': '2020-01-15'}
        }.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами
        [user_id_name, metric_name, f'{metric_name}_prepilot', f'{metric_name}_cuped'],
        кол-во строк должно быть равно кол-ву элементов в списке list_user_id.
    """
    prepilot_metric_df = calculate_metric(df, value_name, user_id_name,
                                          list_user_id, date_name,
                                          periods['prepilot'], metric_name)
    pilot_metric_df = calculate_metric(df, value_name, user_id_name,
                                       list_user_id, date_name,
                                       periods['pilot'], metric_name)

    full_df = (pilot_metric_df
               .merge(prepilot_metric_df,
                      how='inner',
                      on=user_id_name,
                      suffixes=('', '_prepilot'),
                      validate='1:1'))

    y = full_df[metric_name].values
    y_cov = full_df[f'{metric_name}_prepilot'].values
    covariance = np.cov(y_cov, y)[0, 1]
    variance = y_cov.var()
    theta = covariance / variance

    full_df[f'{metric_name}_cuped'] = y - theta * y_cov
    return full_df


if __name__ == '__main__':
    np.random.seed(110894)

    size = 100_000
    dates = pd.date_range(start='2021-10-01', periods=30, freq='D')
    user_ids = np.arange(10)

    df = pd.DataFrame({'date': np.random.choice(dates, size, True),
                       'user': np.random.choice(user_ids, size, True),
                       'value': np.random.normal(100, 1, size), })
    print(df.head())

    print(calculate_metric(df, 'value', 'user', user_ids, 'date',
                           period={'begin': '2021-10-01', 'end': '2021-10-15'},
                           metric_name='total', ))
    df_cuped = calculate_metric_cuped(df, 'value', 'user', [0, 1], 'date',
                                      periods={'prepilot': {'begin': '2021-10-01', 'end': '2021-10-15'},
                                               'pilot': {'begin': '2021-10-15', 'end': '2021-10-30'},}, 
                                      metric_name='total', )
    print(df_cuped.head())
    print(df_cuped.var())
