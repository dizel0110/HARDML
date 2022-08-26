import numpy as np
import pandas as pd


def calculate_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name
):
    """Вычисляет значение метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словать с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит в
        полуинтервал, а дата окончания нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    df_filtered = (
        df[
            df[user_id_name].isin(list_user_id)
            & (df[date_name] >= period['begin'])
            & (df[date_name] < period['end'])
        ]
        .copy()
    )
    df_user = pd.DataFrame({user_id_name: list_user_id})
    
    df_agg = (
        df_filtered
        .groupby(user_id_name)[[value_name]].sum()
        .rename(columns={value_name: metric_name})
        .reset_index()
    )
    df_res = pd.merge(df_user, df_agg, on=user_id_name, how='outer').fillna(0)
    return df_res


def calculate_metric_cuped(
    df, value_name, user_id_name, list_user_id, date_name, periods, metric_name
):
    """Вычисляет значение метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    periods - dict, словать с датами начала и конца периода пилота и препилота.
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
    df_prepilot = calculate_metric(
        df, value_name, user_id_name, list_user_id, date_name,
        periods['prepilot'], metric_name
    ).rename(columns={metric_name: f'{metric_name}_prepilot'})
    df_pilot = calculate_metric(
        df, value_name, user_id_name, list_user_id, date_name,
        periods['pilot'], metric_name
    )
    df = pd.merge(
        df_prepilot,
        df_pilot,
        on=user_id_name
    )
    # CUPED
    target_values = df[metric_name].values
    covariate_values = df[f'{metric_name}_prepilot'].values
    covariance = np.cov(target_values, covariate_values)[0, 1]
    variance = covariate_values.var()
    theta = covariance / variance
    df[f'{metric_name}_cuped'] = (
        target_values - theta * covariate_values
    )
    return df
