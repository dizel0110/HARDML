import numpy as np
import pandas as pd


def calculate_linearized_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name, kappa=None
):
    """Вычисляет значение линеаризованной метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словать с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит в
        полуинтервал, а дата окончания нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики
    kappa - float, коэффициент в функции линеаризации.
        Если None, то посчитать как ratio метрику по имеющимся данным.

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
        .groupby(user_id_name)[[value_name]].agg(['sum', 'count'])
    )
    df_agg.columns = df_agg.columns.get_level_values(1)
    
    if kappa is None:
        kappa = df_agg['sum'].sum() / df_agg['count'].sum()
    df_agg[metric_name] = df_agg['sum'] - kappa * df_agg['count']
    
    df_res = pd.merge(
        df_user,
        df_agg[[metric_name]].reset_index(),
        on=user_id_name,
        how='outer'
    ).fillna(0)
    return df_res