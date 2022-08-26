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
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит в
        полуинтервал, а дата окончания нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики
    kappa - float, коэффициент в функции линеаризации.
        Если None, то посчитать как ratio метрику по имеющимся данным.

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    mask = (df[user_id_name].isin(list_user_id)
            & (df[date_name] >= period['begin'])
            & (df[date_name] < period['end']))
    flt_df = df.loc[mask, :]

    x = flt_df.groupby(user_id_name)[value_name].sum()
    y = flt_df.groupby(user_id_name).size()

    if kappa is None:
        kappa = x.sum() / y.sum()

    metric_lin = x - kappa * y
    metric_lin.name = metric_name

    metric_df = pd.DataFrame({user_id_name: list_user_id})
    metric_df = (metric_df
                 .merge(metric_lin,
                        how='outer',
                        left_on=user_id_name,
                        right_index=True, )
                 .fillna(0))
    return metric_df


if __name__ == '__main__':
    np.random.seed(110894)

    size = 100_000
    dates = pd.date_range(start='2021-10-01', periods=30, freq='D')
    user_ids = [f'{i}'*2 for i in np.arange(100)]

    df = pd.DataFrame({'date': np.random.choice(dates, size, True),
                       'user': np.random.choice(user_ids, size, True),
                       'value': np.random.normal(100, 1, size), })
    print(df.head())

    print(calculate_linearized_metric(df, 'value', 'user', user_ids, 'date',
                                      period={'begin': '2021-10-01', 'end': '2021-10-16'},
                                      metric_name='metric_lin',))
