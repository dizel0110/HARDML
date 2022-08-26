import numpy as np
import pandas as pd


def calculate_sales_metrics(df, cost_name, date_name, sale_id_name, period, filters=None):
    """Вычисляет метрики по продажам.
    
    df - pd.DataFrame, датафрейм с данными. Пример
        pd.DataFrame(
            [[820, '2021-04-03', 1, 213]],
            columns=['cost', 'date', 'sale_id', 'shop_id']
        )
    cost_name - str, название столбца с стоимостью товара
    date_name - str, название столбца с датой покупки
    sale_id_name - str, название столбца с идентификатором покупки (в одной покупке может быть несколько товаров)
    period - dict, словать с датами начала и конца периода пилота и препилота.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    filters - dict, словарь с фильтрами. Ключ - название поля, по которому фильтруем, значение - список значений,
        которые нужно оставить. Например, {'user_id': [111, 123, 943]}.
        Если None, то фильтровать не нужно.

    return - pd.DataFrame, в индексах все даты из указанного периода отсортированные по возрастанию, 
        столбцы - метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items'].
    """
    period['begin'] = pd.to_datetime(period['begin'])
    period['end'] = pd.to_datetime(period['end'])
    df[date_name] = pd.to_datetime(df[date_name])

    mask = ((df[date_name] >= period['begin']) & (df[date_name] < period['end'])).values
    if filters:
        for column, values in filters.items():
            mask = mask & df[column].isin(values).values
    df_filtered = df.iloc[mask]

    dates = pd.date_range(start=period['begin'], end=period['end'], freq='D')
    dates = dates[dates < period['end']]
    df_dates = pd.DataFrame(index=dates)

    df_revenue = (
        df_filtered
        .groupby(date_name)[[cost_name]].sum()
        .rename(columns={cost_name: 'revenue'})
    )
    df_number_purchases = (
        df_filtered
        .groupby(date_name)[[sale_id_name]].nunique()
        .rename(columns={sale_id_name: 'number_purchases'})
    )
    df_average_check = (
        df_filtered
        .groupby([date_name, sale_id_name])[[cost_name]].sum()
        .reset_index()
        .groupby(date_name)[[cost_name]].mean()
        .rename(columns={cost_name: 'average_check'})
    )
    df_average_number_items = (
        df_filtered
        .groupby([date_name, sale_id_name])[[cost_name]].count()
        .reset_index()
        .groupby(date_name)[[cost_name]].mean()
        .rename(columns={cost_name: 'average_number_items'})
    )
    list_df = [df_revenue, df_number_purchases, df_average_check, df_average_number_items]
    df_res = df_dates.copy()
    for df_ in list_df:
        df_res = pd.merge(df_res, df_, how='outer', left_index=True, right_index=True)
    df_res.sort_index(inplace=True)
    df_res.fillna(0, inplace=True)
    return df_res
