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
    period - dict, словарь с датами начала и конца периода пилота.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    filters - dict, словарь с фильтрами. Ключ - название поля, по которому фильтруем, значение - список значений,
        которые нужно оставить. Например, {'user_id': [111, 123, 943]}.
        Если None, то фильтровать не нужно.

    return - pd.DataFrame, в индексах все даты из указанного периода отсортированные по возрастанию, 
        столбцы - метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items'].
        Формат данных столбцов - float, формат данных индекса - datetime64[ns].
    """
    df[date_name] = pd.to_datetime(df[date_name])

    filter_ = ((df[date_name] >= pd.to_datetime(period['begin'])) 
               & (df[date_name] < pd.to_datetime(period['end'])))
    if filters is not None:
        for c, v in filters.items():
            filter_ &= df[c].isin(v)
    
    flt_df = df.loc[filter_, :]

    dates = pd.date_range(start=period['begin'], end=period['end'], freq='D')
    dates = dates[dates < period['end']]
    results_df = pd.DataFrame(index=dates)

    agg_df = (flt_df
              .groupby(date_name)
              .agg(revenue=(cost_name, 'sum'),
                   number_purchases=(sale_id_name, 'nunique'), ))

    sales_agg_df = (flt_df
                    .groupby([date_name, sale_id_name], as_index=False)
                    .agg(sum_check=(cost_name, 'sum'),
                         number_items=(cost_name, 'count'), )
                    .groupby(date_name)
                    .agg(average_check=('sum_check', 'mean'),
                         average_number_items=('number_items', 'mean'), ))

    agg_df = agg_df.join(sales_agg_df, how='outer').astype(float)
    return results_df.join(agg_df, how='outer').fillna(0.0)


if __name__ == '__main__':
    size = 100
    dates = ['2021-03-01', '2021-04-01', '2021-05-01', ]
    sales_ids = []
    np.random.seed(110894)
    df = pd.DataFrame({'cost': np.random.uniform(100, 200, size=size),
                       'date': np.random.choice(dates, size=size, replace=True),
                       'sale_id': np.random.choice(np.arange(1, 4), size=size, replace=True),
                       'shop_id': np.random.choice(np.arange(1, 3), size=size, replace=True), })
    print(df.sort_values(['date', 'sale_id', ]).head(10))

    print(calculate_sales_metrics(df,
                                  cost_name='cost',
                                  date_name='date',
                                  sale_id_name='sale_id',
                                  period={'begin': '2021-04-01', 'end': '2021-06-01'},
                                  filters=None, ))
