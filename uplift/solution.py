import datetime
import dask.dataframe as dd
import pandas as pd
import numpy as np
try:
    import data_config as cfg
except ImportError:
    pass
import featurelib as fl
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import LeaveOneOutEncoder


class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self,
                 engine: fl.Engine, 
                 date_to: datetime.date,
                 delta: int,
                 **kwargs):
        super().__init__(engine=engine,
                         date_to=date_to,
                         **kwargs)

        self.engine = engine
        self.delta = delta

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.get_table('receipts')
        date_to = pd.to_datetime(self.date_to)
        date_from = pd.to_datetime(self.date_to - datetime.timedelta(days=self.delta))

        mask = ((source_dd['transaction_datetime'] < date_to)
                & (source_dd['transaction_datetime'] >= date_from))
        source_flt_dd = source_dd[mask]

        table_tmp_dd = source_flt_dd.copy()
        table_tmp_dd['day_of_week'] = (source_flt_dd['transaction_datetime']
                                       .dt.dayofweek
                                       .astype('category')
                                       .cat
                                       .as_known())

        features_dd = table_tmp_dd.pivot_table(index='client_id',
                                               columns='day_of_week',
                                               values='transaction_id',
                                               aggfunc='count')
        features_dd.columns = [f'purchases_count_dw{f}__{self.delta}d' 
                               for f in features_dd.columns]
        features_dd = features_dd.reset_index()

        return features_dd


class FavouriteStoreCalcer(fl.DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self,
                 engine: fl.Engine, 
                 date_to: datetime.date,
                 delta: int,
                 **kwargs):
        super().__init__(engine=engine,
                         date_to=date_to,
                         **kwargs)

        self.engine = engine
        self.delta = delta

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.get_table('receipts')
        date_to = pd.to_datetime(self.date_to)
        date_from = pd.to_datetime(self.date_to - datetime.timedelta(days=self.delta))

        mask = ((source_dd['transaction_datetime'] < date_to)
                & (source_dd['transaction_datetime'] >= date_from))
        source_flt_dd = source_dd[mask]

        def get_mode(x):
            values, counts = x.values[0]
            argmax = counts.argmax()
            if (counts == counts[argmax]).sum() > 1:
                return values[counts == counts[argmax]].max()
            else:
                return values[counts.argmax()]

        mode = dd.Aggregation('mode',
                              lambda x: x.apply(lambda y: np.unique(y, return_counts=True)),
                              lambda x: x.apply(lambda y: get_mode(y)),)
        features_dd = source_flt_dd.groupby('client_id').agg({'store_id': mode})
        features_dd.columns = [f'favourite_store_id__{self.delta}d']
        features_dd = features_dd.reset_index()

        return features_dd


class AgeGenderCalcer(fl.FeatureCalcer):
    name = 'age_gender'
    keys = ['client_id']

    def __init__(self, engine: fl.Engine):
        super().__init__(engine)

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.get_table('client_profile')
        return source_dd[['client_id', 'age', 'gender']]


class TargetFromCampaignsCalcer(fl.DateFeatureCalcer):
    name = 'target_from_campaigns'
    keys = ['client_id']

    def __init__(self, engine: fl.Engine, date_to: datetime.date, **kwargs):
        super().__init__(engine=engine,
                         date_to=date_to, **kwargs)

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.get_table('campaigns')

        date_to = pd.to_datetime(self.date_to)
        source_flt_dd = source_dd[source_dd['treatment_date'].astype('M8[us]') <= date_to]

        return source_flt_dd[['client_id', 
                              'treatment_flg',
                              'target_purchases_sum',
                              'target_purchases_count',
                              'target_campaign_points_spent',]]


class ExpressionTransformer(BaseEstimator, TransformerMixin):
    name = 'expression'

    def __init__(self, expression: str, col_result: str):
        super().__init__()

        self.expression = expression
        self.col_result = col_result

    def fit(self, *args, **kwargs):
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        data[self.col_result] = eval(self.expression.format(d='data'))
        return data


class LOOMeanTargetEncoder(BaseEstimator, TransformerMixin):
    name = 'loo_mean_target_encoder'

    def __init__(self, 
                 col_categorical: str,
                 col_target: str,
                 col_result: str):
        super().__init__()
        self.col_categorical = col_categorical
        self.col_target = col_target
        self.col_result = col_result

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        X = data[[self.col_categorical]]
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]

        self.encoder = LeaveOneOutEncoder()
        self.encoder = self.encoder.fit(X, y)
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        X = data[[self.col_categorical]]
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]
        
        data[self.col_result] = self.encoder.transform(X, y)[self.col_categorical]
        return data


if __name__ == '__main__':
    engine = fl.Engine(tables={})
    engine.register_table(table=dd.read_parquet('task4/data/purchases.parquet'),
                          name='purchases')
    engine.register_table(table=dd.read_parquet('task4/data/receipts.parquet'),
                          name='receipts')
    engine.register_table(table=dd.read_csv('task4/data/products.csv'),
                          name='products')
    engine.register_table(table=dd.read_csv('task4/data/client_profile.csv'),
                          name='client_profile')
    engine.register_table(table=dd.read_csv('task4/data/campaigns.csv'),
                          name='campaigns')

    test_df = pd.read_csv('task4/dataset_mini.csv')
    
    # test calcers
    calcers = {item.name: item 
               for item in [DayOfWeekReceiptsCalcer,
                            FavouriteStoreCalcer,]}
    print('Tets calcers')
    print('-' * 100)
    for calcer_cfg in cfg.data_config['calcers']:
        name, args = calcer_cfg.values()
        if calcers.get(name) is None:
            continue
        calcer = calcers[name](engine, **args)
        result_df = calcer.compute().compute()

        assert np.allclose(result_df.values,
                           test_df[result_df.columns].values,
                           atol=1e-5)
        
        print(f'Result of {name} calcer \nwith params {args}')
        print('-' * 100)
        print(result_df.head())
        print('-' * 100)

    #test trasformers
    fl.register_calcer(DayOfWeekReceiptsCalcer)
    fl.register_calcer(FavouriteStoreCalcer)
    fl.register_calcer(AgeGenderCalcer)
    fl.register_calcer(TargetFromCampaignsCalcer)
    features_df = fl.compute_features(engine, cfg.data_config['calcers']).compute()
    
    transformers = {item.name: item 
                    for item in [ExpressionTransformer, 
                                 LOOMeanTargetEncoder,]}
    print('Tets transformers')
    print('-' * 100)
    for transformer_cfg in cfg.data_config['transforms']:
        name, args = transformer_cfg.values()
        if transformers.get(name) is None:
            continue
        transformer = transformers[name](**args)
        result_df = transformer.fit_transform(features_df)

        assert np.allclose(result_df[transformer.col_result].values,
                           test_df[transformer.col_result].values,
                           atol=1e-5)
        
        print(f'Result of {name} transformer \nwith params {args}')
        print('-' * 100)
        print(result_df.head())
        print('-' * 100)

