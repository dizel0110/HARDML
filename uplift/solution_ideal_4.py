import dask.dataframe as dd
import pandas as pd
import datetime
import featurelib as fl
import category_encoders as ce
import sklearn.base as skbase


class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)
    
    def compute(self) -> dd.DataFrame:
        purchases = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (purchases['transaction_datetime'] >= date_from) & (purchases['transaction_datetime'] < date_to)

        purchases = purchases.loc[date_mask]
        purchases['day_of_week'] = purchases['transaction_datetime'].dt.weekday
        purchases = purchases.categorize(columns=['day_of_week'])
        features = purchases.pivot_table(
            index='client_id', columns='day_of_week', values='transaction_id', aggfunc='count'
        )

        DAYS_OF_WEEK = range(7)
        for day in DAYS_OF_WEEK:
            if day not in features.columns:
                features[day] = 0

        features = features.rename(columns={
            day: f'purchases_count_dw{day}__{self.delta}d' for day in features.columns
        }).reset_index()

        return features


class FavouriteStoreCalcer(fl.DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        receipts = receipts.loc[date_mask]

        client_store_n_receipts = (
            receipts
            .groupby(by=['client_id', 'store_id'])
            ['transaction_id'].count()
            .reset_index()
        )
        client_top_store_n_receipts = (
            client_store_n_receipts
            .groupby(by=['client_id'])
            ['transaction_id'].max()
            .reset_index()
        )
        features = (
            client_top_store_n_receipts
            .merge(
                client_store_n_receipts,
                on=['client_id', 'transaction_id']
            )
            .groupby(by=['client_id'])
            ['store_id'].max()
            .reset_index()
            .rename(columns={'store_id': f'favourite_store_id__{self.delta}d'})
        )

        return features


@fl.functional_transformer
def ExpressionTransformer(data: pd.DataFrame, expression: str, col_result: str) -> pd.DataFrame:
    data[col_result] = eval(expression.format(d='data'))
    return data


class LOOMeanTargetEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, col_categorical: str, col_target: str, col_result: str, **loo_params):
        self.col_categorical = col_categorical
        self.col_target = col_target
        self.col_result = col_result
        self.encoder_ = ce.LeaveOneOutEncoder(cols=[col_categorical], **(loo_params or {}))

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]
        self.encoder_.fit(data[self.col_categorical], y=y)
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]
        data[self.col_result] = self.encoder_.transform(data[self.col_categorical], y=y)
        return data
