from typing import List

from pandas.api.types import CategoricalDtype
from dask import dataframe as dd

from .base import FeatureCalcer, DateFeatureCalcer
from ..source import Engine


class PurchasesAggCalcer(DateFeatureCalcer):
    name = 'purchases_agg'
    keys = ['customer_id']

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.getTable('receipts')
        source_flt_dd = self.flt(source_dd)

        features_dd = (source_flt_dd
                       .groupby(self.keys)
                       .agg({'date': 'count',
                             'purchase_sum': 'sum',}))
        features_dd.columns = [f'purchases_{self.delta}d__count',
                               f'purchases_sum_{self.delta}d__sum',]
        return features_dd


class DayOfWeekPurchasesCalcer(DateFeatureCalcer):
    name = 'day_of_week_purchases'
    keys = ['customer_id']

    def __init__(self,
                 engine: Engine,
                 col_date: str,
                 date_to: int,
                 delta: int,
                 weekends: List[int] = None,):
        super().__init__(engine,
                         col_date,
                         date_to, delta)
        self.weekends = weekends

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.getTable('receipts')
        source_flt_dd = self.flt(source_dd)

        source_flt_dd['day_of_week'] = ((source_flt_dd['date'] % 7)
                                        .astype(CategoricalDtype(list(range(7)))))
        features_dd = source_flt_dd.pivot_table(index='customer_id',
                                                columns='day_of_week',
                                                values='date',
                                                aggfunc='count')
        features_dd.columns = [f'purchases_wd{f}_{self.delta}d__count' 
                               for f in features_dd.columns]

        if not self.weekends is None or len(self.weekends) == 0:
            col_weekends = [f'purchases_wd{wd}_{self.delta}d__count' 
                            for wd in self.weekends]
            features_dd[f'weekend_purchases_{self.delta}d__ratio'] = (features_dd[col_weekends].sum(axis=1) 
                                                                      / features_dd.sum(axis=1))

        return features_dd


class AgeLocationCalcer(FeatureCalcer):
    name = 'age_location'
    keys = ['customer_id']

    def compute(self) -> dd.DataFrame:
        source_dd = self.engine.getTable('customers')
        source_dd = source_dd.set_index(self.keys)

        return source_dd[['age', 'location']]


class CampaignCalcer(FeatureCalcer):
    name = 'campaign'
    keys = ['customer_id']

    def __init__(self,
                 engine: Engine,
                 date_start_offer: int,
                 days_offer: int,
                 **kwargs):
        super().__init__(engine, **kwargs)
        self.date_start_offer = date_start_offer
        self.days_offer = days_offer

    def compute(self) -> dd.DataFrame:
        customers_dd = self.engine.getTable('customers')
        campaigns_dd = self.engine.getTable('campaigns')
        receipts_dd = self.engine.getTable('receipts')

        mask_offer = ((receipts_dd['date'] >= self.date_start_offer)
                      & (receipts_dd['date'] < (self.date_start_offer + self.days_offer)))
        receipts_flt_dd = receipts_dd[mask_offer]

        receipts_agg_dd = (receipts_flt_dd
                           .groupby(self.keys)
                           .agg({'purchase_amt': 'sum',
                                 'discount': 'sum',}))
        receipts_agg_dd.columns = ['target_purchase_amt',
                                   'target_discount_sum',]
        receipts_agg_dd = receipts_agg_dd.reset_index()
        
        features_dd = (customers_dd[['customer_id']]
                       .merge(campaigns_dd[['customer_id', 'target_group_flag']], how='left', on='customer_id')
                       .merge(receipts_agg_dd, how='left', on='customer_id'))
        features_dd = features_dd.fillna(0)
        features_dd = features_dd.set_index('customer_id')
        return features_dd    

