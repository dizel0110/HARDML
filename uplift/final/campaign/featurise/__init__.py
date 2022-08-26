from .calcers import *
from .compute import (registerCalcer,
                      compute_features,)

registerCalcer(PurchasesAggCalcer)
registerCalcer(DayOfWeekPurchasesCalcer)
registerCalcer(AgeLocationCalcer)
registerCalcer(CampaignCalcer)

