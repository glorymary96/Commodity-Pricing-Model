from CommodityData import CommodityData
from CommodityModel import CommodityModel
from CommodityStatsModel import CommodityStatsModel
from PARAMS import commodities, start_date, end_date, models, LOG
from DailyCommodityForecaster import RobustDailyCommodityForecaster

import matplotlib.pyplot as plt
if __name__ == "__main__":

    plt.close('all')
    for commodity,ticker in commodities.items():
        LOG(commodity)
        experiment =  CommodityModel(
            dataset = CommodityData(
                commodity = commodity,
                commodity_mapping=commodities,
                start_date = start_date,
                end_date = end_date
            ),
            models = models
        )

        # experiment = RobustDailyCommodityForecaster(
        #     commodity = commodity,
        #     forecast_days= 60,
        #     dataset=CommodityData(
        #         commodity=commodity,
        #         commodity_mapping=commodities,
        #         start_date=start_date,
        #         end_date=end_date
        #     )
        # )