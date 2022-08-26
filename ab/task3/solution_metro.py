import numpy as np
import pandas as pd
from scipy import stats


def get_sample_size(mu, std, eff=1.01, alpha=0.05, beta=0.2):
    t_alpha = abs(stats.norm.ppf(alpha / 2, loc=0, scale=1))
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)

    mu_diff_squared = (mu - mu * eff) ** 2
    z_scores_sum_squared = (t_alpha + t_beta) ** 2
    disp_sum = 2 * (std ** 2)
    sample_size = int(
        np.ceil(
            z_scores_sum_squared * disp_sum / mu_diff_squared
        )
    )
    return sample_size


def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']    
    """
    metric_values = df[metric_name].values
    mu = metric_values.mean()
    std = metric_values.std()
    sample_sizes = [get_sample_size(mu, std, effect, alpha, beta) for effect in effects]
    res_df = pd.DataFrame({'effect': effects, 'sample_size': sample_sizes})
    return res_df