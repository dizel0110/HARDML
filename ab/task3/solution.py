import numpy as np
import pandas as pd
from scipy.stats import norm


def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']    
    """
    mu = df[metric_name].values.mean()
    se = df[metric_name].values.std()

    results = list()
    for effect in effects:
        effect_abs = (mu - mu * effect) ** 2
        se_sum = 2 * (se**2)

        F_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
        F_beta = norm.ppf(1 - beta, loc=0, scale=1)
        F_square = (F_alpha + F_beta)**2

        n = int(np.ceil(F_square * se_sum / effect_abs))
        results.append((effect, n))
    
    return pd.DataFrame.from_records(results,
                                     columns=['effect', 'sample_size'])


if __name__ == '__main__':
    np.random.seed(110894)

    size = 10_000
    df = pd.DataFrame({'metric': np.random.normal(loc=10, scale=5, size=size)})
    effects = list(np.arange(1.01, 1.21, 0.01))

    print(df['metric'].mean(), df['metric'].std())
    print(estimate_sample_size(df, 'metric', effects))
