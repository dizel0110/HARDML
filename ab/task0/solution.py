import numpy as np


def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.

    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """
    mu = values.mean()
    se = np.sqrt(mu * (1 - mu) / len(values))
    z_alpha = 1.96

    return np.clip((mu - z_alpha * se, mu + z_alpha * se), 0, 1)

if __name__ == '__main__':
    import sys
    from scipy.stats import bernoulli

    p, size = sys.argv[1:]
    p, size = float(p), int(size)

    values = bernoulli(p=p).rvs(size=size)
    print(get_bernoulli_confidence_interval(values))
    
