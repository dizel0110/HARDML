import numpy as np


def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.
    
    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """
    z = 1.96
    p = np.mean(values)
    std = (p * (1 - p) / len(values)) ** 0.5
    ci = np.clip((p - z * std, p + z * std), 0, 1)
    return ci
