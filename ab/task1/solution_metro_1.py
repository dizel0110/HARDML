import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """
    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.

    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int, состояние генератора случайных чисел

    return - float, ошибка первого рода
    """
    if seed is not None:
        np.random.seed(seed)
    pilot_values = df_pilot_group[metric_name].values
    control_values = df_control_group[metric_name].values
    len_pilot = len(df_pilot_group)
    len_control = len(df_control_group)

    pvalues = []
    for _ in range(n_iter):
        bs_pilot_values = np.random.choice(pilot_values, len_pilot, True)
        bs_control_values = np.random.choice(control_values, len_control, True)
        _, pvalue = ttest_ind(bs_pilot_values, bs_control_values)
        pvalues.append(pvalue)
    first_type_error = np.mean(np.array(pvalues) < alpha)
    return first_type_error
