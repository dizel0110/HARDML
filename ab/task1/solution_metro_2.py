import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000, seed=None):
    """
    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    if seed is not None:
        np.random.seed(seed)
    pilot_values = df_pilot_group[metric_name].values
    control_values = df_control_group[metric_name].values
    len_pilot = len(df_pilot_group)
    len_control = len(df_control_group)
    mean_pilot_values = np.mean(pilot_values)
    std_pilot_values = np.std(pilot_values)

    effect_to_second_type_error = {}
    for effect in effects:
        pvalues = []
        for _ in range(n_iter):
            bs_pilot_values = np.random.choice(pilot_values, len_pilot, True)
            bs_pilot_values += np.random.normal(mean_pilot_values * (effect - 1), std_pilot_values / 10, len_pilot)
            bs_control_values = np.random.choice(control_values, len_control, True)
            _, pvalue = ttest_ind(bs_pilot_values, bs_control_values)
            pvalues.append(pvalue)
        second_type_error = np.mean(np.array(pvalues) > alpha)
        effect_to_second_type_error[effect] = second_type_error
    return effect_to_second_type_error