import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    np.random.seed(seed)

    p_values = list()
    for _ in range(n_iter):
        pilot = np.random.choice(df_pilot_group[metric_name], 
                                 size=len(df_pilot_group),
                                 replace=True)
        control = np.random.choice(df_control_group[metric_name],
                                   size=len(df_control_group), 
                                   replace=True)

        _, p_value = ttest_ind(pilot,
                               control,
                               equal_var=False, )
        p_values.append(p_value)
    p_values = np.array(p_values)

    return (p_values < alpha).sum() / n_iter


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    np.random.seed(seed)

    results = dict()
    for effect in effects:

        p_values = list()
        for _ in range(n_iter):
            pilot = np.random.choice(df_pilot_group[metric_name], 
                                    size=len(df_pilot_group),
                                    replace=True)
            pilot = pilot * effect
            control = np.random.choice(df_control_group[metric_name],
                                       size=len(df_control_group), 
                                       replace=True)

            _, p_value = ttest_ind(pilot,
                                   control,
                                   equal_var=False, )
            p_values.append(p_value)
        
        p_values = np.array(p_values)

        results[effect] = (p_values >= alpha).sum() / n_iter
    
    return results


if __name__ == '__main__':
    metric_name = 'metric'
    effects = np.arange(5, 25, 1)
    n_iter = 10_000
    df_pilot_group = pd.DataFrame({metric_name: np.random.normal(loc=0.0, scale=0.1, size=1_000)})
    df_control_group = pd.DataFrame({metric_name: np.random.normal(loc=0.0, scale=0.2, size=2_000)})

    print(estimate_first_type_error(df_pilot_group,
                                    df_control_group,
                                    metric_name,
                                    n_iter=n_iter, ))
    print(estimate_second_type_error(df_pilot_group,
                                     df_control_group,
                                     metric_name, 
                                     effects,
                                     n_iter=n_iter, ))