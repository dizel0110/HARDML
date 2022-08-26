import numpy as np
import pandas as pd


def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    """Подбирает стратифицированные группы для эксперимента.

    data - pd.DataFrame, датафрейм с описанием объектов, содержит атрибуты для стратификации.
    strat_columns - List[str], список названий столбцов, по которым нужно стратифицировать.
    group_size - int, размеры групп.
    weights - dict, словарь весов страт {strat: weight}, где strat - tuple значений элементов страт,
        например, для strat_columns=['os', 'gender', 'birth_year'] будет ('ios', 'man', 1992).
        Если None, определить веса пропорционально доле страт в датафрейме data.
    seed - int, исходное состояние генератора случайных чисел для воспроизводимости
        результатов. Если None, то состояние генератора не устанавливается.

    return (data_pilot, data_control) - два датафрейма того же формата что и data
        c пилотной и контрольной группами.
    """
    if seed:
        np.random.seed(seed)

    if weights is None:
        len_data = len(data)
        weights = {strat: len(df_) / len_data for strat, df_ in data.groupby(strat_columns)}

    # кол-во элементов страты в группе
    strat_count_in_group = {strat: int(round(group_size * weight)) for strat, weight in weights.items()}

    pilot_dfs = []
    control_dfs = []
    for strat, data_strat in data.groupby(strat_columns):
        if strat in strat_count_in_group:
            count_in_group = strat_count_in_group[strat]
            index_data_groups = np.random.choice(
                np.arange(len(data_strat)),
                count_in_group * 2,
                False
            )
            pilot_dfs.append(data_strat.iloc[index_data_groups[:count_in_group]])
            control_dfs.append(data_strat.iloc[index_data_groups[count_in_group:]])
    data_pilot = pd.concat(pilot_dfs)
    data_control = pd.concat(control_dfs)
    return (data_pilot, data_control)
