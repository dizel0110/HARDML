import numpy as np
import pandas as pd
import itertools


def get_strat_mask(data, strat, strat_columns):
    if type(strat) != tuple:
        strat = (strat, )
    strat_mask = np.ones(shape=data.shape[0], dtype=np.bool8)
    for col, val in zip(strat_columns, strat):
        strat_mask &= data[col] == val
    return strat_mask


def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    """Подбирает стратифицированные группы для эксперимента.

    data - pd.DataFrame, датафрейм с описанием объектов, содержит атрибуты для стратификации.
    strat_columns - List[str], список названий столбцов, по которым нужно стратифицировать.
    group_size - int, размеры групп.
    weights - dict, словарь весов страт {strat: weight}, где strat - либо tuple значений элементов страт,
        например, для strat_columns=['os', 'gender', 'birth_year'] будет ('ios', 'man', 1992), либо просто строка/число.
        Если None, определить веса пропорционально доле страт в датафрейме data.
    seed - int, исходное состояние генератора случайных чисел для воспроизводимости
        результатов. Если None, то состояние генератора не устанавливается.

    return (data_pilot, data_control) - два датафрейма того же формата что и data
        c пилотной и контрольной группами.
    """
    if seed is not None:
        np.random.seed(seed)

    pilot_idxs = []
    control_idxs = []

    if weights is not None:
        strats, weights = list(weights.keys()), list(weights.values())
    else:
        strats = list(itertools.product(*[list(data[c].unique()) 
                                          for c in strat_columns]))
        weights = [get_strat_mask(data, strat, strat_columns).sum() / len(data) 
                   for strat in strats]
    
    for strat, weight in zip(strats, weights):
        strat_mask = get_strat_mask(data, strat, strat_columns)
        data_strat = data.loc[strat_mask, :]

        assert data_strat.shape[0] != 0

        size = int(round(weight * group_size))
        strat_pilot_idx = np.random.choice(data_strat.index, size, False)
        strat_control_idx = np.random.choice(data_strat.index[~data_strat.index.isin(strat_pilot_idx)],
                                             size, False)

        pilot_idxs.append(strat_pilot_idx)
        control_idxs.append(strat_control_idx)

    pilot_idxs = np.concatenate(pilot_idxs)
    control_idxs = np.concatenate(control_idxs)
      
    return data.loc[pilot_idxs, :], data.loc[control_idxs, :]


if __name__ == '__main__':
    seed = 110894
    size = 100_000
    os = ['ios', 'android']
    genders = ['men', 'woomen']
    birth_years = [2019, 2020, 2021]
    data = pd.DataFrame({'metric': np.random.normal(loc=10, scale=1, size=size),
                         'os': np.random.choice(os, size=size, replace=True),
                         'gender': np.random.choice(genders, size=size, replace=True),
                         'birth_year': np.random.choice(birth_years, size=size, replace=True), })
    print(data.describe())

    strat_columns = ['os',
                    #  'gender',
                    #  'birth_year', 
                     ]
    # weights = None
    weights = {'ios': 0.3,
               'android': 0.7, }
    # weights={('ios', 'men'): 0.3,
    #          ('ios', 'woomen'): 0.1,
    #          ('android', 'men'): 0.5,
    #          ('android', 'woomen'): 0.1, }
    group_size = 2_000

    (data_pilot,
     data_control, ) = select_stratified_groups(data,
                                                strat_columns=strat_columns,
                                                group_size=group_size,
                                                weights=weights,
                                                seed=seed, )
    
    print(data_pilot.shape, data_control.shape)

    for strat, weight in weights.items():
        pilot_strat_mask = get_strat_mask(data_pilot, strat, strat_columns)
        control_strat_mask = get_strat_mask(data_control, strat, strat_columns)

        p_fracs = pilot_strat_mask.sum() / group_size
        c_fracs = control_strat_mask.sum() / group_size

        diff = max(abs(p_fracs - weight), abs(c_fracs - weight))
        print(f'For strat "{strat}" max diff {diff}, check {diff < (2 / group_size)}')

