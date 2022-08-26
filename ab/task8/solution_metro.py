import numpy as np
import pandas as pd
import scipy.stats as stats


class SequentialTester:
    def __init__(
        self, metric_name, time_column_name,
        alpha, beta, pdf_one, pdf_two
    ):
        """Создаём класс для проверки гипотезы о равенстве средних тестом Вальда.

        Предполагается, что среднее значение метрики у распределения альтернативной
        гипотезы с плотность pdf_two больше.

        :param metric_name: str, название стобца со значениями измерений.
        :param time_column_name: str, названия столбца с датой и временем измерения.
        :param alpha: float, допустимая ошибка первого рода.
        :param beta: float, допустимая ошибка второго рода.
        :param pdf_one: function, функция плотности распределения метрики при H0.
        :param pdf_two: function, функция плотности распределения метрики при H1.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        self.metric_name = metric_name
        self.time_column_name = time_column_name
        self.alpha = alpha
        self.beta = beta
        self.pdf_one = pdf_one
        self.pdf_two = pdf_two

        self.lower_bound = np.log(beta / (1 - alpha))
        self.upper_bound = np.log((1 - beta) / alpha)

        self.sum_log_ll = 0
        self.count_data = 0
        self.stop_test = False
        self.results = None

    def run_test(self, data_control, data_pilot):
        """Запускаем новый тест.
        
        :param data_control: pd.DataFrame, данные контрольной группы.
        :param data_pilot: pd.DataFrame, данные пилотной группы.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        self.sum_log_ll = 0
        self.count_data = 0
        self.stop_test = False
        self.results = None
        return self._check_test(data_control, data_pilot)

    def add_data(self, data_control, data_pilot):
        """Добавляет новые данные для теста.
        
        Гарантируется, что они новые и не дублируют ранее добавленные данные.
        
        :param data_control: pd.DataFrame, новые данные контрольной группы.
        :param data_pilot: pd.DataFrame, новые данные пилотной группы.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        if self.stop_test:
            return self.results
        return self._check_test(data_control, data_pilot)

    def _process_data(self, *list_data):
        """Сортирует данные по времени и возвращает массивы со значением метрики."""
        list_values = [
            data.sort_values(self.time_column_name)[self.metric_name].values
            for data in list_data
        ]
        return list_values

    def _check_test(self, data_control, data_pilot):
        """Последовательно проверяет отличие по мере поступления данных.

        pdf_one, pdf_two - функции плотности распределения при нулевой и альтернативной гипотезах

        Возвращает 1, если были найдены значимые отличия, иначе - 0.
            И кол-во объектов при принятии решения.
        """
        values_control, values_pilot = self._process_data(data_control, data_pilot)
        len_new_data = len(values_control)
        delta_values = values_pilot - values_control

        pdf_one_values = self.pdf_one(delta_values)
        pdf_two_values = self.pdf_two(delta_values)
        z_cumsum = np.cumsum(np.log(pdf_two_values / pdf_one_values)) + self.sum_log_ll

        indexes_lower = np.arange(len_new_data)[z_cumsum < self.lower_bound]
        indexes_upper = np.arange(len_new_data)[z_cumsum > self.upper_bound]
        first_index_lower = indexes_lower[0] if len(indexes_lower) > 0 else len_new_data
        first_index_upper = indexes_upper[0] if len(indexes_upper) > 0 else len_new_data

        if first_index_lower < first_index_upper:
            self.results = (0., self.count_data + first_index_lower + 1)
            self.stop_test = True
            return self.results
        elif first_index_lower > first_index_upper:
            self.results = (1., self.count_data + first_index_upper + 1)
            self.stop_test = True
            return self.results
        else:
            self.count_data += len_new_data
            self.sum_log_ll = z_cumsum[-1]
            self.results = (0.5, self.count_data)
            return self.results