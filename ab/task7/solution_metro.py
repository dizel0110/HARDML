import numpy as np
import hashlib


class ABSplitter:
    def __init__(self, count_slots, salt_one, salt_two):
        self.count_slots = count_slots
        self.salt_one = salt_one
        self.salt_two = salt_two

        self.slots = np.arange(count_slots)
        self.experiments = []
        self.experiment_to_slots = dict()
        self.slot_to_experiments = dict()

    def split_experiments(self, experiments):
        """Устанавливает множество экспериментов, распределяет их по слотам.

        Нужно определить атрибуты класса:
            self.experiments - список словарей с экспериментами
            self.experiment_to_slots - словать, {эксперимент: слоты}
            self.slot_to_experiments - словать, {слот: эксперименты}

        experiments - список словарей, описывающих пилот. Содержит два ключа:
            experiment_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_experiments - list, идентификаторы несовместных экспериментов.
        return: List[dict], список экспериментов, которые не удалось разместить по слотам.
            Возвращает пустой список, если всем экспериментам хватило слотов.
        """
        self.experiments = experiments
        experiments = sorted(
            experiments,
            key=lambda x: len(x['conflict_experiments']),
            reverse=True
        )

        experiments_without_place = []
        slot_to_experiments = {slot: [] for slot in self.slots}
        experiment_to_slots = {experiment['experiment_id']: [] for experiment in experiments}
        for experiment in experiments:
            # найдём доступные слоты
            notavailable_slots = []
            for conflict_experiment_id in experiment['conflict_experiments']:
                notavailable_slots += experiment_to_slots[conflict_experiment_id]
            available_slots = list(set(self.slots) - set(notavailable_slots))

            if experiment['count_slots'] > len(available_slots):
                experiments_without_place.append(experiment)
                continue

            # shuffle - чтобы внести случайность, иначе они все упорядочены будут по номеру slot
            np.random.shuffle(available_slots)
            available_slots_orderby_count_experiment = sorted(
                available_slots,
                key=lambda x: len(slot_to_experiments[x]), reverse=True
            )
            experiment_slots = available_slots_orderby_count_experiment[:experiment['count_slots']]
            experiment_to_slots[experiment['experiment_id']] = experiment_slots
            for slot in experiment_slots:
                slot_to_experiments[slot].append(experiment['experiment_id'])
        self.slot_to_experiments = slot_to_experiments
        self.experiment_to_slots = experiment_to_slots
        return experiments_without_place

    def _get_hash_modulo(self, value: str, modulo: int, salt: int):
        """Вычисляем остаток от деления: (hash(value) + salt) % modulo."""
        hash_value = int(hashlib.md5(str.encode(value)).hexdigest(), 16)
        return (hash_value + salt) % modulo
    

    def process_user(self, user_id: str):
        """Определяет в какие эксперименты попадает пользователь.

        Сначала нужно определить слот пользователя.
        Затем для каждого эксперимента в этом слоте выбрать пилотную или контрольную группу.

        user_id - идентификатор пользователя.

        return - (int, List[tuple]), слот и список пар (experiment_id, pilot/control group).
            Example: (2, [('exp 3', 'pilot'), ('exp 5', 'control')]).
        """
        slot = self._get_hash_modulo(user_id, self.count_slots, self.salt_one)
        slot_experiments_id = self.slot_to_experiments[slot]
        slot_experiments = [
            experiment for experiment in self.experiments
            if experiment['experiment_id'] in slot_experiments_id
        ]
        
        pairs_experiment_group = []
        for experiment in slot_experiments:
            second_hash = self._get_hash_modulo(
                user_id + experiment['experiment_id'],
                2,
                self.salt_two
            )
            group = 'pilot' if second_hash == 1 else 'control'
            pairs_experiment_group.append((experiment['experiment_id'], group))
        return (slot, pairs_experiment_group)