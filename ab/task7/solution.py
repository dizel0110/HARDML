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
            self.experiment_to_slots - словарь, {эксперимент: слоты}
            self.slot_to_experiments - словарь, {слот: эксперименты}
        experiments - список словарей, описывающих пилот. Словари содержит три ключа:
            experiment_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_experiments - list, идентификаторы несовместных экспериментов.
            Пример: {'experiment_id': 'exp_16', 'count_slots': 3, 'conflict_experiments': ['exp_13']}
        return: List[dict], список экспериментов, которые не удалось разместить по слотам.
            Возвращает пустой список, если всем экспериментам хватило слотов.
        """
        self.experiments = experiments
        slot_to_experiments = {slot: [] for slot in self.slots}
        experiment_to_slots = {experiment['experiment_id']: [] for experiment in experiments}

        sorted_experiments = list(sorted(self.experiments,
                                         key=lambda x: len(x['conflict_experiments']),
                                         reverse=True))

        no_slots_experiments = list()
        for experiment in sorted_experiments:
            n_slots = experiment['count_slots']
            conflict_exps = experiment['conflict_experiments']

            conflict_slots = list()
            for exp in conflict_exps:
                if exp in experiment_to_slots:
                    conflict_slots.extend(experiment_to_slots[exp])

            allowed_slots = [slot 
                             for slot in self.slots 
                             if slot not in conflict_slots]

            if len(allowed_slots) < n_slots:
                no_slots_experiments.append(experiment)
                continue

            np.random.shuffle(allowed_slots)
            sorted_allowed_slots = list((sorted(allowed_slots,
                                                key=lambda x: len(slot_to_experiments[x]),
                                                reverse=True)))
            
            slots = sorted_allowed_slots[:n_slots]

            experiment_to_slots[experiment['experiment_id']] = slots
            for slot in slots:
                slot_to_experiments[slot].append(experiment['experiment_id'])

        self.experiment_to_slots = experiment_to_slots
        self.slot_to_experiments = slot_to_experiments

        return no_slots_experiments

    def process_user(self, user_id: str):
        """Определяет в какие эксперименты попадает пользователь.

        Сначала нужно определить слот пользователя.
        Затем для каждого эксперимента в этом слоте выбрать пилотную или контрольную группу.

        user_id - идентификатор пользователя.

        return - (int, List[tuple]), слот и список пар (experiment_id, pilot/control group).
            Example: (2, [('exp 3', 'pilot'), ('exp 5', 'control')]).
        """
        hash_one = int(hashlib.md5(str.encode(str(user_id) + str(self.salt_one))).hexdigest(), 16)
        n_slot = hash_one % self.count_slots

        slot_experimets = [exp 
                           for exp in self.experiments
                           if exp['experiment_id'] in self.slot_to_experiments[n_slot]]

        results = list()
        for experiment in slot_experimets:
            hash_two = int(hashlib.md5(str.encode(str(user_id) 
                                                  + str(experiment['experiment_id']) 
                                                  + str(self.salt_two))).hexdigest(), 16)
            results.append((experiment['experiment_id'],
                           'pilot' if hash_two % 2 == 1 else 'control'))
        
        return (n_slot, results)


if __name__ == '__main__':
    count_slots = 6
    experiments = [{'experiment_id': 'exp_1', 'count_slots': 2, 'conflict_experiments': []},
                   {'experiment_id': 'exp_2', 'count_slots': 2, 'conflict_experiments': []},
                   {'experiment_id': 'exp_3', 'count_slots': 4, 'conflict_experiments': []},]
    user_ids = [str(id) for id in np.arange(1000)]
    
    splitter = ABSplitter(count_slots, 'salt1', 'salt2')

    print(splitter.split_experiments(experiments))
    print(splitter.experiment_to_slots)
    print([splitter.process_user(id) for id in user_ids[:5]])
