from random import Random
import numpy as np


class Partition:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        data_idx = self.index[item]
        return self.data[data_idx]


class DataPartitioner:
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1111):
        self.data = data
        self.partitions = []
        rand = Random()
        rand.seed(seed)
        data_len = len(data)
        indexes = np.arange(data_len)
        rand.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])



