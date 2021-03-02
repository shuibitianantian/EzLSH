import numpy as np


class Table:
    def __init__(self, index):
        self.storage = {}
        self.index = index

    def __setitem__(self, key, values):
        self.storage[key] = values

    def __getitem__(self, item):
        return self.storage.get(item, [])

    def append_val(self, key, value):
        self.storage.setdefault(key, []).append(value)

    def __str__(self):
        return str(self.storage)

    def __iter__(self):
        return iter(self.storage)


def generate_data(n_points=100, dimension=100):
    return np.random.randn(n_points, dimension)


