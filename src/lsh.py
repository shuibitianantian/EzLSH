# Reference: https://github.com/kayzhu/LSHash
import numpy as np


class Table:
    def __init__(self, index):
        self.storage = {}  # real storage
        self.index = index  # not necessary right now

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


class LSH:
    def __init__(self, hash_size, input_dim, num_tables):
        self.hash_size = hash_size  # dimension after hashing
        self.input_dim = input_dim  # input dimension
        self.num_tables = num_tables  # number of tables, more tables means more hash collision
        self._init_projections()  # generate random projection plane, each plane corresponding to 1 hash table
        self._init_storage()

    def _generate_uniform_planes(self):
        """
        Generate projection plane randomly
        :return: np.ndarray
        """
        return np.random.randn(self.hash_size, self.input_dim)

    def _init_projections(self):
        self.projections = [self._generate_uniform_planes() for _ in range(self.num_tables)]

    def _init_storage(self):
        self.hash_tables = [Table(i) for i in range(self.num_tables)]

    def _index(self, input_point):
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.projections[i], input_point)
            table.append_val(binary_hash, tuple(input_point))

    def index(self, inputs):
        """
        Indexing a list of input points
        :param inputs:
        :return:
        """
        for p in inputs:
            self._index(p)

    @staticmethod
    def _hash(planes, input_point):
        input_point_np = np.array(input_point)
        return ''.join(['1' if c > 0 else '0' for c in np.dot(planes, input_point_np)])



