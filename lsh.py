# Reference: https://github.com/kayzhu/LSHash
import numpy as np
from uitls import Table


class LSH:
    def __init__(self, hash_size, input_dim, num_tables):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_tables = num_tables
        self._init_projections()
        self._init_storage()

    def _generate_uniform_planes(self):
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
        for p in inputs:
            self._index(p)

    @staticmethod
    def _hash(planes, input_point):
        input_point_np = np.array(input_point)
        return ''.join(['1' if c > 0 else '0' for c in np.dot(planes, input_point_np)])



