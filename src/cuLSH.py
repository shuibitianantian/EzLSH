import cupy as cp
from lsh import Table
import numpy as np
# import torch


cp.random.seed(18)


def _hash(_inputs, projections, bits):
    xp = cp.get_array_module(_inputs)
    signs = ~xp.signbit(xp.matmul(_inputs, projections))
    h = xp.matmul(signs, bits)

    return h


class CuLSH:
    def __init__(self, hash_size, input_dim, num_tables):
        self.hash_size = hash_size  # dimension after hashing
        self.input_dim = input_dim  # input dimension
        self.num_tables = num_tables  # number of tables, more tables means more hash collision
        self._init_projections()  # generate random projection plane, each plane corresponding to 1 hash table
        self._init_storage()
        self._init_bit()

    @staticmethod
    def choose_device(shape):
        if shape[0] * shape[1] > 100000:
            return 'gpu'
        else:
            return 'cpu'

    def _init_projections(self):
        self.projections = np.random.randn(self.num_tables, self.hash_size, self.input_dim)

    def _init_storage(self):
        self.hash_tables = [Table(i) for i in range(self.num_tables)]

    def _init_bit(self):
        self.bits = np.array(list(reversed([2**i for i in range(self.hash_size)])))

    def _move_to_cuda(self):
        self.projections = cp.asarray(self.projections)
        self.bits = cp.asarray(self.bits)

    def _move_to_cpu(self):
        if isinstance(self.projections, cp.ndarray):
            self.projections = cp.asnumpy(self.projections)
        if isinstance(self.bits, cp.ndarray):
            self.bits = cp.asnumpy(self.projections)

    def index(self, inputs, device='gpu'):
        """
        Indexing a list of input points
        :param inputs:
        :param device:
        :return:
        """
        # device = CuLSH.choose_device(inputs.shape)

        if device == 'gpu':
            self._move_to_cuda()
            gpu_inputs = cp.asarray(inputs)
            flags = _hash(gpu_inputs, self.projections.transpose(0, 2, 1), self.bits)
        elif device == 'cpu':
            self._move_to_cpu()
            flags = _hash(inputs, self.projections.transpose(0, 2, 1), self.bits)
        else:
            raise Exception("Can not decide which device to use")

        flags = cp.asnumpy(flags)
        # for i, b in enumerate(flags):
        #     for j, v in enumerate(b):
        #         self.hash_tables[i].storage.setdefault(v, []).append(inputs[j])


import sys
sys.path.append('..')
from src.utils import generate_data
from lsh import LSH


dimension = 500
size = 600000
hash_size = 4
num_tables = 4


if __name__ == '__main__':
    data = generate_data(size, dimension)

    import time
    # =============
    t = time.time()
    lsh = CuLSH(hash_size, dimension, num_tables)
    lsh.index(data, 'cpu')
    print(time.time() - t)

    # t = time.time()
    # lsh_1 = LSH(hash_size, dimension, num_tables)
    # lsh_1.projections = cp.asnumpy(lsh.projections)
    # lsh_1.index(data)
    # print(time.time() - t)

    # for i in range(num_tables):
    #     # t1 = lsh.hash_tables[i]
    #     t2 = lsh_1.hash_tables[i]
    #
    #     for k in t2:
    #         # assert len(t1[int(k, 2)]) == len(t2[k]) != 0
    #         assert len(t1[k]) == len(t2[k])
