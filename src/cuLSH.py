import cupy as cp
import numpy as np
import time

cp.random.seed(18)

"""
This is a simple implementation of random projection locality sensitive hashing using cupy

References: 
    https://github.com/kayzhu/LSHash
    https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23
    
"""


def _hash(_inputs, projections, bits):
    xp = cp.get_array_module(_inputs)
    signs = ~xp.signbit(xp.matmul(_inputs, projections))
    h = xp.matmul(signs, bits)

    return h


def load_data(file_name):
    meta = file_name.split('/')[-1].split('_')
    dim, s = meta[0], meta[1][:-4]
    return int(dim), int(s), np.load(file_name)


def cosine_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CuLSH:
    def __init__(self, hash_size, input_dim, num_tables):
        """
        initialize lsh
        :param hash_size: hash space size
        :param input_dim: input dimension
        :param num_tables: number of tables (in order to maximize hash collision)
        """
        self.inputs = None
        self.hash_size = hash_size  # dimension after hashing
        self.input_dim = input_dim  # input dimension
        self.num_tables = num_tables  # number of tables, more tables means more hash collision
        self._init_projections()  # generate random projection plane, each plane corresponding to 1 hash table
        self._init_storage()
        self._init_bit()

    def _init_projections(self, sigma=2, mu=1):
        self.projections = sigma * np.random.randn(self.num_tables, self.hash_size, self.input_dim) + mu

    def _init_storage(self):
        self.hash_tables = [{} for _ in range(self.num_tables)]

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

    def hash(self, inputs, device='cpu'):
        if device == 'gpu':
            self._move_to_cuda()
            gpu_inputs = cp.asarray(inputs)
            flags = _hash(gpu_inputs, self.projections.transpose(0, 2, 1), self.bits)
        elif device == 'cpu':
            self._move_to_cpu()
            flags = _hash(inputs, self.projections.transpose(0, 2, 1), self.bits)
        else:
            raise Exception("Can not decide which device to use")

        return cp.asnumpy(flags)

    def index(self, inputs, device='cpu'):
        """
        Indexing a list of input points
        :param inputs:
        :param device:
        :return:
        """
        if self.inputs is not None:
            cur_size = self.inputs.shape[0]
            self.inputs = np.vstack((self.inputs, inputs))
        else:
            cur_size = 0
            self.inputs = inputs

        flags = self.hash(inputs, device)

        for i, b in enumerate(flags):
            for j, v in enumerate(b):
                self.hash_tables[i].setdefault(v, []).append(int(cur_size + j))

    def query(self, inputs, k, device='cpu'):
        flags = self.hash(inputs, device).T

        # transpose flags, so the horizontal dimension is the number of hash tables,
        # the vertical dimension is the number of queried data

        results = np.empty((inputs.shape[0], k, self.input_dim), dtype=object)

        for i, t in enumerate(flags):  # i represents the index of data point
            similar_data = np.zeros(self.inputs.shape[0], dtype=bool)
            for idx, hv in enumerate(t):  # idx represents the index of hash table
                similar_data[np.asarray(self.hash_tables[idx].get(hv))] = True

            similar_data = self.inputs[np.asarray(similar_data)]

            similarities = similar_data.dot(inputs[i]) / (np.linalg.norm(similar_data) * np.linalg.norm(inputs[i]))

            # in case of k larger than the size of similarities array
            s = k if k < len(similar_data) else len(similar_data)
            mx_args = np.argsort(similarities)[::-1][:s]
            results[i, :] = self.inputs[mx_args]

        return results

    @property
    def indexed_size(self):
        return self.inputs.shape[0]


if __name__ == '__main__':

    hash_size = 12
    num_tables = 4

    dimension, size, data = load_data('../data/100_500000.npy')
    _, _, query_data = load_data('../data/100_500000.npy')

    print("loading finished..")

    t = time.time()
    lsh = CuLSH(hash_size, dimension, num_tables)
    lsh.index(data, 'gpu')
    # queried = lsh.query(data[:500], 2, 'cpu')

    print(time.time() - t)
