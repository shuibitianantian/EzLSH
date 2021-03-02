import sys
sys.path.append('..')
from LocalSensitiveHashing.lsh import LSH
from LocalSensitiveHashing.uitls import generate_data
import numpy as np
from lshash.lshash import LSHash

np.random.seed(18)
dimension = 10
size = 1000
hash_size = 3
num_tables = 8


if __name__ == '__main__':
    data = generate_data(size, dimension)

    # =============
    lsh = LSH(hash_size, dimension, num_tables)
    lsh.index(data)

    lsh_1 = LSHash(hash_size, dimension, num_tables)
    lsh_1.uniform_planes = lsh.projections

    for d in data:
        lsh_1.index(d)

    for i in range(num_tables):
        t1 = lsh.hash_tables[i]
        t2 = lsh_1.hash_tables[i]

        for k in t1:
            assert t1[k] == t2.get_val(k)

