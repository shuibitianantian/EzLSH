import sys
import numpy as np
import os


if __name__ == '__main__':
    size = int(sys.argv[1])
    dimension = int(sys.argv[2])

    sigma, mu = 2, 0.2

    data = sigma * np.random.randn(size, dimension) + mu
    file_name = f'{dimension}_{size}.npy'

    if file_name not in os.listdir('.'):
        with open(file_name, 'wb') as f:
            np.save(f, data)
