import sys
sys.path.append('..')
from src.utils import generate_data
import numpy as np
import os


if __name__ == '__main__':
    size = int(sys.argv[1])
    dimension = int(sys.argv[2])

    data = generate_data(size, dimension)
    file_name = f'{dimension}_{size}.txt'
    if file_name not in os.listdir('.'):
        np.savetxt(file_name, data)
