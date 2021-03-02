import numpy as np


def generate_data(n_points=100, dimension=100):
    """
    Generate random data points
    :param n_points: number of points
    :param dimension: dimension of each data point
    :return: np.ndarray
    """
    return np.random.randn(n_points, dimension)


