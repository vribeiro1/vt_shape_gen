import numpy as np
import random
import torch


def set_seeds(*args):
    """
    Set random seeds.
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def load_articulator_array(filepath, norm_value=None):
    """
    Loads the target array with the proper orientation (right to left)

    Args:
    filepath (str): Path to the articulator array
    """
    articul_array = np.load(filepath)
    n_rows, _ = articul_array.shape
    if n_rows == 2:
        articul_array = articul_array.T

    # All the countors should be oriented from right to left. If it is the opposite,
    # we flip the array.
    if articul_array[0][0] < articul_array[-1][0]:
        articul_array = np.flip(articul_array, axis=0)

    if norm_value is not None:
        articul_array = articul_array.copy() / norm_value

    return articul_array
