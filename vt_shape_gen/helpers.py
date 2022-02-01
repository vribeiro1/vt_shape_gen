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
