import numpy as np



def seeding_np_random(seed):
    rng = np.random.RandomState(seed=seed)
    return rng, seed
