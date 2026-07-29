import numpy as np

def readout():
    bit =  np.random.default_rng().choice([0, 1], p=[0.5, 0.5])
    return bit

def arm():
    pass