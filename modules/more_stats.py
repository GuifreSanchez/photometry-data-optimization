import math
import numpy as np

def rms(arr):
    m = np.mean(arr)
    return np.sqrt(((arr / m - 1.)**2).mean())
