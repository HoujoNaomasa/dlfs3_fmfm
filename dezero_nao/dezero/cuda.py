from multiprocessing import cpu_count
import numpy as np
gpu_enable = False
try:
    import cupy as np
    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable

def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data
    
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp