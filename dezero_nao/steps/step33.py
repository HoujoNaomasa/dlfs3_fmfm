if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.clearngrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.clearngrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data