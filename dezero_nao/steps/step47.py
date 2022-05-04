if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero.models import MLP

model = MLP((10, 3))

x = np.array([0.2, -0.4])
y = model(x)

print(y)