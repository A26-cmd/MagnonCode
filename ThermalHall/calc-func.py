from mpmath import mp
import numpy as np

import matplotlib.pyplot as plt


def c2(x):
    return (1+x)*(mp.log(1+1/x))**2-(mp.log(x))**2 - 2*mp.polylog(2, -x)


def n(x):
    return 1/(mp.exp(x)-1)


xs = np.logspace(-6, 5, 1000)

data = np.empty((xs.size, 2))

for i, x in enumerate(xs):
    data[i, :] = x, float(c2(n(x)))

data = np.r_[np.array([[0, np.pi**2/3]]), data]

np.savetxt('func.txt', data)
