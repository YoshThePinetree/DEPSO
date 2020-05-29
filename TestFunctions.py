import numpy as np
import math


class Functions:
    def __init__(self, name):
        self.Fname = name

    def calc(self, x):
        if self.Fname == "Rastrigin":   # allows any number of dimensions
            # Limitation of search domain
            x = np.where(x <= -5.12, -5.12, x)
            x = np.where(x >= 5.12, 5.12, x)

            A = 10
            f = np.zeros(np.shape(x)[0])
            for i in range(np.shape(x)[0]):
                for j in range(np.shape(x)[1]):
                    f[i] = f[i] + (x[i,j] ** 2) - (A * math.cos(2 * math.pi * x[i,j]))
                f[i] = f[i] + A * np.shape(x)[1]

            return f
        if self.Fname == "Ackley":  # 2 dimensions only
            # Limitation of search domain
            x[:,0] = np.where(x[:,0] <= -5, -5, x[:,0])
            x[:,1] = np.where(x[:,1] >= 5, 5, x[:,1])

            f = np.zeros(np.shape(x)[0])
            for i in range(np.shape(x)[0]):
                f[i] = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x[i,0]**2 + x[i,1]**2))) - \
                       math.exp(0.5 * (math.cos(2*math.pi * x[i,0]) + math.cos(2*math.pi * x[i,1]))) + \
                       math.exp(1) + 20

            return f
        else:
            y = x + 2
            return y
