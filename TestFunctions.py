import numpy as np
import math


class Functions:
    def __init__(self, name):
        self.Fname = name

    def calc(self, x):
        if self.Fname == "Rastrigin":
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
        if self.Fname == "Ackley":
            y = x + 100
            return y
        else:
            y = x + 2
            return y
