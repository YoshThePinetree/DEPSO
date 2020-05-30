import numpy as np
import math
import random


class Functions:
    def __init__(self, name):
        self.Fname = name
        if self.Fname == "Rastrigin":
            self.llim = -5.12
            self.ulim = 5.12
        if self.Fname == "Ackley":
            self.llim = -5
            self.ulim = 5
        if self.Fname == "Rosenbrock":
            self.llim = -10
            self.ulim = 10
        if self.Fname == "Griewank":
            self.llim = -600
            self.ulim = 600

    def initialize(self, npop, ndim):
        y = np.zeros((npop, ndim))
        for i in range(npop):
            for j in range(ndim):
                y[i,j] = random.uniform(self.llim,self.ulim)
        return y

    def calc(self, x):
        # Limitation of search domain
        x = np.where(x <= self.llim, self.llim, x)
        x = np.where(x >= self.ulim, self.ulim, x)

        if self.Fname == "Rastrigin":   # allows any number of dimensions
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
        if self.Fname == "Rosenbrock":  # allows any number of dimensions
            f = np.zeros(np.shape(x)[0])
            for i in range(np.shape(x)[0]):
                for j in range(np.shape(x)[1]-1):
                    f[i] = f[i] + (100*(x[i,j+1] - x[i,j]**2)**2 + (1 - x[i,j])**2)
            return f

        if self.Fname == "Griewank":  # allows any number of dimensions
            f1 = np.zeros((np.shape(x)[0]))
            f2 = np.ones((np.shape(x)[0]))
            for i in range(np.shape(x)[0]):
                for j in range(np.shape(x)[1]):
                    f1[i] = f1[i] + (x[i,j]**2 / 4000)
                    f2[i] = f2[i] * (math.cos(x[i,j] / math.sqrt(i+1)))
            f = f1 - f2 + 1
            return f
