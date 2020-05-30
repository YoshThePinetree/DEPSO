import TestFunctions
import numpy as np
import random


print("DEPSO Start")
# ========== #
# Initiation #
# ========== #
### Parameters
# test = TestFunctions.Functions("Rastrigin")
# test = TestFunctions.Functions("Ackley")
# test = TestFunctions.Functions("Rosenbrock")
test = TestFunctions.Functions("Griewank")
tmax = 10   # the number of maximum trial
imax = 200  # the number of maximum iteration
npop = 20   # the number of particles
ndim = 5    # the number of dimensions
w = 0.4     # the inertia weight
c1 = 2      # weight for the personal best
c2 = 2      # weight for the global best

### Memory Allocation
F = np.zeros((tmax,imax))
Xarc = np.zeros((tmax,imax,ndim))
random.seed(0)

# ========= #
# Main Loop #
# ========= #
for t in range(tmax):
    X = test.initialize(npop, ndim)  # the particle position
    V = test.initialize(npop, ndim)  # the particle velocity
    X1 = np.zeros((npop, ndim))
    V1 = np.zeros((npop, ndim))
    Xp = X                           # personal best individuals
    f = test.calc(X)                 # first fitness
    f1 = np.zeros_like(f)
    Xg = X[np.argmin(f),:]           # the personal best
    fg = np.min(f)

    ### Loop of a trial
    for ite in range(imax):
        for i in range(npop):
            for j in range(ndim):
                V1[i,j] = w*V[i,j] + c1*random.random()*(Xp[i,j] - X[i,j]) + \
                          c2*random.random()*(Xg[j] - X[i,j])
        X1 = X + V1  # position update
        f1 = test.calc(X1)  # updated fitness

        for i in range(npop):   # personal best update
            if f1[i] < f[i]:
                f[i] = f1[i]
                Xp[i,:] = X1[i,:]
        fg1 = np.min(f1)        # global best update
        if fg1 < fg:
            fg = fg1
            Xg = X1[np.argmin(f1),:]

        X = X1
        V = V1
        F[t,ite] = fg
        Xarc[t,ite,:] = Xg
        print("OF ",t,"-",ite," :",fg)

