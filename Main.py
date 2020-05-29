import numpy as np
import TestFunctions


# test = TestFunctions.Functions("Rastrigin")
test = TestFunctions.Functions("Ackley")
# x = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
# x = np.array([[0.0, 0.0, 0.0], [-5.12, -5.12, -0.3], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
x = np.array([[0.0, 0.0],[-5.8, -100]])
x = test.calc(x)
print(x)

