from time import time

import numpy as np

from sdmm import matdot_floating_point

p = 2
s = 5000

A = np.random.normal(loc=0.0, scale=1.0, size=(s, s))
B = np.random.normal(loc=0.0, scale=1.0, size=(s, s))

start = time()

C = matdot_floating_point(A, B, p=p, X=2, N=10, sigmaa=1.0, sigmab=1.0, delta=0.01)

print(f"Took {time() - start} seconds")

start = time()

AB = A @ B

print(f"Took {time() - start} seconds")

if np.isclose(AB, C).all():
    print("correct product")
else:
    print("wrong product")
