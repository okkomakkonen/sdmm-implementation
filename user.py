import numpy as np

from .sdmm import matdot

p = 2
s = 10
M = 10
q = 32749

A = np.random.randint(M, size=(s, s))
B = np.random.randint(M, size=(s, s))

C = matdot(A, B, p=p, q=q, X=2, N=10)

if (C == A@B).all():
    print("correct product")
else:
    print("wrong product")


