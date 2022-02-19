import numpy as np
import time
from galois import GF

from sdmm import matdot

start = time.time()

p = 4
s = 200
M = 10
q = 2147483647

A = np.random.randint(M, size=(s, s))
B = np.random.randint(M, size=(s, s))

C = matdot(A, B, p=p, q=q, X=1, N=10)

print(f"Took {time.time() - start}s")

start = time.time()

F = GF(q)
A = F(A)
B = F(B)

AB = A @ B

print(f"Took {time.time() - start}s")

if (C == AB).all():
    print("correct product")
else:
    print("wrong product")
