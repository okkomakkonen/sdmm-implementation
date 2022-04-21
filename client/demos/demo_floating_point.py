from time import time

import numpy as np

from sdmm import MatDotFloatingPoint

urls = ["http://localhost:5000/multiply"] * 22

matdot = MatDotFloatingPoint(
    num_partitions=5,
    num_colluding=5,
    urls=urls,
    rel_leakage=1e-3,
    std_a=1.0,
    std_b=1.0,
)

A = np.random.normal(loc=0.0, scale=1.0, size=(500, 500))
B = np.random.normal(loc=0.0, scale=1.0, size=(500, 500))

print("Computing using secure MatDot over floating point")
start = time()

C = matdot(A, B)

print(f"Took {time() - start:.3f}s")

print("Computing locally")
start = time()

AB = A @ B

print(f"Took {time() - start:.3f}s")

if np.isclose(AB, C).all():
    print("correct product")
else:
    print("wrong product")
