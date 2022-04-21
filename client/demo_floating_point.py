from time import time

import numpy as np

from sdmm import MatDotFloatingPoint

urls = ["http://localhost:5000/multiply"] * 20

matdot = MatDotFloatingPoint(
    num_partitions=5,
    num_colluding=5,
    num_servers=22,
    urls=urls,
    rel_leakage=1e-3,
    std_a=1.0,
    std_b=1.0,
)

A = np.random.normal(loc=0.0, scale=1.0, size=(5000, 5000))
B = np.random.normal(loc=0.0, scale=1.0, size=(5000, 5000))

start = time()

C = matdot(A, B)

print(f"Took {time() - start:.2f}s")

start = time()

AB = A @ B

print(f"Took {time() - start:.2f}s")

if np.isclose(AB, C).all():
    print("correct product")
else:
    print("wrong product")
