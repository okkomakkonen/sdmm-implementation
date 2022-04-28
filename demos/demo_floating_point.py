from time import time

import numpy as np

from sdmm import MatDotFloatingPoint
from sdmm.utils import fake_multiply, fast_multiply, slow_multiply

urls = ["http://localhost:5000"] * 25

matdot = MatDotFloatingPoint(
    num_partitions=10,
    num_colluding=1,
    urls=urls,
    rel_leakage=1e-3,
    std_a=1.0,
    std_b=1.0,
    threaded=True,
    slow_multiplication=False,
)

(t, s, r) = (2000, 2000, 2000)

A = np.random.normal(loc=0.0, scale=1.0, size=(t, s))
B = np.random.normal(loc=0.0, scale=1.0, size=(s, r))

print(f"Multiplying {t} x {s} and {s} x {r} matrices")
print("Computing using secure MatDot over floating point")
start = time()

C = matdot(A, B)

print(f"Took \033[1m{time() - start:.3f}s\033[m")

print("Computing locally")
start = time()

AB = fast_multiply(A, B)

print(f"Took \033[1m{time() - start:.3f}s\033[m")

if np.isclose(AB, C).all():
    print("\033[32;1mcorrect product\033[m")
else:
    print("\033[31;1mwrong product\033[m")
