import numpy as np

from sdmm import MatDotFloatingPoint

urls = ["http://localhost:5000"] * 25

matdot = MatDotFloatingPoint(
    num_partitions=10,
    num_colluding=1,
    urls=urls,
    rel_leakage=1e-3,
    std_a=1.0,
    std_b=1.0,
    threaded=False,
    slow_multiplication=False,
)

(t, s, r) = (2000, 2000, 2000)

A = np.random.normal(loc=0.0, scale=1.0, size=(t, s))
B = np.random.normal(loc=0.0, scale=1.0, size=(s, r))

C = matdot(A, B)
