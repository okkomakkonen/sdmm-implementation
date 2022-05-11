from time import time

import numpy as np

from sdmm import MatDotFloatingPoint
from sdmm.utils import fake_multiply, fast_multiply, slow_multiply

# If SLOW = True, then the multiplication will be performed using the slow matrix multiplication, which is not optimized
# If SLOW = False, then the multiplication will be performed using the optimized matrix multiplication algorithm in numpy
SLOW = False

# Urls of the servers to use
# The server instances have to be running at this address for this demo to work
urls = ["http://localhost:5000"] * 25

# Initiaizing the secure MatDot instance
matdot = MatDotFloatingPoint(
    num_partitions=10,
    num_colluding=1,
    urls=urls,
    rel_leakage=1e-3,
    std_a=1.0,
    std_b=1.0,
    threaded=True,
    slow_multiplication=SLOW,
)

# The sizes of the matrices
# A is t x s and B is s x r
(t, s, r) = (400, 400, 400)

# Drawing random matrices to use for the multiplication
A = np.random.normal(loc=0.0, scale=1.0, size=(t, s))
B = np.random.normal(loc=0.0, scale=1.0, size=(s, r))

print(f"Multiplying {t} x {s} and {s} x {r} matrices")

# ----- COMPUTING USING MATDOT -----
print("Computing using secure MatDot over floating point")
start = time()

C = matdot(A, B)

print(f"Took \033[1m{time() - start:.3f}s\033[m")

# ----- COMPUTING LOCALLY -----
print("Computing locally")
start = time()

if SLOW:
    AB = slow_multiply(A, B)
else:
    AB = fast_multiply(A, B)

print(f"Took \033[1m{time() - start:.3f}s\033[m")

# Checking the result
if np.isclose(AB, C).all():
    print("\033[32;1mcorrect product\033[m")
else:
    print("\033[31;1mwrong product\033[m")
