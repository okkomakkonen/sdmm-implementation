import numpy as np
import yappi

from sdmm import MatDotFloatingPoint

urls = ["http://localhost:5000"] * 25
# urls = [f"http://localhost:{port}" for port in range(5000, 5000 + 25)]

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

# Starting profiling after setup

yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
yappi.start()

C = matdot(A, B)

yappi.stop()

print("Function stats for script")
stats = yappi.get_func_stats()
stats = yappi.convert2pstats(stats)
stats.sort_stats("cumulative")
stats.print_stats(0.1)

threads = yappi.get_thread_stats()
for thread in threads:
    print(f"Function stats for ({thread.name}) ({thread.id})")
    stats = yappi.get_func_stats(ctx_id=thread.id)
    stats = yappi.convert2pstats(stats)
    stats.sort_stats("cumulative")
    stats.print_stats(0.1)
