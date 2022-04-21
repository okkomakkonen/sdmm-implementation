import numpy as np

from sdmm import MatDotFloatingPoint

BASE_URL = "http://localhost:5000"


def test_secure_matdot_floating_point_real():

    urls = [BASE_URL + "/multiply"] * 20

    matdot = MatDotFloatingPoint(
        num_partitions=5,
        num_colluding=5,
        num_servers=22,
        urls=urls,
        rel_leakage=1e-3,
        std_a=1.0,
        std_b=1.0,
    )

    A = np.random.normal(loc=0.0, scale=1.0, size=(50, 50))
    B = np.random.normal(loc=0.0, scale=1.0, size=(50, 50))

    C = matdot(A, B)

    assert np.isclose(A @ B, C).all()


def test_secure_matdot_floating_point_complex():

    urls = [BASE_URL + "/multiply"] * 20

    matdot = MatDotFloatingPoint(
        num_partitions=5,
        num_colluding=5,
        num_servers=22,
        urls=urls,
        rel_leakage=1e-3,
        std_a=1.0,
        std_b=1.0,
    )

    A = np.random.normal(loc=0.0, scale=1.0, size=(50, 50)) + 1j * np.random.normal(
        loc=0.0, scale=1.0, size=(50, 50)
    )
    B = np.random.normal(loc=0.0, scale=1.0, size=(50, 50)) + 1j * np.random.normal(
        loc=0.0, scale=1.0, size=(50, 50)
    )

    C = matdot(A, B)

    assert np.isclose(A @ B, C).all()
