import pytest

import galois

from sdmm import MatDotFiniteField

BASE_URL = "http://localhost:5000"


@pytest.mark.xfail
def test_secure_matdot_finite_field():

    urls = [BASE_URL + "/multiply"] * 20

    matdot = MatDotFiniteField(
        num_partitions=5, num_colluding=5, num_servers=22, urls=urls, field_size=23
    )

    F = galois.GF(19)
    A = F.Random((50, 50))
    B = F.Random((50, 50))

    C = matdot(A, B)

    assert (C == A @ B).all()
