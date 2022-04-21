import numpy as np
import json

from utils.distributed_multiplication import (
    multiply_at_server,
    multiply_at_servers,
    serialize_np_array,
    deserialize_np_array,
)

BASE_URL = "http://localhost:5000"


def test_multiplication_at_server_floating_point():

    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)

    url = BASE_URL + "/multiply"

    C = multiply_at_server(A, B, url)

    assert C.shape == (A @ B).shape
    assert (C == A @ B).all()


def test_multiplication_at_server_complex():

    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    B = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)

    url = BASE_URL + "/multiply"

    C = multiply_at_server(A, B, url)

    assert C.shape == (A @ B).shape
    assert (C == A @ B).all()


def test_multiplication_at_servers_floating_point():

    A_encoded = [np.random.rand(10, 10) for _ in range(10)]
    B_encoded = [np.random.rand(10, 10) for _ in range(10)]

    urls = [BASE_URL + "/multiply"] * 10

    C_encoded = multiply_at_servers(A_encoded, B_encoded, urls, num_responses=8)

    assert len(C_encoded) == 8

    for i, C in C_encoded:
        A = A_encoded[i]
        B = B_encoded[i]
        assert (C == A @ B).all()


def test_multiplication_at_servers_complex():

    A_encoded = [
        np.random.rand(10, 10) + 1j * np.random.rand(10, 10) for _ in range(10)
    ]
    B_encoded = [
        np.random.rand(10, 10) + 1j * np.random.rand(10, 10) for _ in range(10)
    ]

    urls = [BASE_URL + "/multiply"] * 10

    C_encoded = multiply_at_servers(A_encoded, B_encoded, urls, num_responses=8)

    assert len(C_encoded) == 8

    for i, C in C_encoded:
        A = A_encoded[i]
        B = B_encoded[i]
        assert (C == A @ B).all()


def test_serialization():

    A = np.random.rand(10, 10)

    s = json.dumps(serialize_np_array(A))
    d = json.loads(s)

    assert (A == deserialize_np_array(d)).all()
