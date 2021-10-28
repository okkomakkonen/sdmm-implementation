"""Provides some utility functions"""

from typing import Tuple, Optional

from functools import reduce
import os
from math import sqrt

import numpy as np

from Crypto.Cipher import AES


def partition_matrix(m, rp: int, cp: int):
    """Return a (multidimensional) list of submatrices of m"""
    r, c = m.shape
    assert r % rp == 0 and c % cp == 0, "matrix isn't evenly partitioned"
    R, C = r // rp, c // cp

    if rp == 1:
        return [m[:, C * j : C * (j + 1)] for j in range(cp)]
    if cp == 1:
        return [m[R * i : R * (i + 1), :] for i in range(rp)]
    return [
        [m[R * i : R * (i + 1), C * j : C * (j + 1)] for j in range(cp)]
        for i in range(rp)
    ]


def random_matrix(
    FF, shape: Tuple[int, int], seed: Optional[bytes] = None
) -> np.ndarray:
    """Return a random matrix from /dev/urandom (if seed is not defined) or from AES cipher

    seed: 40 bytes
    """

    bytes_per_symbol = FF(0).itemsize
    dtype = FF.dtypes[0]
    size = reduce(int.__mul__, shape)
    if seed is None:
        # return randomness from /dev/urandom
        random_bytes = os.urandom(bytes_per_symbol * size)
    else:
        # return AES randomness using counter mode and encrypting zero bytes
        key, nonce = seed[:32], seed[32:40]
        cipher = AES.new(key=key, mode=AES.MODE_CTR, nonce=nonce)
        z = np.zeros(bytes_per_symbol * size, dtype=np.uint8)
        random_bytes = cipher.encrypt(z.data)
    Z = np.frombuffer(random_bytes, dtype=dtype).reshape(shape) % FF.order
    return FF(Z)


def vandermonde_determinant(ev):

    N = len(ev)

    det = 1
    for j in range(N):
        for i in range(j):
            det *= ev[j] - ev[i]

    return det


def complex_random_matrix(var, shape: Tuple[int, int]) -> np.ndarray:
    """Return a complex normal random matrix with zero mean and variance var"""

    return np.random.normal(0, sqrt(var / 2), size=shape) + 1j * np.random.normal(
        0, sqrt(var / 2), size=shape
    )
