"""Provides some utility functions"""

from typing import Tuple, Optional, overload, List
from functools import reduce
import os

import numpy as np
from Crypto.Cipher import AES


class MatricesNotConformableException(Exception):
    """Matrices are not conformable"""


@overload
def partition_matrix(M: np.ndarray, horizontally: int) -> List[np.ndarray]:
    pass


@overload
def partition_matrix(M: np.ndarray, vertically: int) -> List[np.ndarray]:
    pass


@overload
def partition_matrix(
    M: np.ndarray, *, horizontally: int, vertically: int
) -> List[List[np.ndarray]]:
    pass


def partition_matrix(M, *, horizontally=None, vertically=None):
    n, m = M.shape
    if horizontally is not None and vertically is None:
        # split horizontally
        if m % horizontally != 0:
            raise ValueError("can't be evenly split")
        ms = m // horizontally
        return [M[:, i * ms : (i + 1) * ms] for i in range(horizontally)]
    if horizontally is None and vertically is not None:
        # split vertically
        if n % vertically != 0:
            raise ValueError("can't be evenly split")
        ns = n // vertically
        return [M[i * ns : (i + 1) * ns, :] for i in range(vertically)]
    if horizontally is not None and vertically is not None:
        # split both
        if n % vertically != 0 or m % horizontally != 0:
            raise ValueError("can't be evenly split")
        ms = m // horizontally
        ns = n // vertically
        return [
            [M[i * ns : (i + 1) * ns, j * ms : (j + 1) * ms] for j in range(vertically)]
            for i in range(vertically)
        ]
    raise ValueError("matrix must be split either horizontally or vertically (or both)")


def pad_matrix(A, *, horizontally=None, vertically=None):

    t, s = A.shape

    if horizontally is not None and vertically is None:
        pad = (-s) % horizontally
        return np.pad(A, ((0, 0), (0, pad)), mode="constant")

    if horizontally is None and vertically is not None:
        pad = (-t) % vertically
        return np.pad(A, ((0, pad), (0, 0)), mode="constant")

    if horizontally is not None and vertically is not None:
        pad1 = (-t) % horizontally
        pad2 = (-s) % vertically
        return np.pad(A, ((0, pad1), (0, pad2)), mode="constant")

    return A


def check_conformable_and_compute_shapes(A, B):

    t, sA = A.shape
    sB, r = B.shape

    if sA != sB:
        raise MatricesNotConformableException("Matrix dimensions do not match")
    s = sA

    return t, s, r


def safe_random_matrix(
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


def complex_normal(
    loc: Optional[float] = 0.0,
    scale: Optional[float] = 1.0,
    size: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    return np.random.normal(
        loc=loc.real, scale=scale / 2.0, size=size
    ) + 1j * np.random.normal(loc=loc.imag, scale=scale / 2.0, size=size)


def vandermonde_determinant(ev):

    N = len(ev)

    det = 1
    for j in range(N):
        for i in range(j):
            det *= ev[j] - ev[i]

    return det
