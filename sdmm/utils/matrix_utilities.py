"""Provides some utility functions"""

import os
from functools import reduce
from typing import List, Optional, Tuple, overload

import numpy as np
from Crypto.Cipher import AES  # type: ignore


class MatricesNotConformableException(Exception):
    """Matrices are not conformable"""


class MatrixDimensionNotDivisibleException(Exception):
    """Matrix dimension is not evenly divisible"""


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
    """Return a list of partitions of the matrix"""
    n, m = M.shape
    if horizontally is not None and vertically is None:
        # split horizontally
        if m % horizontally != 0:
            raise MatrixDimensionNotDivisibleException("Can not be evenly split")
        ms = m // horizontally
        return [M[:, i * ms : (i + 1) * ms] for i in range(horizontally)]
    if horizontally is None and vertically is not None:
        # split vertically
        if n % vertically != 0:
            raise MatrixDimensionNotDivisibleException("Can not be evenly split")
        ns = n // vertically
        return [M[i * ns : (i + 1) * ns, :] for i in range(vertically)]
    if horizontally is not None and vertically is not None:
        # split both
        if n % vertically != 0 or m % horizontally != 0:
            raise MatrixDimensionNotDivisibleException("Can not be evenly split")
        ms = m // horizontally
        ns = n // vertically
        return [
            [M[i * ns : (i + 1) * ns, j * ms : (j + 1) * ms] for j in range(vertically)]
            for i in range(vertically)
        ]
    raise ValueError("Matrix must be split either horizontally or vertically (or both)")


def pad_matrix(A, *, horizontally=None, vertically=None):
    """Pad the matrix with zeros such that the specified axis is divisible by the specified value"""

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


def check_conformable_and_compute_shapes(
    A: np.ndarray, B: np.ndarray
) -> Tuple[int, int, int]:
    """Check that the matrices are conformable and return the dimensions"""

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
    loc: float = 0.0,
    scale: float = 1.0,
    size: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:

    return np.random.normal(
        loc=loc.real, scale=scale / 2.0, size=size
    ) + 1j * np.random.normal(loc=loc.imag, scale=scale / 2.0, size=size)


def fake_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return a zero matrix that matches the size of A*B"""

    t, _ = A.shape
    _, r = B.shape

    return np.zeros((t, r), dtype=A.dtype)


def fast_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    return A @ B


def slow_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the product A*B using the definition of matrix product

    Warning: this is slow!
    """

    t, s = A.shape
    s, r = B.shape

    # Support finite fields as well
    if hasattr(type(A), "order"):
        zero = type(A)(0)
        return np.array(
            [
                [
                    sum((A[i, j] * B[j, k] for j in range(s)), start=zero)
                    for k in range(r)
                ]
                for i in range(t)
            ]
        )

    return np.array(
        [[sum(A[i, j] * B[j, k] for j in range(s)) for k in range(r)] for i in range(t)]
    )
