"""Provides an implementation of the secure MatDot code for floating point numbers."""
from math import log, sqrt, pi
from multiprocessing.sharedctypes import Value
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

from utils import (
    complex_normal,
    partition_matrix,
    multiply_at_servers,
    pad_matrix,
    check_conformable_and_compute_shapes,
)


class MatDotFloatingPoint:
    def __init__(
        self,
        *,
        num_partitions: int,
        num_colluding: int,
        num_servers: int,
        urls: List[str],
        std_a: Optional[float] = None,
        std_b: Optional[float] = None,
        rel_leakage: Optional[float] = None
    ) -> None:

        # TODO: validate inputs
        if num_partitions <= 0 or num_colluding <= 0 or num_servers <= 0:
            raise ValueError("Number of servers and partitions has to be positive")

        self.p = num_partitions
        self.X = num_colluding
        self.N = num_servers
        self.K = 2 * self.p + 2 * self.X - 1

        if self.N < self.K:
            raise ValueError("Too few servers for SDMM")

        self.urls = urls

        if rel_leakage <= 0:
            raise ValueError("Relative leakage has to be positive")

        if std_a <= 0 or std_b <= 0:
            raise ValueError("Standard deviations have to be positive")

        self.rel_leakage = rel_leakage
        self.std_a = std_a
        self.std_b = std_b

        # precompute the evaluation points that we will use
        self.alphas = np.exp([2j * pi * n / self.N for n in range(self.N)])

        self.trr: Optional[float] = None
        self.trs: Optional[float] = None

    def _compute_required_std(
        self,
        shape: Tuple[int, int, int],
        std_a: Optional[float],
        std_b: Optional[float],
        rel_leakage: Optional[float],
    ) -> float:
        """Compute the standard deviation to achieve rel_leakage of information leakage"""

        if rel_leakage is None:
            rel_leakage = self.rel_leakage

        if rel_leakage is None:
            raise RuntimeError("rel_leakage has to be specified")

        t, s, r = shape

        if self.trr is None or self.trs is None:
            V = np.vander(self.alphas[: self.X], self.p, increasing=True)
            U = np.diag(self.alphas[: self.X] ** self.p) @ np.vander(
                self.alphas[: self.X], self.X, increasing=True
            )
            M = np.linalg.inv(U) @ V
            M = M.conjugate().transpose() @ M
            self.trr = M.trace().real

            V = np.vander(self.alphas[: self.X], self.p, increasing=False)
            U = np.diag(self.alphas[: self.X] ** self.p) @ np.vander(
                self.alphas[: self.X], self.X, increasing=True
            )
            M = np.linalg.inv(U) @ V
            M = M.conjugate().transpose() @ M
            self.trs = M.trace().real

        return 0.0001

    def encode_A(self, A: np.ndarray, std: float) -> List[np.ndarray]:
        """Encode the matrix A"""

        AP = partition_matrix(A, horizontally=self.p)
        size = AP[0].shape
        R = [complex_normal(scale=std, size=size) for _ in range(self.X)]
        APR = AP + R
        AT = iter(sum(a * x**i for i, a in enumerate(APR)) for x in self.alphas)
        return AT

    def encode_B(self, B: np.ndarray, std: float) -> List[np.ndarray]:
        """Encode the matrix B"""

        BP = partition_matrix(B, vertically=self.p)
        size = BP[0].shape
        S = [complex_normal(scale=std, size=size) for _ in range(self.X)]
        BPS = list(reversed(BP)) + S
        BT = iter(sum(b * x**i for i, b in enumerate(BPS)) for x in self.alphas)
        return BT

    def __call__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        std_a: Optional[float] = None,
        std_b: Optional[float] = None,
        rel_leakage: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the matrix product of A and B using secure MatDot for floating point numbers"""

        # check that the matrices are the right sizes
        t, s, r = check_conformable_and_compute_shapes(A, B)

        if std_a is None:
            std_a = self.std_a

        if std_b is None:
            std_b = self.std_b

        if std_a is None or std_b is None:
            raise ValueError("std_a and std_b have to be set")

        # pad the matrices
        A = pad_matrix(A, horizontally=self.p)
        B = pad_matrix(B, vertically=self.p)

        # compute the required standard deviation for the random matrices
        std = self._compute_required_std((t, s, r), std_a, std_b, rel_leakage)

        # execute the algorithm
        C = self._secure_matdot(A, B, std=std)

        # remove the padding
        C = np.array(C)[:t, :r]
        return C

    def _secure_matdot(self, A: np.ndarray, B: np.ndarray, *, std: float) -> np.ndarray:
        """Perform the secure MatDot algorithm for matrices with floating point entries"""

        # encode A
        A_encoded = self.encode_A(A, std)

        # encode B
        B_encoded = self.encode_B(B, std)

        # multiply at servers
        fastest_responses = multiply_at_servers(
            A_encoded, B_encoded, self.urls, num_responses=self.K
        )

        # fastest responses and the associated alphas
        server_ids, C_encoded = zip(*fastest_responses)
        alphas = self.alphas[list(server_ids)]

        # interpolate using the results
        G = np.linalg.inv(np.vander(alphas, increasing=True))[self.p - 1, :]
        C = sum(Ct * g for g, Ct in zip(G, C_encoded))

        return C
