"""Provides an implementation of the secure MatDot code for floating point numbers."""
from math import log, sqrt, pi
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterator, List, Optional, Tuple

import numpy as np

from sdmm.utils import (
    complex_normal,
    partition_matrix,
    multiply_at_servers,
    multiply_at_server,
    pad_matrix,
    check_conformable_and_compute_shapes,
)
from sdmm.utils.matrix_utilities import fake_multiply


class MatDotFloatingPoint:
    """Implements the secure MatDot scheme for real numbers"""

    def __init__(
        self,
        *,
        num_partitions: int,
        num_colluding: int,
        urls: List[str],
        std_a: Optional[float] = None,
        std_b: Optional[float] = None,
        rel_leakage: Optional[float] = None,
        threaded: bool = False,
        slow_multiplication: bool = False
    ) -> None:

        num_servers = len(urls)

        if num_partitions <= 0 or num_colluding <= 0 or num_servers <= 0:
            raise ValueError("Number of servers and partitions has to be positive")

        self.p = num_partitions
        self.X = num_colluding
        self.N = num_servers
        self.K = 2 * self.p + 2 * self.X - 1

        if self.N < self.K:
            raise ValueError("Too few servers for SDMM")

        self.urls = [url + ("/slow_multiply" if slow_multiplication else "/multiply") for url in urls]
        self.threaded = threaded

        if rel_leakage is not None and rel_leakage <= 0:
            raise ValueError("Relative leakage has to be positive")

        if std_a is not None and std_a <= 0:
            raise ValueError("Standard deviations have to be positive")

        if std_b is not None and std_b <= 0:
            raise ValueError("Standard deviations have to be positive")

        self.rel_leakage = rel_leakage
        self.std_a = std_a
        self.std_b = std_b

        # precompute the evaluation points that we will use
        # TODO: do this in a numerically stable way
        self.alphas = np.exp([2j * pi * n / self.N for n in range(self.N)])

        self.trr: Optional[float] = None
        self.trs: Optional[float] = None

    def __repr__(self) -> str:
        return f"MatDotFloatingPoint(num_partitions={self.p}, num_colluding={self.X}, num_servers={self.N})"

    def __str__(self) -> str:
        return f"Secure MatDot code over floating point numbers with p = {self.p}, X = {self.X}, N = {self.N}"

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

        # TODO: finish this method
        return 0.0001

    def _partition_A(self, A: np.ndarray) -> List[np.ndarray]:
        """Partition the matrix A to p pieces horizontally"""

        return partition_matrix(A, horizontally=self.p)

    def _partition_B(self, B: np.ndarray) -> List[np.ndarray]:
        """Partition the matrix B to p pieces vertically"""

        return partition_matrix(B, vertically=self.p)

    def _draw_random_for_A(self, A: np.ndarray, std: float) -> List[np.ndarray]:
        """Draw the random matrices of given standard deviation associated to the matrix A"""

        t, s = A.shape
        size = (t, s // self.p)
        return [complex_normal(scale=std, size=size) for _ in range(self.X)]

    def _draw_random_for_B(self, B: np.ndarray, std: float) -> List[np.ndarray]:
        """Draw the random matrices of given standard deviation associated to the matrix B"""

        s, r = B.shape
        size = (s // self.p, r)
        return [complex_normal(scale=std, size=size) for _ in range(self.X)]

    def _encode_A_at(
        self, alpha: complex, AP: List[np.ndarray], R: List[np.ndarray]
    ) -> np.ndarray:
        """Compute the polynomial f(x) at alpha"""

        APR = AP + R
        return sum(a * alpha**i for i, a in enumerate(APR))

    def _encode_B_at(self, alpha: complex, BP: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Compute the polynomial g(x) at alpha"""

        BPS = list(reversed(BP)) + S
        return sum(b * alpha**i for i, b in enumerate(BPS))

    def _encode_A(self, A: np.ndarray, std: float) -> Iterator[np.ndarray]:
        """Encode the matrix A"""

        AP = self._partition_A(A)
        R = self._draw_random_for_A(A, std)
        AT = [self._encode_A_at(alpha, AP, R) for alpha in self.alphas]
        return AT

    def _encode_B(self, B: np.ndarray, std: float) -> Iterator[np.ndarray]:
        """Encode the matrix B"""

        BP = self._partition_B(B)
        S = self._draw_random_for_B(B, std)
        BT = [self._encode_B_at(alpha, BP, S) for alpha in self.alphas]
        return BT

    def _interpolate(self, responses: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Interpolates the product AB using the fastest responses"""

        server_ids, C_encoded = zip(*responses)
        alphas = self.alphas[list(server_ids)]

        # interpolate using the results
        G = np.linalg.inv(np.vander(alphas, increasing=True))[self.p - 1, :]
        C = sum(Ct * g for g, Ct in zip(G, C_encoded))

        return C

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

        # validate input
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

        # execute the algorithm either multithreaded or not
        if self.threaded:
            C = self._secure_matdot_threaded(A, B, std=std)
        else:
            C = self._secure_matdot(A, B, std=std)

        # remove the padding
        C = np.array(C)[:t, :r]
        return C

    def _secure_matdot(self, A: np.ndarray, B: np.ndarray, *, std: float) -> np.ndarray:
        """Perform the secure MatDot algorithm for matrices with floating point entries"""

        # encode A
        A_encoded = self._encode_A(A, std)

        # encode B
        B_encoded = self._encode_B(B, std)

        # multiply at servers
        fastest_responses = multiply_at_servers(
            A_encoded, B_encoded, self.urls, num_responses=self.K
        )

        # interpolate using the fastest results
        C = self._interpolate(fastest_responses)

        return C

    def _encode_and_multiply(self, AP, BP, R, S, alpha, url):

        At = self._encode_A_at(alpha, AP, R)
        Bt = self._encode_B_at(alpha, BP, S)

        return multiply_at_server(At, Bt, url)

    def _secure_matdot_threaded(
        self, A: np.ndarray, B: np.ndarray, *, std: float
    ) -> np.ndarray:
        """Perform the secure MatDot algorithm for matrices with floating point entries"""

        # partition the matrices
        AP = self._partition_A(A)
        BP = self._partition_B(B)

        # draw random matrices
        R = self._draw_random_for_A(A, std)
        S = self._draw_random_for_B(B, std)

        with ThreadPoolExecutor(max_workers=self.N) as executor:

            futures = dict()

            for url, alpha in zip(self.urls, self.alphas):

                future = executor.submit(
                    self._encode_and_multiply, AP, BP, R, S, alpha, url
                )
                futures[future] = alpha

            fastest_responses = []

            for future in as_completed(futures):
                alpha = futures[future]
                i = list(self.alphas).index(alpha)
                try:
                    res = future.result()
                    fastest_responses.append((i, res))
                except Exception:
                    pass

                if len(fastest_responses) == self.K:
                    break

            for future in futures:
                future.cancel()

        if len(fastest_responses) < self.K:
            raise RuntimeError("Did not receive enough responses")

        C = self._interpolate(fastest_responses)

        return C
