"""Provides an implementation of the secure MatDot code for finite fields."""
from typing import List, Optional, Tuple

import galois  # type: ignore
import numpy as np

from sdmm.utils import (check_conformable_and_compute_shapes,
                        multiply_at_servers, pad_matrix, partition_matrix,
                        safe_random_matrix)


class MatDotFiniteField:
    def __init__(
        self,
        *,
        num_partitions: int,
        num_colluding: int,
        num_servers: int,
        urls: List[str],
        field_size: int,
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

        if field_size < self.N + 1:
            raise ValueError("field size is too small")

        self.field = galois.Field(field_size)

        # precompute the evaluation points that we will use
        self.alphas = self.field(list(range(1, self.N + 1)))

    def encode_A(self, A: np.ndarray) -> List[np.ndarray]:
        """Encode the matrix A"""

        AP = partition_matrix(A, horizontally=self.p)
        shape = AP[0].shape
        R = [safe_random_matrix(self.field, shape=shape) for _ in range(self.X)]
        APR = AP + R
        AT = iter(
            sum((a * x**i for i, a in enumerate(APR)), start=self.field(0))
            for x in self.alphas
        )
        return AT

    def encode_B(self, B: np.ndarray) -> List[np.ndarray]:
        """Encode the matrix B"""

        BP = partition_matrix(B, vertically=self.p)
        shape = BP[0].shape
        S = [safe_random_matrix(self.field, shape=shape) for _ in range(self.X)]
        BPS = list(reversed(BP)) + S
        BT = iter(
            sum((b * x**i for i, b in enumerate(BPS)), start=self.field(0))
            for x in self.alphas
        )
        return BT

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the matrix product of A and B using secure MatDot for floating point numbers"""

        # check that the matrices are the right sizes
        t, s, r = check_conformable_and_compute_shapes(A, B)

        # pad the matrices
        A = pad_matrix(A, horizontally=self.p)
        B = pad_matrix(B, vertically=self.p)

        # execute the algorithm
        C = self._secure_matdot(A, B)

        # remove the padding
        C = np.array(C)[:t, :r]
        return C

    def _secure_matdot(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Perform the secure MatDot algorithm for matrices with floating point entries"""

        # encode A
        A_encoded = self.encode_A(A)

        # encode B
        B_encoded = self.encode_B(B)

        # multiply at servers
        fastest_responses = multiply_at_servers(
            A_encoded, B_encoded, self.urls, num_responses=self.K
        )

        # fastest responses and the associated alphas
        server_ids, C_encoded = zip(*fastest_responses)
        alphas = self.alphas[list(server_ids)]

        # interpolate using the results
        G = np.linalg.inv(
            self.field(np.array([[x**i for i in range(len(alphas))] for x in alphas]))
        )[self.p - 1, :]
        C = sum((Ct * g for g, Ct in zip(G, C_encoded)), start=self.field(0))

        return C
