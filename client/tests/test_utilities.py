from tabnanny import check
import numpy as np
import pytest

from utils.utilities import (
    pad_matrix,
    check_conformable_and_compute_shapes,
    MatricesNotConformableException,
    partition_matrix,
)


def test_check_conformable_and_compute_shapes():

    with pytest.raises(MatricesNotConformableException):
        A = np.zeros((1, 1))
        B = np.zeros((20, 20))
        check_conformable_and_compute_shapes(A, B)

    A = np.zeros((1, 2))
    B = np.zeros((2, 3))

    assert check_conformable_and_compute_shapes(A, B) == (1, 2, 3)


def test_partitioning():

    A = np.random.rand(10, 10)

    part = partition_matrix(A, horizontally=5)

    assert isinstance(part, list) and len(part) == 5
    assert (np.block(part) == A).all()

    part = partition_matrix(A, vertically=5)

    assert isinstance(part, list) and len(part) == 5
    assert (np.block([[p] for p in part]) == A).all()

    part = partition_matrix(A, horizontally=5, vertically=5)

    assert (
        isinstance(part, list)
        and len(part) == 5
        and all(isinstance(p, list) and len(p) == 5 for p in part)
    )
    assert (np.block(part) == A).all()


def test_pad_matrices():

    A = np.random.rand(10, 10)

    AP = pad_matrix(A, horizontally=13)
    assert AP.shape == (10, 13)
    assert (AP[:, :10] == A).all()
    assert (AP[:, 10:] == 0 * AP[:, 10:]).all()
