import json
from math import log2

import galois
import numpy as np
import pytest

from sdmm.utils.serialization import deserialize_np_array, serialize_np_array


def test_serialization_keys_floating_point_real():

    A = np.random.rand(10, 10)
    d = serialize_np_array(A)
    assert "data" in d
    assert "shape" in d
    assert "dtype" in d
    assert "order" not in d


def test_serialization_keys_floating_point_complex():

    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    d = serialize_np_array(A)
    assert "data" in d
    assert "shape" in d
    assert "dtype" in d
    assert "order" not in d


def test_serialization_keys_finite_field():

    F = galois.GF(19)
    B = F.Random((10, 10))
    d = serialize_np_array(B)

    assert "data" in d
    assert "shape" in d
    assert "dtype" in d
    assert "order" in d


def test_serialization_size():

    A = np.random.rand(100, 100)

    ser = json.dumps(serialize_np_array(A))

    # num_elements_in_matrix * num_bytes_per_element * num_bits_in_byte / num_bits_used_in_byte + overhead
    assert len(ser) <= 100 * 100 * 8 * 8 / log2(64) + 200


def test_serialization_floating_point_real():

    A = np.random.rand(10, 10)

    d = serialize_np_array(A)
    d = json.loads(json.dumps(d))

    assert (A == deserialize_np_array(d)).all()


def test_serialization_floating_point_complex():

    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)

    d = serialize_np_array(A)
    d = json.loads(json.dumps(d))

    assert (A == deserialize_np_array(d)).all()


def test_serialization_finite_field():

    F = galois.Field(19)

    A = F.Random((10, 10))

    d = serialize_np_array(A)
    d = json.loads(json.dumps(d))

    assert (A == deserialize_np_array(d)).all()
