"""Implements serialization for numpy arrays"""

from typing import Any, Dict
import numpy as np
import galois  # type: ignore
import base64


def serialize_np_array(A: np.ndarray) -> Dict[str, Any]:

    data = A.tobytes()
    shape = A.shape
    dtype = A.dtype.name

    d = {
        "data": base64.b85encode(data).decode("utf-8"),
        "shape": shape,
        "dtype": dtype,
    }

    if hasattr(type(A), "order"):
        d["order"] = type(A).order  # type: ignore

    return d


def deserialize_np_array(d: Dict[str, Any]) -> np.ndarray:

    dtype = np.dtype(d["dtype"])
    data = base64.b85decode(d["data"])
    shape = d["shape"]

    A = np.reshape(np.frombuffer(data, dtype=dtype), newshape=shape)

    if "order" in d:
        F = galois.GF(d["order"])
        A = F(A)

    return A
