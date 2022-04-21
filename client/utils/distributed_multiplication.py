import base64
import tempfile
import threading
import json
import base64
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
import requests


def serialize_np_array(A: np.ndarray):

    data = A.tobytes()
    shape = A.shape
    dtype = A.dtype.name

    return {
        "data": base64.b64encode(data).decode("utf-8"),
        "shape": shape,
        "dtype": dtype,
    }


def deserialize_np_array(d) -> np.ndarray:

    dtype = np.dtype(d["dtype"])
    data = base64.b64decode(d["data"])
    return np.reshape(np.frombuffer(data, dtype=dtype), d["shape"])


def multiply_at_server(A: np.ndarray, B: np.ndarray, url: str) -> np.ndarray:

    AE = serialize_np_array(A)
    BE = serialize_np_array(B)

    data = {"A": AE, "B": BE}

    res = requests.post(url, json=data)

    if res.status_code != 200:
        raise RuntimeError(f"Server returned with code {res.status_code}")

    CE = json.loads(res.text)
    C = deserialize_np_array(CE)

    return C


def multiply_at_servers(
    A_encoded: List[np.ndarray],
    B_encoded: List[np.ndarray],
    urls: List[str],
    *,
    num_responses: Optional[int] = None,
) -> List[np.ndarray]:

    num_servers = len(urls)

    if num_responses is None:
        num_responses = num_servers

    if num_servers <= 0 or num_responses <= 0:
        raise ValueError("Number of servers has to be positive")

    p = Pool(num_servers)
    res = list(
        zip(
            range(num_servers),
            p.starmap(multiply_at_server, zip(A_encoded, B_encoded, urls)),
        )
    )
    return res[:num_responses]
