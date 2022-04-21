import json
from multiprocessing import Pool
from typing import Iterator, List, Optional, Tuple

import numpy as np
import requests

from sdmm.utils.serialization import serialize_np_array, deserialize_np_array


def multiply_at_server(
    A: np.ndarray, B: np.ndarray, url: str, order: Optional[int] = None
) -> np.ndarray:
    """Compute the product of A and B using a helper server"""

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
    A_encoded: Iterator[np.ndarray],
    B_encoded: Iterator[np.ndarray],
    urls: List[str],
    *,
    num_responses: Optional[int] = None,
) -> List[Tuple[int, np.ndarray]]:
    """Compute the matrix products in the lists using the help of servers.
    Returns the result from the fastest num_responses if set.

    """

    num_servers = len(urls)

    if num_responses is None:
        num_responses = num_servers

    if num_servers <= 0 or num_responses <= 0:
        raise ValueError("Number of servers has to be positive")

    with Pool(num_servers) as p:
        res = list(
            enumerate(
                p.starmap(multiply_at_server, zip(A_encoded, B_encoded, urls)),
            )
        )

    return res[:num_responses]
