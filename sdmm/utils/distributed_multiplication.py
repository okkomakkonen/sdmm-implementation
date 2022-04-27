import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, List, Optional, Tuple

import numpy as np
import requests

from sdmm.utils.matrix_utilities import fake_multiply
from sdmm.utils.serialization import serialize_np_array, deserialize_np_array


def multiply_at_server(
    A: np.ndarray, B: np.ndarray, url: str, order: Optional[int] = None
) -> np.ndarray:
    """Compute the product of A and B using a helper server"""

    # shortcircuits the computation if we want to compute locally
    if url == "fake":
        return fake_multiply(A, B)

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

    with ThreadPoolExecutor(max_workers=num_servers) as executor:

        futures = dict()

        for i, (url, AP, BP) in enumerate(zip(urls, A_encoded, B_encoded)):

            future = executor.submit(
                multiply_at_server, AP, BP, url
            )
            futures[future] = i

        fastest_responses = []

        for future in as_completed(futures):
            i = futures[future]
            try:
                res = future.result()
                fastest_responses.append((i, res))
            except Exception:
                pass

            if len(fastest_responses) == num_responses:
                break

        for future in futures:
            future.cancel()

    return fastest_responses
