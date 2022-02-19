"""Provides an implementation of the secure MatDot code"""
from typing import Tuple, Optional, overload, List

import tempfile
import json
from math import log, sqrt, pi

import numpy as np

import aiohttp
import asyncio
import requests

from utils import get_urls

def complex_normal(loc: Optional[float]=0.0, scale: Optional[float]=1.0, size: Optional[Tuple[int, ...]]=None) -> np.ndarray:
    return np.random.normal(loc=loc.real, scale=scale/2.0, size=size) + 1j*np.random.normal(loc=loc.imag, scale=scale/2.0, size=size)

def vandermonde_matrix(x: np.ndarray, m: Optional[int]=None) -> np.ndarray:
    n = len(x)
    if m is None:
        m = n
    return np.array([[x[i]**j for j in range(m)] for i in range(n)])

@overload
def partition_matrix(M: np.ndarray, *, horizontally: int) -> List[np.ndarray]:
    pass


@overload
def partition_matrix(M: np.ndarray, *, vertically: int) -> List[np.ndarray]:
    pass


@overload
def partition_matrix(
    M: np.ndarray, *, horizontally: int, vertically: int
) -> List[List[np.ndarray]]:
    pass


def partition_matrix(M, *, horizontally=None, vertically=None):
    n, m = M.shape
    if horizontally is not None and vertically is None:
        # split horizontally
        if m % horizontally != 0:
            raise ValueError("can't be evenly split")
        ms = m // horizontally
        return [M[:, i * ms : (i + 1) * ms] for i in range(horizontally)]
    if horizontally is None and vertically is not None:
        # split vertically
        if n % vertically != 0:
            raise ValueError("can't be evenly split")
        ns = n // vertically
        return [M[i * ns : (i + 1) * ns, :] for i in range(vertically)]
    if horizontally is not None and vertically is not None:
        # split both
        if n % vertically != 0 or m % horizontally != 0:
            raise ValueError("can't be evenly split")
        ms = m // horizontally
        ns = n // vertically
        return [
            [M[i * ns : (i + 1) * ns, j * ms : (j + 1) * ms] for j in range(vertically)]
            for i in range(vertically)
        ]
    raise ValueError("matrix must be split either horizontally or vertically (or both)")


async def multiply_at_server(session, base_url, A, B, sid):

    print(sid, "starting")

    url = base_url + "/multiply"

    data = aiohttp.FormData()

    file = tempfile.SpooledTemporaryFile()
    np.savez(file, A, B)
    file.seek(0)
    data.add_field("file", file)

    data.add_field("json", json.dumps({}))

    print(sid, "sending")

    async with session.post(url, data=data) as res:
        file.close()
        print(sid, "status:", res.status)
        if res.status != 200:
            return None
        with tempfile.TemporaryFile() as file:
            file.write(await res.content.read())
            file.seek(0)
            C = np.load(file)["arr_0"]

    print(sid, "returning")

    return (C, sid)

async def multiply_at_server_requests(base_url, A, B, sid):

    print(sid, "starting")

    url = base_url + "/multiply"

    file = tempfile.TemporaryFile()
    np.savez(file, A, B)
    file.seek(0)
    files = {"file": ("file", file, "application/octet-stream")}
    data = {}

    res = requests.post(url, data=data, files=files)
    file.close()
    print(sid, "status:", res.status_code)
    if res.status_code != 200:
        return None
    with tempfile.TemporaryFile() as file:
        file.write(res.content)
        file.seek(0)
        C = np.load(file)["arr_0"]

    print(sid, "returning")

    return (C, sid)


async def async_matdot_floating_point(A, B, p, X, N, sigmaa, sigmab, delta):

    vara = sigmaa**2
    varb = sigmab**2

    t, s = A.shape
    s, r = B.shape

    # compute recovery threshold
    Rc = 2 * p + 2 * X - 1

    # choose evaluation points
    x = np.exp([2j*pi*n / N for n in range(N)])

    # compute required standard deviation for random parts
    V = vandermonde_matrix(x[:X], p)
    U = np.diag(x[:X]**p) @ vandermonde_matrix(x[:X], X)
    M = np.linalg.inv(U) @ V
    M = M.conjugate().transpose() @ M
    varr = vara*M.trace().real / (p*delta)
    sigmar = sqrt(varr)

    V = vandermonde_matrix(x[:X], p)
    U = np.diag(x[:X]**p) @ vandermonde_matrix(x[:X], X)
    M = np.linalg.inv(U) @ V
    M = M.conjugate().transpose() @ M
    vars = varb*M.trace().real / (p*delta)
    sigmas = sqrt(vars)

    # encode A
    AP = partition_matrix(A, horizontally=p)
    R = [complex_normal(scale=sigmar, size=(t, s // p)) for _ in range(X)]
    APR = AP + R
    AT = iter(sum(a*x[n]**i for i, a in enumerate(APR)) for n in range(N))

    # encode B
    BP = partition_matrix(B, vertically=p)
    S = [complex_normal(scale=sigmas, size=(s // p, r)) for _ in range(X)]
    BPS = list(reversed(BP)) + S
    BT = iter(sum(b*x[n]**i for i, b in enumerate(BPS)) for n in range(N))

    # multiply at servers
    async with aiohttp.ClientSession() as session:

        tasks = []

        for i, base_url, At, Bt in zip(range(N), get_urls(N), AT, BT):
            tasks.append(
                asyncio.ensure_future(
                    multiply_at_server(session, base_url, At, Bt, i)
                )
            )

        fastest_responses = []

        for res in asyncio.as_completed(tasks):
            result = await res
            if result is None:
                continue
            fastest_responses.append(result)
            if len(fastest_responses) == Rc:
                break

        for task in tasks:
            task.cancel()

    if len(fastest_responses) < Rc:
        raise Exception("didn't get enough responses")

    # fastest responses
    CT, sids = zip(*fastest_responses)
    x = x[list(sids)]

    # interpolate using the results
    G = np.linalg.inv(vandermonde_matrix(x))[p-1, :]
    C = sum(Ct*g for g, Ct in zip(G, CT))

    return C


def matdot_floating_point(
    A, B, *, p, X, N, sigmaa, sigmab, delta
):
    """Perform the MatDot algorithm with the given parameters
    input
    -----
    A: np.ndarray of size t \times s
    B: np.ndarray of size s \times r
    p: 

    """

    t, sA = A.shape
    sB, r = B.shape

    if sA != sB:
        raise ValueError("matrix dimensions don't match")
    s = sA

    # pad the common dimension to a multiple of p
    pad = (-s) % p
    A = np.pad(A, ((0, 0), (0, pad)), mode="constant")
    B = np.pad(B, ((0, pad), (0, 0)), mode="constant")

    loop = asyncio.get_event_loop()
    C = loop.run_until_complete(async_matdot_floating_point(A, B, p, X, N, sigmaa, sigmab, delta))

    # remove the padding
    C = np.array(C)[:t, :r]
    return C


# A simple test of the functionality
if __name__ == "__main__":

    p = 2
    s = 10
    M = 10

    A = np.random.normal(loc=0.0, scale=1.0, size=(s, s))
    B = np.random.normal(loc=0.0, scale=1.0, size=(s, s))

    C = matdot_floating_point(A, B, p=p, X=2, N=10, sigmaa=1.0, sigmab=1.0, delta=0.1)

    if (C == A @ B).all():
        print("correct product")
    else:
        print("wrong product")
