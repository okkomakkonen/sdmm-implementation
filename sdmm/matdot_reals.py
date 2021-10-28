"""Provides an implementation of the secure MatDot code"""
from typing import Tuple, Optional

import tempfile
import json

import numpy as np
from galois import GF

import aiohttp
import asyncio

from utils import complex_random_matrix, partition_matrix, get_urls


def encode_A(A, p, X, ev, varr):

    assert A.shape[1] % p == 0, "p doesn't divide second dimension of A"

    A_enc_shape = (A.shape[0], A.shape[1] // p)
    AP = partition_matrix(A, 1, p)

    R = [complex_random_matrix(varr, A_enc_shape) for _ in range(X)]

    AT = (
        sum(AP[i] * e ** i for i in range(p))
        + sum(R[i] * e ** (i + p) for i in range(X))
        for e in ev
    )

    return AT


def encode_B(B, p, X, ev, vars) -> None:

    assert B.shape[0] % p == 0, "p doesn't divide first dimension of B"

    B_enc_shape = (B.shape[0] // p, B.shape[1])
    BP = partition_matrix(B, p, 1)

    S = [complex_random_matrix(vars, B_enc_shape) for _ in range(X)]

    BT = (
        sum(BP[i] * e ** (p - 1 - i) for i in range(p))
        + sum(S[i] * e ** (i + p) for i in range(X))
        for e in ev
    )

    return BT


async def multiply_at_server(session, base_url, A, B, sid):

    url = base_url + "/multiply"

    data = aiohttp.FormData()

    file = tempfile.TemporaryFile()
    np.savez_compressed(file, A, B)
    file.seek(0)
    data.add_field("file", file)

    async with session.post(url, data=data) as res:
        with tempfile.TemporaryFile() as file:
            file.write(await res.content.read())
            file.seek(0)
            C = np.load(file)["arr_0"]

    return (C, sid)


def interpolation_matrix(ev, p):
    """Return the interpolation matrix that is used to decode the result"""

    V = np.array([[e ** i for i in range(len(ev))] for e in ev])
    return np.linalg.inv(V)[p - 1, :]


def interpolate(ev, Z, p):
    """Return the product of A and B given the results from the servers"""
    inter = interpolation_matrix(ev, p)
    return sum(i * z for i, z in zip(inter, Z))


async def matdot_finite_field(A, B, p, X, N, sigmar, sigmas):

    # check that parameters are valid
    _, sA = A.shape
    sB, _ = B.shape

    if sA != sB:
        raise ValueError("matrix dimensions don't match")
    s = sA

    if s % p != 0:
        raise ValueError("matrix can't be evenly split")

    Rc = 2 * p + 2 * X - 1
    if N < Rc:
        raise ValueError("not enough servers")

    # choose evaluation points
    ev = np.exp(1j * 2 * np.pi * np.arange(N) / N)

    AT = iter(encode_A(A, p, X, ev, sigmar))
    BT = iter(encode_B(B, p, X, ev, sigmas))

    try:
        urls = get_urls(N, X)
    except RuntimeError as e:
        raise RuntimeError("couldn't find enough servers") from e

    async with aiohttp.ClientSession() as session:

        tasks = []

        for i, base_url, At, Bt in zip(range(N), urls, AT, BT):
            tasks.append(
                asyncio.ensure_future(multiply_at_server(session, base_url, At, Bt, i))
            )

        fastest_responses = []

        for res in asyncio.as_completed(tasks):
            fastest_responses.append(await res)
            if len(fastest_responses) == Rc:
                break

    R, sids = zip(*fastest_responses)
    ev = ev[list(sids)]

    C = interpolate(ev, R, p)

    return C


def matdot_reals(A, B, *, p, X, N, sigmaA, sigmaB, delta):

    # compute sigmar, sigmas

    loop = asyncio.get_event_loop()
    C = loop.run_until_complete(matdot_finite_field(A, B, q, p, X, N, varr, vars))
    return C


# A simple test of the functionality
if __name__ == "__main__":

    p = 2
    s = 10
    M = 10
    q = 32749

    A = np.random.normal(size=(s, s))
    B = np.random.normal(size=(s, s))

    C = matdot_reals(A, B, p=p, q=q, X=2, N=10, sigmaA=1, sigmaB=1, delta=0.1)

    if (C == A @ B).all():
        print("correct product")
    else:
        print("wrong product")
