"""Provides an implementation of the secure MatDot code"""
from typing import Tuple, Optional

import tempfile
import json

import numpy as np
from galois import GF

import aiohttp
import asyncio

from utils import random_matrix, partition_matrix, get_urls

def encode_A(A, p, X, ev):

    FF = type(A)

    assert A.shape[1] % p == 0, "p doesn't divide second dimension of A"

    A_enc_shape = (A.shape[0], A.shape[1] // p)
    AP = partition_matrix(A, 1, p)

    R = [random_matrix(FF, A_enc_shape) for _ in range(X)]

    AT = (
        sum((AP[i] * e ** i for i in range(p)), start=FF(0))
        + sum((R[i] * e ** (i + p) for i in range(X)), start=FF(0))
        for e in ev
    )

    return AT

def encode_B(B, p, X, ev) -> None:

    FF = type(B)

    assert B.shape[0] % p == 0, "p doesn't divide first dimension of B"

    B_enc_shape = (B.shape[0] // p, B.shape[1])
    BP = partition_matrix(B, p, 1)

    S = [random_matrix(FF, B_enc_shape) for _ in range(X)]

    BT = (
        sum((BP[i] * e ** (p - 1 - i) for i in range(p)), start=FF(0))
        + sum((S[i] * e ** (i + p) for i in range(X)), start=FF(0))
        for e in ev
    )

    return BT

async def multiply_at_server(session, base_url, A, B, q, sid):

    url = base_url + "/multiply"

    FF = GF(q)

    data = aiohttp.FormData()

    file = tempfile.TemporaryFile()
    np.savez_compressed(file, A, B)
    file.seek(0)
    data.add_field("file", file)

    data.add_field("json", json.dumps({"field_size": q}))

    async with session.post(url, data=data) as res:
        with tempfile.TemporaryFile() as file:
            file.write(await res.content.read())
            file.seek(0)
            C = np.load(file)["arr_0"]
            C = FF(C)

    return (C, sid)

def interpolation_matrix(FF, ev, p):
    """Return the interpolation matrix that is used to decode the result"""

    V = FF(np.array([[e ** i for i in range(len(ev))] for e in ev]))
    return np.linalg.inv(V)[p - 1, :]

def interpolate(FF, ev, Z, p):
    """Return the product of A and B given the results from the servers"""
    inter = interpolation_matrix(FF, ev, p)
    return sum((i * z for i, z in zip(inter, Z)), start=FF(0))


async def matdot_finite_field(A, B, q, p, X, N):

    # check that parameters are valid
    if q < N + 1:
        raise ValueError("field size too small")
    
    _, sA = A.shape
    sB, _ = B.shape

    if sA != sB:
        raise ValueError("matrix dimensions don't match")
    s = sA

    if s % p != 0:
        raise ValueError("matrix can't be evenly split")

    Rc = 2*p + 2*X - 1
    if N < Rc:
        raise ValueError("not enough servers")

    # construct finite field
    try:
        FF = GF(q)
    except ValueError as e:
        raise ValueError("finite field size must be a prime power") from e

    # choose evaluation points
    ev = FF(list(range(1, N + 1)))

    AT = iter(encode_A(A, p, X, ev))
    BT = iter(encode_B(B, p, X, ev))

    async with aiohttp.ClientSession() as session:

        tasks = []

        for i, base_url, At, Bt in zip(range(N), get_urls(N), AT, BT):
            tasks.append(asyncio.ensure_future(multiply_at_server(session, base_url, At, Bt, q, i)))

        fastest_responses = []

        for res in asyncio.as_completed(tasks):
            fastest_responses.append(await res)
            if len(fastest_responses) == Rc:
                break

    R, sids = zip(*fastest_responses)
    ev = ev[list(sids)]

    C = interpolate(FF, ev, R, p)

    return C

def matdot(A, B, *, p, q, X, N, ):

    FF = GF(q)

    A = FF(A)
    B = FF(B)
    
    loop = asyncio.get_event_loop()
    C = loop.run_until_complete(matdot_finite_field(A, B, q, p, X, N))
    C = np.array(C)
    return C



# A simple test of the functionality
if __name__ == "__main__":

    p = 2
    s = 10
    M = 10
    q = 32749

    A = np.random.randint(M, size=(s, s))
    B = np.random.randint(M, size=(s, s))

    C = matdot(A, B, p=p, q=q, X=2, N=10)

    if (C == A@B).all():
        print("correct product")
    else:
        print("wrong product")
