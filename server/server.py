"""Simple Flask server for SDMM computations"""

from typing import Dict, Any

import json
import base64
import argparse

from flask import Flask, request
import galois # type: ignore
import numpy as np # type: ignore

app = Flask(__name__)


def serialize_np_array(A: np.ndarray) -> Dict[str, Any]:

    data = A.tobytes()
    shape = A.shape
    dtype = A.dtype.name

    d = {
        "data": base64.b64encode(data).decode("utf-8"),
        "shape": shape,
        "dtype": dtype,
    }

    if hasattr(type(A), "order"):
        d["order"] = type(A).order

    return d


def deserialize_np_array(d: Dict[str, Any]) -> np.ndarray:

    dtype = np.dtype(d["dtype"])
    data = base64.b64decode(d["data"])
    shape = d["shape"]

    A = np.reshape(np.frombuffer(data, dtype=dtype), newshape=shape)

    if "order" in d:
        F = galois.GF(d["order"])
        A = F(A)

    return A

@app.route("/")
def main():
    """Status indicator"""
    return "Hello, world!"


@app.route("/multiply", methods=["POST"])
def multiply():
    """Multiply the matrices and return the product

    Takes input as b64 encoded numpy arrays
    """

    # parse input
    data = request.get_json()

    A = deserialize_np_array(dict(data["A"]))
    B = deserialize_np_array(dict(data["B"]))

    # perform multiplication
    C = A @ B

    # encode the result and send it
    CE = serialize_np_array(C)

    return json.dumps(CE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Serve SDMM multiplication server")
    parser.add_argument("--port", default=5000, help="Port to run server on")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
