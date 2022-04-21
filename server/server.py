"""Simple Flask server for SDMM computations"""

import json
import base64

from flask import Flask, request
from galois import GF
import numpy as np

app = Flask(__name__)


def serialize_np_array(A):

    data = A.tobytes()
    shape = A.shape
    dtype = A.dtype.name

    return {
        "data": base64.b64encode(data).decode("utf-8"),
        "shape": shape,
        "dtype": dtype,
    }


def deserialize_np_array(d):

    dtype = np.dtype(d["dtype"])
    data = base64.b64decode(d["data"])
    return np.reshape(np.frombuffer(data, dtype=dtype), d["shape"])


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

    app.run(host="0.0.0.0", port=5000)
