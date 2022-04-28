"""Simple Flask server for SDMM computations"""

import argparse
import json

from flask import Flask, request

import sdmm.utils

app = Flask(__name__)


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

    A = sdmm.utils.deserialize_np_array(dict(data["A"]))
    B = sdmm.utils.deserialize_np_array(dict(data["B"]))

    # perform multiplication
    C = sdmm.utils.fast_multiply(A, B)

    # encode the result and send it
    CE = sdmm.utils.serialize_np_array(C)

    return json.dumps(CE)


@app.route("/slow_multiply", methods=["POST"])
def slow_multiply():
    """Multiply the matrices and return the product

    Takes input as b64 encoded numpy arrays
    """

    # parse input
    data = request.get_json()

    A = sdmm.utils.deserialize_np_array(dict(data["A"]))
    B = sdmm.utils.deserialize_np_array(dict(data["B"]))

    # perform multiplication
    C = sdmm.utils.slow_multiply(A, B)

    # encode the result and send it
    CE = sdmm.utils.serialize_np_array(C)

    return json.dumps(CE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Serve SDMM multiplication server")
    parser.add_argument("--port", default=5000, help="Port to run server on")
    parser.add_argument("--processes", default=1, help="Number of processes to use")
    args = parser.parse_args()

    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=False,
        processes=int(args.processes),
        threaded=False,
    )
