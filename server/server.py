"""Simple Flask server for SDMM computations"""

import tempfile
import json
import os
import random

from flask import Flask, request, send_file, make_response, abort
from galois import GF
import numpy as np

app = Flask(__name__)


@app.route("/")
def main():
    """Status indicator"""
    return "Hello, world!"

@app.route("/ping")
def ping():
    """Return the current node name"""
    return os.uname()[1]

@app.route("/test", methods=["GET", "POST"])
def test():

    if request.method == "GET":
        C = np.random.rand(3, 3)

        file = tempfile.TemporaryFile()
        np.savez_compressed(file, C)
        file.seek(0)
        resp = make_response(send_file(file, mimetype="application/octet-stream"))
        return resp

    if request.method == "POST":

        print(request.get_data())

        return "Hello"


@app.route("/multiply", methods=["POST"])
def multiply():
    """Takes two matrices and returns their product
    
    if field_size is defined, then converts the matrices to GF(field_size) elements and computes the product
    if not defined then computes the product of the numpy arrays
    """

    # read parameters and data
    r = random.randint(100, 999)

    print(r, "starting")
    params = json.loads(request.form.get("json", "{}"))

    data = request.files["file"]

    print(r, "before loading array")
    npz = np.load(data)
    A = npz["arr_0"]
    B = npz["arr_1"]
    print(r, "after loading array")

    if "field_size" in params:
        q = params["field_size"]
        FF = GF(q)
        A = FF(A)
        B = FF(B)

    print(r, "starting multiplication")

    C = A @ B

    print(r, "finish multiplication")

    file = tempfile.TemporaryFile()
    np.savez_compressed(file, C)
    file.seek(0)

    print(r, "sending file")

    return send_file(file, mimetype="application/octet-stream")

@app.route("/fake_multiply", methods=["POST"])
def fake_multiply():
    """Takes two matrices and returns their product
    
    if field_size is defined, then converts the matrices to GF(field_size) elements and computes the product
    if not defined then computes the product of the numpy arrays
    """

    # read parameters and data
    params = json.loads(request.form.get("json", "{}"))

    data = request.files["file"]

    npz = np.load(data)
    A = npz["arr_0"]
    B = npz["arr_1"]

    if "field_size" in params:
        q = params["field_size"]
        FF = GF(q)
        A = FF(A)
        B = FF(B)

    file = tempfile.TemporaryFile()
    np.savez_compressed(file, A)
    file.seek(0)

    return send_file(file, mimetype="application/octet-stream")


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
