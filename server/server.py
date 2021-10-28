"""Simple Flask server for SDMM computations"""

import tempfile
import json

from flask import Flask, request, send_file
from galois import GF
import numpy as np

app = Flask(__name__)


@app.route("/")
def main():
    """Status indicator"""
    return "Hello, world!"


@app.route("/multiply", methods=["POST"])
def multiply():
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

    C = A @ B

    file = tempfile.TemporaryFile()
    np.savez_compressed(file, C)
    file.seek(0)

    return send_file(file, mimetype="application/octet-stream")


if __name__ == "__main__":

    app.run("0.0.0.0", 5000)
