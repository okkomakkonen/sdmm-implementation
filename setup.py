from setuptools import setup

setup(
    name="sdmm",
    version="0.0.1",
    description="Secure Distributed Matrix Multiplication (SDMM) client side functionality",
    author="Okko Makkonen",
    author_email="okko.makkonen@aalto.fi",
    packages=["sdmm", "sdmm.schemes", "sdmm.utils"],
    install_requires=["galois", "numpy", "pycryptodome", "requests"],
)
