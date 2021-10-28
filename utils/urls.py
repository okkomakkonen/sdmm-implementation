"""Proveides the get_urls method that can (dynamically) give the URLS of the available servers"""


# TODO: make this be dynamic
SERVER_URL = "http://127.0.0.1:5000"

def get_urls(N, *, X=None):
    """Returns the URLs to N servers such that X-collusion is satisfied"""
    return [SERVER_URL] * N