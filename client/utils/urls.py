"""Provides the get_urls method that can (dynamically) give the URLs of the available servers"""


# TODO: make this be dynamic

SERVER_URL = "http://sdmm.server"


def get_urls(N, *, X=None):
    """Returns the URLs to N servers such that X-collusion is satisfied"""
    return [SERVER_URL] * N
