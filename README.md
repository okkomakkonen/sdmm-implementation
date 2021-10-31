# SDMM implementation

Implementations of SDMM protocols. The `client` directory contains the necessary client side package and the `server` contains an implementation of the server side functionality.

## Things that need to be done

- `get_urls` should return a list of URLs of servers that are available
  - currently returns just the local address
- Add cooperative MatDot code
  - both encryption version and X-collusion based
- Add MatDot over reals