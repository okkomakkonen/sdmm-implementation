# SDMM implementation

Implementations of SDMM protocols. The `sdmm` directory contains the necessary client side package and the `server` contains an implementation of the server side functionality.

## TODO

- Secure MatDot over finite fields does not work at the moment
- Only wait for the fastest responses instead of waiting for the slowest one to respond
- Add logging

## Setup

Install the package with

```bash
pip install .
```

or directly from GitHub with

```bash
pip install git+https://github.com/okkomakkonen/sdmm-implementation.git
```

## Demo

The demo can be run with

```bash
python demos/demo_floating_point.py
```

This requires that the server is running at `http://localhost:5000`.

## Testing

Run the test packages using `pytest` by running

```bash
pytest .
```

This requires that the server is running at `http://localhost:5000`.
