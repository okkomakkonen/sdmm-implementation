# SDMM client

This contains the python package that computes matrix products using Secure Distributed Matrix Multiplication (SDMM) by connecting to helper servers.

## Setup

Install the package with

```bash
pip install .
```

## Demo

The demo can be run with

```bash
python demos/demo_floating_point.py
```

This requires that the server is running at `http://localhost:5000`.

## Testing

Run the test packages using pytest (`pip install pytest`) by running

```bash
pytest .
```

This requires that the server is running at `http://localhost:5000`.
