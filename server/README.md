# SDMM server

The SDMM server is a basic API for multiplying matrices with the help of helper servers.

## Running the server

The server can be run using

```bash
python server.py --port 5000 --processes 25
```

The `--port` argument specifies the port to run on and `--processes` specifies how many processes can be used to handle the requests.

Alternatively, multiple server instances can be run with

```bash
bash run_servers.sh 25
```

where the argument is the number of instances to run on ports 5000, 5001, etc.
