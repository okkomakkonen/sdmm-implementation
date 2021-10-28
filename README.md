# SDMM implementation

Implementations of SDMM protocols

## Plan

Write different files for different entities in the system, i.e. the client and the servers. These entities, or servers, interface with each other through the Kubernetes system.

### Client

- Takes two matrices and some parameters (number of servers, etc.)
- Outputs encoded matrices for each server
- (If cooperative) Outputs encryption keys for each server
  

### Server

- Takes two encoded matrices
- Multiplies the matrices
- (If cooperative) Encrypts product using key
- Outputs product (maybe encrypted) to either master server or client

### (If cooperative) Master server

- Takes encrypted products and interpolates using them
- Sends the result to client

### Client (again)

- Interpolates using the answers from each server
- (If cooperative) Interpolates using the encryption matrices and subtracts that from the results


## Running the demo

Start the server in a Docker instance

```bash
docker build -t sdmm-server server/
docker run -p 5000:5000 sdmm-server
```

or standalone
```bash
python server/server.py
```

Run the demo with

```bash
python demo.py
```

## Things that need to be done

- `get_urls` should return a list of URLs of servers that are available
  - currently returns just the local address
- Deploy with kubernetes