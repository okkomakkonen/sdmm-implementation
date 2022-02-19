# SDMM server

The SDMM server is a basic API for multiplying matrices with the help of helper servers.

## Get the docker image

```bash
docker pull okkomakkenen/sdmm-server
```

## Run the server with docker locally

Start the server in a docker container

```bash
docker run -p 5000:5000 okkomakkonen/sdmm-server
```

## Deploy locally with minikube

Install minikube and set the vm-driver.

Start minikube with

```bash
minikube start
```

Deploy the cluster

```bash
sh deploy.sh
```

Configure the /etc/hosts file to use the ip address of the cluster

```bash
echo "$(minikube ip) sdmm.server" | sudo tee -a /etc/hosts
```

or edit directly with
```bash
sudo vim /etc/hosts
```

To delete the minikube cluster, first stop the server and then delete the cluster
```bash
minikube stop
minikube delete
```