#!/bin/bash

echo "Installing nginx ingress controller..."

helm install ingress-nginx ingress-nginx/ingress-nginx

echo "Creating the flask service and ingress..."

kubectl apply -f ./kubernetes/linode-config.yml

echo "Done!"