#!/bin/bash

echo "Deleting nginx ingress controller..."

kubectl delete service nginx-ingress-controller

echo "Deleting flask service and ingress..."

kubectl delete -f kubernetes/linode-config.yml

echo "Done!"