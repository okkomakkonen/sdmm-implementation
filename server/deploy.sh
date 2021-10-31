#!/bin/bash

echo "Creating the flask deployment and service..."

kubectl create -f ./kubernetes/config.yml

echo "Adding the ingress..."

minikube addons enable ingress
kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission
kubectl apply -f ./kubernetes/minikube-ingress.yml

echo "Done!"