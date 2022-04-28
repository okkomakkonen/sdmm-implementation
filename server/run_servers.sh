#!/usr/bin/bash

start_server_instance() {

    python server.py --port $1 &

}

first=5000
last=`expr $first + $1`

for i in $(seq $first 1 $last)
do
    start_server_instance $i
done

# kill all processes when terminating
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

read -r -d '' _ </dev/tty