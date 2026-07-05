#!/bin/bash

# Verify if a Python file was provided as argument
if [ -z "$1" ]; then
    echo "ERROR: No Python file specified."
    echo "EXAMPLE: ./run_cluster.sh main.py"
    exit 1
fi

TARGET_SCRIPT=$1

# Define your worker IPs (comma-separated and no spaces)
export WORKER_IPS="worker_1_ip, worker_2_ip, ..., worker_n_ip"
MASTER_IP="master_ip"

#Activate dedicated python enviroment
source pyvenv/bin/activate

# Launch the spark-submit command applied to the target script. Change specs based on resources and run objective
spark-submit \
  --master spark://$MASTER_IP:7077 \
  --executor-memory 2G \
  --total-executor-cores 6 \
  --executor-cores 2 \
  --py-files functions.py \
  $TARGET_SCRIPT
