#!/bin/bash

# Verify if a Python file was provided as argument
if [ -z "$1" ]; then
    echo "ERROR: No Python file specified."
    echo "EXAMPLE: ./run_cluster.sh main.py"
    exit 1
fi

TARGET_SCRIPT=$1

# Load IPs from ips.env file
if [ -f ips.env ]; then
    echo "Loading IPs from ips.env..."
    set -a         
    source ips.env    
    set +a         

fi

#Activate dedicated python enviroment in case it is not already activated
source pyvenv/bin/activate

# Launch the spark-submit command applied to the target script. Change specs based on resources and run objective
NUM_PARTITIONS=12

spark-submit \
  --master spark://$MASTER_IP:7077 \
  --executor-memory 1G \
  --total-executor-cores 6 \
  --executor-cores 2 \
  --conf spark.default.parallelism=12 \
  --py-files functions.py \
  $TARGET_SCRIPT
