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
    
    set -a         
    source ips.env    
    set +a         

fi

#Activate dedicated python enviroment in case it is not already activated
source pyvenv/bin/activate




# --executor-memory 2G: Every executor has 2GB of memory dedicated to JVM for spark
# --conf spark.executor.memoryOverhead=1024: Gives 1GB to python operations
# --conf spark.memory.fraction=0.8: Tells the JVMs to use 80% of their memory for spark engine
# --conf spark.memory.storageFraction=0.7: Of that 80%, use 70% to cache and store the data

# Launch the spark-submit command applied to the target script. Change specs based on resources and run objective
NUM_PARTITIONS=12

spark-submit \
  --master spark://$MASTER_IP:7077 \
  --driver-memory 2G \
  --executor-memory 2G \
  --conf spark.executor.memoryOverhead=1024 \
  --conf spark.memory.fraction=0.8 \
  --conf spark.memory.storageFraction=0.7 \
  --total-executor-cores 6 \
  --executor-cores 2 \
  --conf spark.default.parallelism=$NUM_PARTITIONS \
  --py-files functions.py \
  $TARGET_SCRIPT
