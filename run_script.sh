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

#=============================================
#CLUSTER SPECIFIC
#=============================================

if [ -n "$WORKER_IPS" ]; then

    echo "================================================="
    echo "[INFO] CLUSTER MODE DETECTED."
    echo "       Creating SparkSession with run_cluster.sh specs..."
    echo "       Submitting to Master: $MASTER_IP"
    echo "================================================="
        
    #Parameters for cluster deployment
    # --executor-memory 2G: Every executor has 2GB of memory dedicated to JVM for spark
    # --conf spark.executor.memoryOverhead=1024: Gives 1GB to python operations
    # --conf spark.memory.fraction=0.8: Tells the JVMs to use 80% of their memory for spark engine
    # --conf spark.memory.storageFraction=0.7: Of that 80%, use 70% to cache and store the data

spark-submit \
  --master spark://$MASTER_IP:7077 \
  --driver-memory 2G \
  --executor-memory 5G \
  --conf spark.executor.memoryOverhead=2048 \
  --conf spark.memory.fraction=0.8 \
  --conf spark.memory.storageFraction=0.7 \
  --total-executor-cores 16 \
  --executor-cores 4 \
  --conf spark.default.parallelism=32 \
  --py-files functions.py \
  $TARGET_SCRIPT

#=============================================
#LOCAL ENVIROMENT SPECIFIC
#=============================================

else

    echo "================================================="
    echo "[INFO] LOCAL ENVIRONMENT DETECTED."
    echo "       Using fallback configuration (local[*])."
    echo "-------------------------------------------------"
    echo "[WARN] Are you actually on a cluster? Don't worry!"
    echo "       Make sure ips.env is loaded with the Worker IPs."
    echo "       Check instructions.md if you have any doubts."
    echo "================================================="
  
    # Local execution parameters
    MASTER_URL="local[8]"
    
    # No need to define executors since it uses the RAM of our single machine
    spark-submit \
      --master $MASTER_URL \
      --driver-memory 4G \
      --py-files functions.py \
      $TARGET_SCRIPT
fi
