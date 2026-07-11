import os
import sys
import json
from dotenv import load_dotenv

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Load .env file containing IP addresses
load_dotenv("ips.env")     

import numpy as np
from pyspark.sql import SparkSession
from functions import classic_kmeans_run

# Global Variables
K = 4
NUM_ITER = 50                 # Iteration for statistics
EPOCHS = 20                    # Training steps             
RUN_IDENTIFIER = "k_means_16c"  
NUM_PARTITIONS = 48

# Directory setup
os.makedirs("runs", exist_ok=True)

# Dynamic Output Paths
FINAL_RAW_CSV = f"runs/{RUN_IDENTIFIER}.csv"
FINAL_STATS_CSV = f"runs/stats_{RUN_IDENTIFIER}.csv"

# Updated Path pointing to the dense parquet
DATASET_PATH = "data/rcv1_dataset"

if __name__ == "__main__":
    
    # Verify existence of required files
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Please run generate_data.py first.")

    #Creates spark session with run_script.sh specs
    spark = SparkSession.builder \
            .appName(f"Classic_Kmeans_{RUN_IDENTIFIER}") \
            .config("spark.api.mode", " classic") \
            .getOrCreate()

            
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading RCV1 Parquet Dataset...")
    
    # Read dataset
    df = spark.read.parquet(DATASET_PATH).repartition(NUM_PARTITIONS)
    # Added .count() to show exact dataset size
    total_docs = df.count()
    rdd_data = df.rdd.map(lambda row: np.array(row.features, dtype=np.float32))

    
    print(f"\n=========================================")
    print(f"Dataset Size: {total_docs} documents")
    print(f"Partitions  : {NUM_PARTITIONS}")
    print(f"Iterations  : {NUM_ITER}")
    print(f"Epochs      : {EPOCHS}")
    print(f"=========================================\n")
    
    # Execute the core final benchmark
    stats_df = classic_kmeans_run(
        rdd_data=rdd_data,
        K= K,
        epochs=EPOCHS,                       
        num_iter=NUM_ITER,
        raw_csv=FINAL_RAW_CSV,
        stats_csv=FINAL_STATS_CSV
    )
    
    print(f"\n=========================================")
    print(f"Classic K-Means Complete!")
    print(f"{total_docs} Documents Analyzed.")
    print(f"=========================================")

    spark.stop()
