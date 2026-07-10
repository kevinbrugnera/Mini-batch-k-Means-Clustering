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
from functions import mini_batch_run

# Global Variables
K = 4
NUM_ITER = 50                 # Iteration for statistics
EPOCHS = 20                    # Training steps
SAMPLE_SIZE = 0               #Documents to analyze. Set to zero if you want toanalyze full dataset
RUN_IDENTIFIER = "thin_topology"  
NUM_PARTITIONS = 16            

# Directory setup
os.makedirs("runs", exist_ok=True)

# Dynamic Output Paths
FINAL_RAW_CSV = f"runs/{RUN_IDENTIFIER}.csv"
FINAL_STATS_CSV = f"runs/stats_{RUN_IDENTIFIER}.csv"

#Read initialization file with best parameters
PARAMS_JSON = "runs/b_search_metadata.json" 

# Updated Path pointing to the dense parquet
DATASET_PATH = "data/rcv1_dataset"

if __name__ == "__main__":
    
    # Verify existence of required files
    if not os.path.exists(PARAMS_JSON):
        raise FileNotFoundError(f"Missing {PARAMS_JSON}. Please run initialization.py first.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Please run generate_data.py first.")
        
    # Load optimal parameters found during the initialization step
    with open(PARAMS_JSON, "r") as f:
        init_metadata = json.load(f)

    #Extract value
    batch_size = int(init_metadata["metrics_summary"]["optimal_b_found"])

    #Creates spark session with run_script.sh specs
    spark = SparkSession.builder \
            .appName(f"MiniBatch_{RUN_IDENTIFIER}") \
            .getOrCreate()

            
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading RCV1 Parquet Dataset...")
    
    # Read dataset
    df = spark.read.parquet(DATASET_PATH).repartition(NUM_PARTITIONS)
    # Added .count() to show exact dataset size, trigger the repartition
    total_docs = df.count()
    partitions = df.rdd.getNumPartitions()

    if SAMPLE_SIZE == 0:

        # Analyze whole dataset
        # Map rows into float32 arrays
        rdd_data = df.rdd.map(lambda row: np.array(row.features, dtype=np.float32))
        dataset_size = total_docs
    
    else:

        fraction = min(1, SAMPLE_SIZE/total_docs)
        df_sample = df.sample(False, fraction, seed=58)
        rdd_data = df_sample.rdd.map(lambda row: np.array(row.features, dtype=np.float32))
        dataset_size = SAMPLE_SIZE
    
    
    print(f"\n=========================================")
    print(f"Dataset Size: {dataset_size} documents")
    print(f"Partitions  : {partitions}")
    print(f"Iterations  : {NUM_ITER}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch Size  : {batch_size}")
    print(f"=========================================\n")
    
    # Execute the core final benchmark
    stats_df = mini_batch_run(
        rdd_data=rdd_data,
        K= K,
        best_b=batch_size,
        epochs=EPOCHS,                       
        num_iter=NUM_ITER,
        raw_csv=FINAL_RAW_CSV,
        stats_csv=FINAL_STATS_CSV
    )
    
    print(f"\n=========================================")
    print(f"Mini Batch Complete!")
    print(f"{dataset_size} Documents Analyzed.")
    print(f"=========================================")

    spark.stop()
