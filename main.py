import os
import sys
import time

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from functions import mini_batch_run

# Global Variables
N_ITER = 20                 # Iteration for statistics
RUN_IDENTIFIER = "change for every run with specifics to facilitate understanding" 

# Directory setup
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Dynamic Output Paths
FINAL_RAW_CSV = os.path.join(DATA_DIR, f"raw_{RUN_IDENTIFIER}.csv")
FINAL_STATS_CSV = os.path.join(DATA_DIR, f"stats_{RUN_IDENTIFIER}.csv")
PARAMS_CSV = os.path.join(DATA_DIR, "best_params.csv")

# Updated Path pointing to the dense parquet
PARQUET_PATH = os.path.join(DATA_DIR, "rcv1.parquet")

if __name__ == "__main__":
    
    # Verify existence of required upstream files
    if not os.path.exists(PARAMS_CSV):
        raise FileNotFoundError(f"Missing {PARAMS_CSV}. Please run initialization.py first.")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing {PARQUET_PATH}. Please run generate_data.py first.")
        
    # Load optimal parameters found during the initialization step
    params_df = pd.read_csv(PARAMS_CSV)
    best_k = int(params_df['best_k'].iloc[0])
    best_b = int(params_df['best_b'].iloc[0])

    # Instantiate Spark Session leveraging logical cores
    spark = SparkSession.builder \
        .appName(f"RCV1-MiniBatch-{RUN_IDENTIFIER}") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading full RCV1 Parquet Dataset...")
    
    # Read dataset
    df = spark.read.parquet(PARQUET_PATH)
    
    # Map rows into float32 arrays
    rdd_data = df.rdd.map(lambda row: np.array(row.features, dtype=np.float32))
    
    # Added .count() to show exact dataset size
    dataset_size = rdd_data.count()
    
    print(f"\n=========================================")
    print(f"Dataset Size      : {dataset_size} documents")
    print(f"Iterations : {N_ITER}")
    print(f"Loaded Optimal K  : {best_k}")
    print(f"Loaded Optimal b  : {best_b}")
    print(f"=========================================\n")
    
    # Execute the core final benchmark
    stats_df = mini_batch_run(
        rdd_data=rdd_data,
        best_k=best_k,
        best_b=best_b,
        n_iter=N_ITER,
        raw_csv=FINAL_RAW_CSV,
        stats_csv=FINAL_STATS_CSV
    )
    
    print(f"\n=========================================")
    print(f"Mini Batch Complete!")
    print(f"=========================================")

    spark.stop()