import os
import sys
import time

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pyspark.sql import SparkSession
from functions import k_search, b_search

# Global Variables
NUM_ITERATIONS = 15                 # Iterations for statistics
EPOCHS = 16                         # Training step for single k-means run
K_RANGE = list(range(10,11))        # K values to test
B_RANGE = [1000]                    # b values to test
SAMPLE_SIZE = 100                   # Sample of RCV1 dataset to analyze

# Directory setup
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Output Paths
K_RAW_CSV = os.path.join(DATA_DIR, "k_search_raw.csv")
K_STATS_CSV = os.path.join(DATA_DIR, "k_search_stats.csv")
B_RAW_CSV = os.path.join(DATA_DIR, "b_search_raw.csv")
B_STATS_CSV = os.path.join(DATA_DIR, "b_search_stats.csv")
PARAMS_CSV = os.path.join(DATA_DIR, "best_params.csv")

# Updated Path pointing to the dense parquet
PARQUET_PATH = os.path.join(DATA_DIR, "rcv1.parquet")

start = time.time()
if __name__ == "__main__":
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing {PARQUET_PATH}. Please run generate_data.py first.")

    spark = SparkSession.builder \
        .appName("RCV1-Initialization-Diagnostics") \
        .master("local[4]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading a sample of {SAMPLE_SIZE} dense documents from Parquet...")
    
    # Read the parquet directly. PyArrow saves it as an ArrayType of floats natively
    df_sample = spark.read.parquet(PARQUET_PATH).limit(SAMPLE_SIZE)
    
    # Map each row to a dense numpy array of float32 for fast computation
    rdd_sample = df_sample.rdd.map(lambda row: np.array(row.features, dtype=np.float32))

    # --- Phase 1: Optimal K ---
    print("\n[1/2] Search for Optimal K...")
    best_k, k_stats = k_search(
        rdd_sample, K_RANGE, EPOCHS, NUM_ITERATIONS, K_RAW_CSV, K_STATS_CSV
    )
    print(f"Optimal K found: {best_k}")

    # --- Phase 2: Optimal Batch Size (b) ---
    print(f"\n[2/2] Search for Optimal Batch Size (b) with Best K={best_k}...")
    best_b, b_stats = b_search(
        rdd_sample, best_k, B_RANGE, EPOCHS, NUM_ITERATIONS, B_RAW_CSV, B_STATS_CSV
    )
    print(f"Optimal Batch Size (b) found: {best_b}")

    # Save Parameters
    params_df = pd.DataFrame([{'best_k': best_k, 'best_b': best_b}])
    params_df.to_csv(PARAMS_CSV, index=False)
    
    duration = time.time() - start
    # Metadata
    with open(PARAMS_CSV.replace('.csv', '_metadata.json'), 'w') as f:
        json.dump({
            "execution_info": {
                "script": "initialization.py",
                "duration": f'{duration} (s)',
            },
            "optimal_parameters_found": {"best_k": int(best_k), "best_b": int(best_b)},
        }, f, indent=4)
    
    print(f"\nInitialization complete!")
    spark.stop()