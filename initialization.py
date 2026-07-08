import os
import sys
import time
from dotenv import load_dotenv

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Load .env file containing IP addresses
load_dotenv("ips.env")       

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pyspark.sql import SparkSession
from functions import b_search

# Global Variables
NUM_ITERATIONS = 50                # Iterations for statistics
K = 4
EPOCHS = 20                         # Training step for single k-means run
#K_RANGE = [2,4,6,8,10]       # K values to test
B_RANGE = [500, 1000, 2000, 4000, 8000, 16000]                 # b values to test
SAMPLE_SIZE = 50000                  # Sample of RCV1 dataset to analyze

# Directory setup
os.makedirs("data", exist_ok=True)

# Output Paths
#K_RAW_CSV = "data/k_search_raw.csv"
#K_STATS_CSV = "data/k_search_stats.csv"
B_RAW_CSV = "data/b_search_raw_try.csv"
B_STATS_CSV = "data/b_search_stats_try.csv"
#PARAMS_CSV = "data/best_b.csv"

# Updated Path pointing to the dense parquet
PARQUET_PATH = "data/rcv1_dataset"

start = time.time()
if __name__ == "__main__":
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing {PARQUET_PATH}. Please run generate_data.py first.")

    # Environment check for the SparkSession
    worker_ips_str = os.getenv("WORKER_IPS", "")
    
    print("\n" + "="*40)
    if worker_ips_str:

        print("[INFO] YOU'RE ON A CLUSTER.")
        print("       Creating SparkSession with run_cluster.sh specs...")
        
        
        # Specs inside the running script
        spark = SparkSession.builder \
            .appName("Grid-Search") \
            .getOrCreate()
    else:

        print("[INFO] YOU'RE IN A LOCAL ENVIROMENT.")
        print("       Using configurationin specified in <initialization.py> at line 69-73.")
        print("       Modify it accordingly to your needs.")
        print("-" * 40)
        print("[WARN] You are indeed on a cluster? Don't worry!")
        print("       You must call ./run_cluster.sh <name of .py scritp> from your terminal.")
        print("       If you have any doubts, follow the instructions at instructions.pdf")
        
        # Local configuration: set number of workers (local[*]) and memory desired
        spark = SparkSession.builder \
            .appName("Grid-Search") \
            .master("local[*]") \
            .config("spark.driver.memory", "512mb") \
            .getOrCreate()
            
    spark.sparkContext.setLogLevel("ERROR")
    print("="*40 + "\n")
    
    print(f"Loading a sample of {SAMPLE_SIZE} documents from .parquet Dataset...")
    
    # Read the parquet directly
    df_full = spark.read.parquet(PARQUET_PATH)

    #We will read only approximately SAMPLE_SIZE documents
    # not exactly that number since we are sampling with sample function
    total_docs = df_full.count()
    fraction = min(1.0, SAMPLE_SIZE / total_docs)   
    df_sample = df_full.sample(False, fraction, seed=28)
    
    # Map each row to a dense numpy array of float32
    rdd_sample = df_sample.rdd.map(lambda row: np.array(row.features, dtype=np.float32))

    '''
    # Optimal K
    print("\n[1/2] Search for Optimal K...")
    best_k, k_stats = k_search(
        rdd_sample, K_RANGE, EPOCHS, NUM_ITERATIONS, K_RAW_CSV, K_STATS_CSV
    )
    print(f"Optimal K found: {best_k}")
    '''

    #  Optimal Batch Size b
    print(f"\nSearch for Optimal Batch Size with K={K}...")
    best_b, b_stats = b_search(                                 #add best_b if using silhouette
        rdd_sample, K, B_RANGE, EPOCHS, NUM_ITERATIONS, B_RAW_CSV, B_STATS_CSV
    )
    print(f"Optimal Batch Size found: {best_b}")

    
    '''
    # Save best b in dedicated file (little overkill but helpful)
    params_df = pd.DataFrame([{ 'best_b': best_b}])
    params_df.to_csv(PARAMS_CSV, index=False)
    '''

    duration = time.time() - start

    '''Useless
    # Metadata
    with open(PARAMS_CSV.replace('.csv', '_metadata.json'), 'w') as f:
        json.dump({
            "execution_info": {
                "script": "initialization.py",
                "duration": f'{duration} (s)',
            },
            "optimal_parameters_found": {"best_b": int(best_b)},
        }, f, indent=4)
    '''

    print(f"\nInitialization complete!")
    spark.stop()