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
os.makedirs("runs", exist_ok=True)

# Output Paths
B_RAW_CSV = "runs/b_search.csv"
B_STATS_CSV = "runs/stats_b_search.csv"

# Updated Path pointing to the dense parquet
PARQUET_PATH = "data/rcv1_dataset"

start = time.time()
if __name__ == "__main__":
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing {PARQUET_PATH}. Please run generate_data.py first.")

    #Creates spark session with run_script.sh specs
    spark = SparkSession.builder \
            .appName("Grid-Search") \
            .config("spark.api.mode", " classic") \
            .getOrCreate()
            
    spark.sparkContext.setLogLevel("ERROR")
    
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

    #  Optimal Batch Size b
    print(f"\nSearch for Optimal Batch Size with K={K}...")
    best_b, b_stats = b_search(                                 #add best_b if using silhouette
        rdd_sample, K, B_RANGE, EPOCHS, NUM_ITERATIONS, B_RAW_CSV, B_STATS_CSV
    )
    print(f"Optimal Batch Size found: {best_b}")
    print(f"\nInitialization complete!")
    spark.stop()
