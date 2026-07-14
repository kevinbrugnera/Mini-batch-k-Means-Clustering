import os
import sys
from dotenv import load_dotenv

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Load .env file containing IP addresses
load_dotenv("ips.env")       

import numpy as np
from pyspark.sql import SparkSession
from functions import b_search

# Global Variables
NUM_ITERATIONS = 50                                 # Iterations for statistics
K = 4                                               # K number of clusters
EPOCHS = 20                                         # Training step for single k-means run
B_RANGE = [500, 1000, 2000, 4000, 8000, 16000]      # Range of b values to test
SAMPLE_SIZE = 50000                                 # Sample of RCV1 dataset to analyze
NUM_PARTITIONS = 16                                 # Number of partitions for the RDD

# Directory setup
os.makedirs("runs", exist_ok=True)

# Output Paths
B_RAW_CSV = "runs/b_search.csv"
B_STATS_CSV = "runs/stats_b_search.csv"

# Path pointing to the parquet dataset
PARQUET_PATH = "data/rcv1_dataset"

if __name__ == "__main__":

    #Check existence of required files
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing {PARQUET_PATH}. Please run generate_data.py first.")

    #Creates spark session with run_script.sh specs
    spark = SparkSession.builder \
            .appName("Grid-Search") \
            .config("spark.api.mode", " classic") \
            .getOrCreate()
            
    # Logs display only errors
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading a sample of {SAMPLE_SIZE} documents from .parquet Dataset...")
    
    # Read the parquet dataset using the specified number of partitions
    df_full = spark.read.parquet(PARQUET_PATH).repartition(NUM_PARTITIONS)

    # Added .count() to show exact dataset size, trigger the repartition
    total_docs = df_full.count()

    # Evaluate fraction corresponding to SAMPLE_SIZE
    fraction = min(1.0, SAMPLE_SIZE / total_docs)   
    df_sample = df_full.sample(False, fraction, seed=28)
    
    # Map rows into float32 arrays. We use float32 to save some memory, no real need for double precision
    rdd_sample = df_sample.rdd.map(lambda row: np.array(row.features, dtype=np.float32))

    #Optimal Batch Size b
    print(f"\nSearch for Optimal Batch Size with K={K}...")
    best_b, b_stats = b_search(                                 #add best_b if using silhouette
        rdd_sample, K, B_RANGE, EPOCHS, NUM_ITERATIONS, B_RAW_CSV, B_STATS_CSV
    )

    print(f"Optimal Batch Size found: {best_b}")
    print(f"\nInitialization complete!")
    spark.stop()
