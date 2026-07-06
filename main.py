import os
import sys
import time
from dotenv import load_dotenv

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Load .env file containing IP addresses
load_dotenv("ips.env")     

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from functions import mini_batch_run

# Global Variables
K = 4
NUM_ITER = 50                 # Iteration for statistics
EPOCHS = 20                 # Training steps
RUN_IDENTIFIER = "change for every run with specifics to facilitate understanding" 

# Directory setup
os.makedirs("data", exist_ok=True)

# Dynamic Output Paths
FINAL_RAW_CSV = f"data/raw_{RUN_IDENTIFIER}.csv"
FINAL_STATS_CSV = f"data/stats_{RUN_IDENTIFIER}.csv"

#Read initialization file with best parameters
PARAMS_CSV = "data/best_params.csv"

# Updated Path pointing to the dense parquet
DATASET_PATH = "~/data/rcv1_dataset"
EVALUATION_PATH = "~/data/evaluation_dataset"

if __name__ == "__main__":
    
    # Verify existence of required upstream files
    if not os.path.exists(PARAMS_CSV):
        raise FileNotFoundError(f"Missing {PARAMS_CSV}. Please run initialization.py first.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Please run generate_data.py first.")
    if not os.path.exists(EVALUATION_PATH): 
        raise FileNotFoundError(f"Missing {EVALUATION_PATH}. Please run generate_data.py first.")
        
    # Load optimal parameters found during the initialization step
    params_df = pd.read_csv(PARAMS_CSV)
    #best_k = int(params_df['best_k'].iloc[0])
    batch_size = int(params_df['best_b'].iloc[0])

    # Environment check for the SparkSession
    worker_ips_str = os.getenv("WORKER_IPS", "")
    
    print("\n" + "="*40)
    if worker_ips_str:

        print("[INFO] YOU'RE ON A CLUSTER.")
        print("       Creating SParkSession with run_cluster.sh specs...")
        
        # Specs inside the running script
        spark = SparkSession.builder \
            .appName(f"MiniBatch_{RUN_IDENTIFIER}") \
            .getOrCreate()
    else:

        print("[INFO] YOU'RE IN A LOCAL ENVIROMENT.")
        print("       Using configurationin specified in <main.py> at line 69-73.")
        print("       Modify it accordingly to your needs.")
        print("-" * 40)
        print("[WARN] You are indeed on a cluster? Don't worry!")
        print("       You must call ./run_cluster.sh <name of .py scritp> from your terminal.")
        print("       If you have any doubts, follow the instructions at instructions.pdf")
        
        # Local configuration: set number of workers (local[*]) and memory desired
        spark = SparkSession.builder \
            .appName(f"MiniBatch_{RUN_IDENTIFIER}") \
            .master("local[*]") \
            .config("spark.driver.memory", "512mb") \
            .getOrCreate()
            
    spark.sparkContext.setLogLevel("ERROR")
    print("="*40 + "\n")
    
    print(f"Loading full RCV1 Parquet Dataset...")
    
    # Read dataset
    df = spark.read.parquet(DATASET_PATH)
    
    # Map rows into float32 arrays
    rdd_data = df.rdd.map(lambda row: np.array(row.features, dtype=np.float32))
    
    # Added .count() to show exact dataset size
    dataset_size = rdd_data.count()
    
    print(f"\n=========================================")
    print(f"Dataset Size: {dataset_size} documents")
    print(f"Iterations  : {NUM_ITER}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch Size  : {batch_size}")
    print(f"=========================================\n")
    
    # Execute the core final benchmark
    stats_df = mini_batch_run(
        rdd_data=rdd_data,
        best_k= K,
        best_b=batch_size,
        epochs=EPOCHS,                       
        num_iter=NUM_ITER,
        raw_csv=FINAL_RAW_CSV,
        stats_csv=FINAL_STATS_CSV
    )
    
    print(f"\n=========================================")
    print(f"Mini Batch Complete!")
    print(f"=========================================")

    spark.stop()