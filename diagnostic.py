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
import pandas as pd
from pyspark.sql import SparkSession
from functions import cluster_diagnostic


# Directory setup
os.makedirs("data", exist_ok=True)

#  Path pointers
EVALUATION_PATH = "data/evaluation_dataset"
CENTROIDS_PATH = "runs/stats_thin_benchmark.csv"

if __name__ == "__main__":
    
    # Verify existence of required upstream files
    if not os.path.exists(EVALUATION_PATH): 
        raise FileNotFoundError(f"Missing {EVALUATION_PATH}. Please run generate_data.py first.")
    if not os.path.exists(CENTROIDS_PATH):
        raise FileNotFoundError(f"Missing {CENTROIDS_PATH}. Please run benchmark.py first.")
        
    # Load best centroids
    stats_df = pd.read_csv(CENTROIDS_PATH)

    centroids_json_string = stats_df['champion_centroids'].iloc[0]

    champion_centroids = np.array(json.loads(centroids_json_string), dtype=np.float32)

    # Environment check for the SparkSession
    worker_ips_str = os.getenv("WORKER_IPS", "")
    
    print("\n" + "="*40)
    if worker_ips_str:

        print("[INFO] YOU'RE ON A CLUSTER.")
        print("       Creating SParkSession with run_cluster.sh specs...")
        
        # Specs inside the running script
        spark = SparkSession.builder \
            .appName(f"Diagnostic") \
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
            .appName(f"Diagnostic") \
            .master("local[*]") \
            .config("spark.driver.memory", "512mb") \
            .getOrCreate()
            
    spark.sparkContext.setLogLevel("ERROR")
    print("="*40 + "\n")

    cluster_diagnostic(spark, EVALUATION_PATH, champion_centroids)
