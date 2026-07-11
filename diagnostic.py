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

#Global Variables
RUN_IDENTIFIER = "minibatch_plot"
NUM_PARTITIONS = 16

# Directory setup
os.makedirs("runs", exist_ok=True)

#  Path pointers
EVALUATION_PATH = "data/evaluation_dataset"
CENTROIDS_PATH = "runs/stats_strong_16c.csv"
FINAL_RAW_CSV = f"runs/{RUN_IDENTIFIER}.csv"

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

    #Creates spark session with run_script.sh specs
    spark = SparkSession.builder \
            .appName(f"Diagnostic_{RUN_IDENTIFIER}") \
            .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print(f"Loading RCV1 Parquet Evaluation Dataset...")

     # Read dataset
    df = spark.read.parquet(EVALUATION_PATH).repartition(NUM_PARTITIONS)
    # Added .count() to show exact dataset size, trigger the repartition
    total_docs = df.count()

    rdd_data = df.select("features", "true_labels").rdd

    print(f"\n=========================================")
    print(f"Dataset Size: {total_docs} documents")
    print(f"Partitions  : {NUM_PARTITIONS}")
    print(f"=========================================\n")


    cluster_diagnostic(rdd_data, champion_centroids, FINAL_RAW_CSV)

    spark.stop()
