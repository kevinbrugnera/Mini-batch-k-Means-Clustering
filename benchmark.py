import os
import time
from dotenv import load_dotenv

# Load .env file containing IP addresses
load_dotenv("ips.env")

import json
import numpy as np
import boto3
import urllib3
from pyspark.sql import SparkSession
from functions import mini_batch_run

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Global Variables
K = 4
NUM_ITER = 50                 # Iterations for statistics
EPOCHS = 20                   # Training steps
SAMPLE_SIZE = 0               # Documents to analyze. Set to zero if you want to analyze full dataset
RUN_IDENTIFIER = "strong_12c"
NUM_PARTITIONS = 48

# S3 Credential Variables
bucket_name = os.getenv("S3_BUCKET")

s3_creds = {'endpoint': os.getenv("S3_ENDPOINT"),
            'access_key': os.getenv("AWS_ACCESS_KEY_ID"),
            'secret_key': os.getenv("AWS_SECRET_ACCESS_KEY"),
            'bucket': bucket_name
}

# Directory setup
os.makedirs("runs", exist_ok=True)

# Output Paths
FINAL_RAW_CSV = f"runs/{RUN_IDENTIFIER}.csv"
FINAL_STATS_CSV = f"runs/stats_{RUN_IDENTIFIER}.csv"
PARAMS_JSON = "runs/b_search_metadata.json"
DATASET_PREFIX = "rcv1_dataset/"

if __name__ == "__main__":
    
    # Verify existence of required parameter file
    if not os.path.exists(PARAMS_JSON):
        raise FileNotFoundError(f"Missing {PARAMS_JSON}. Please run grid_search.py first.")
    
    # Load optimal parameters found during the initialization step
    with open(PARAMS_JSON, "r") as f:
        init_metadata = json.load(f)
    
    # Extract value
    batch_size = int(init_metadata["metrics_summary"]["optimal_b_found"])
    
    print("Initializing SparkSession with S3 configurations...")
    
    # Creates spark session with S3 connector configurations
    spark = SparkSession.builder \
        .master("spark://master:7077") \
        .appName("KMeans_GridSearch") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")\
        .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider')\
        .config('spark.hadoop.fs.s3a.access.key', s3_creds['access_key']) \
        .config('spark.hadoop.fs.s3a.secret.key', s3_creds['secret_key']) \
        .config('spark.hadoop.fs.s3a.endpoint', s3_creds['endpoint']) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.metadatastore.impl", "org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled","false") \
        .config("com.amazonaws.sdk.disableCertChecking","true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading RCV1 Parquet Dataset from Cloud Storage...")
    
    # Costruzione dell'URI S3 per l'intero dataset
    s3_dataset_uri = f"s3a://{s3_creds['bucket']}/{DATASET_PREFIX}"
    
    # Read dataset fully distributed via Spark S3A
    df = spark.read.parquet(s3_dataset_uri).repartition(NUM_PARTITIONS)
    
    # Added .count() to show exact dataset size and trigger the repartition
    total_docs = df.count()

    if SAMPLE_SIZE == 0:
        # Analyze whole dataset
        # Map rows into float32 arrays
        rdd_data = df.rdd.map(lambda row: np.array(row.features, dtype=np.float32))
        dataset_size = total_docs
    else:
        fraction = min(1.0, float(SAMPLE_SIZE) / total_docs)
        df_sample = df.sample(False, fraction, seed=58)
        rdd_data = df_sample.rdd.map(lambda row: np.array(row.features, dtype=np.float32))
        dataset_size = SAMPLE_SIZE
    
    print(f"\n=========================================")
    print(f"Dataset Size: {dataset_size} documents")
    print(f"Partitions  : {NUM_PARTITIONS}")
    print(f"Iterations  : {NUM_ITER}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch Size  : {batch_size}")
    print(f"=========================================\n")
    
    # Execute the core final benchmark
    stats_df = mini_batch_run(
        rdd_data=rdd_data,
        K=K,
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