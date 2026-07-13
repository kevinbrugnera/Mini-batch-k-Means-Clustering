import os
import time
from dotenv import load_dotenv

# Load .env file containing IP addresses
load_dotenv("ips.env")

import json
import numpy as np
import urllib3
from pyspark.sql import SparkSession
from functions import classic_kmeans_run

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Global Variables
K = 4
NUM_ITER = 50                 # Iteration for statistics
EPOCHS = 20                   # Training steps
RUN_IDENTIFIER = "k_means_16c"
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
DATASET_PREFIX = "rcv1_dataset/"

if __name__ == "__main__":
    
    print("Initializing SparkSession with S3 configurations...")
    
    # Creates spark session with S3 connector configurations
    spark = SparkSession.builder \
        .master("spark://master:7077") \
        .appName(f"Classic_Kmeans_{RUN_IDENTIFIER}") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false") \
        .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider') \
        .config('spark.hadoop.fs.s3a.access.key', s3_creds['access_key']) \
        .config('spark.hadoop.fs.s3a.secret.key', s3_creds['secret_key']) \
        .config('spark.hadoop.fs.s3a.endpoint', s3_creds['endpoint']) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.metadatastore.impl", "org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("com.amazonaws.sdk.disableCertChecking", "true") \
        .getOrCreate()
            
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"Loading RCV1 Parquet Dataset from Cloud Storage...")
    
    # Costruzione dell'URI S3
    s3_dataset_uri = f"s3a://{s3_creds['bucket']}/{DATASET_PREFIX}"
    
    # Read dataset
    df = spark.read.parquet(s3_dataset_uri).repartition(NUM_PARTITIONS)
    # Added .count() to show exact dataset size
    total_docs = df.count()
    rdd_data = df.rdd.map(lambda row: np.array(row.features, dtype=np.float32))

    
    print(f"\n=========================================")
    print(f"Dataset Size: {total_docs} documents")
    print(f"Partitions  : {NUM_PARTITIONS}")
    print(f"Iterations  : {NUM_ITER}")
    print(f"Epochs      : {EPOCHS}")
    print(f"=========================================\n")
    
    # Execute the core final benchmark
    stats_df = classic_kmeans_run(
        rdd_data=rdd_data,
        K= K,
        epochs=EPOCHS,                       
        num_iter=NUM_ITER,
        raw_csv=FINAL_RAW_CSV,
        stats_csv=FINAL_STATS_CSV
    )
    
    print(f"\n=========================================")
    print(f"Classic K-Means Complete!")
    print(f"{total_docs} Documents Analyzed.")
    print(f"=========================================")

    spark.stop()