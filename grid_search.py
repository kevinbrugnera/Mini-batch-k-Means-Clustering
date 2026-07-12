import os
import time
from dotenv import load_dotenv

# Load .env file containing IP addresses
load_dotenv("ips.env")       

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pyspark.sql import SparkSession
from functions import b_search
import boto3
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global Variables
NUM_ITERATIONS = 50 # Iterations for statistics
K = 4
EPOCHS = 20  # Training step for single k-means run
B_RANGE = [500, 1000, 2000, 4000, 8000, 16000] # b values to test
SAMPLE_SIZE = 50000 # Sample of RCV1 dataset to analyze

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
B_RAW_CSV = "runs/b_search.csv"
B_STATS_CSV = "runs/stats_b_search.csv"

start = time.time()
if __name__ == "__main__":
    
    spark = SparkSession.builder \
        .master("spark://master:7077") \
        .appName("KMeans_GridSearch") \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.4.1,org.apache.hadoop:hadoop-common:3.4.1') \
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
    
    print(f"Loading the filename list from CloudStorage...")
    
    s3_client = boto3.client("s3",
        endpoint_url=s3_creds['endpoint'],
        aws_access_key_id=s3_creds['access_key'],
        aws_secret_access_key=s3_creds['secret_key'],
        verify=False)
    
    # Retrieving filenames with s3

    file_list = s3_client.list_objects_v2(Bucket=bucket_name, 
                                          Prefix="rcv1_dataset/")
    filenames = [item['Key'] for item in file_list.get('Contents', []) if item['Key'].endswith('.parquet')]

    # Split them into train/test
    np.random.seed(123)
    np.random.shuffle(filenames)
    split_idx = int(len(filenames) * 0.8)
    train_files = filenames[:split_idx]
    test_files = filenames[split_idx:]

    print(f"Driver is distributing files to Workers' executors...")

    num_partitions = 16 # max number of executors we can allocate
    rdd_train_files = spark.sparkContext.parallelize(train_files, numSlices=num_partitions)
    rdd_train_files.persist()
    
    print("Initial centroids extraction...")
    init_df = spark.read.parquet(f"s3a://{s3_creds['bucket']}/{train_files[0]}")
    initial_rows = init_df.take(K)
    initial_centers = np.array([row.features for row in initial_rows], dtype=np.float32)

    print("Building Test RDD...")
    test_paths = [f"s3a://{s3_creds['bucket']}/{f}" for f in test_files]
    df_test = spark.read.parquet(*test_paths)
    rdd_test = df_test.rdd.map(lambda row: np.array(row.features, dtype=np.float32)).persist()

    print(f"\n Search for Optimal Batch Size with K={K}...")
    best_b, b_stats = b_search(rdd_train_files, rdd_test, initial_centers, 
                               K, B_RANGE, EPOCHS, NUM_ITERATIONS, B_RAW_CSV, 
                               B_STATS_CSV, s3_creds
    )
    
    print(f"Optimal Batch Size found: {best_b}")
    print(f"\nInitialization complete!")
    spark.stop()