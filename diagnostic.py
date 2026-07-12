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
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global Variables
RUN_IDENTIFIER = "minibatch_plot"
NUM_PARTITIONS = 16

# S3 Credential Variables
bucket_name = os.getenv("S3_BUCKET")

s3_creds = {
    'endpoint': os.getenv("S3_ENDPOINT"),
    'access_key': os.getenv("AWS_ACCESS_KEY_ID"),
    'secret_key': os.getenv("AWS_SECRET_ACCESS_KEY"),
    'bucket': bucket_name
}

# Directory setup
os.makedirs("runs", exist_ok=True)

# Path pointers
EVALUATION_PREFIX = "evaluation_dataset/"
CENTROIDS_PATH = "runs/stats_strong_16c.csv"
FINAL_RAW_CSV = f"runs/{RUN_IDENTIFIER}.csv"

if __name__ == "__main__":

    # Verify existence of required upstream local files
    if not os.path.exists(CENTROIDS_PATH):
        raise FileNotFoundError(f"Missing {CENTROIDS_PATH}. Please run benchmark.py first.")

    # Load best centroids from local storage
    stats_df = pd.read_csv(CENTROIDS_PATH)
    centroids_json_string = stats_df['champion_centroids'].iloc[0]
    champion_centroids = np.array(json.loads(centroids_json_string), dtype=np.float32)

    print("Inizializzazione SparkSession con configurazioni S3...")

    # Creates spark session with S3 connector configurations
    spark = SparkSession.builder \
        .appName(f"Diagnostic_{RUN_IDENTIFIER}") \
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

    print("Loading RCV1 Parquet Evaluation Dataset from Cloud Storage...")

    # Costruzione dell'URI S3
    s3_eval_uri = f"s3a://{s3_creds['bucket']}/{EVALUATION_PREFIX}"

    # Read dataset fully distributed via Spark S3A
    df = spark.read.parquet(s3_eval_uri).repartition(NUM_PARTITIONS)
    
    # Added .count() to show exact dataset size and trigger the repartition
    total_docs = df.count()

    rdd_data = df.select("features", "true_labels").rdd

    print(f"\n=========================================")
    print(f"Dataset Size: {total_docs} documents")
    print(f"Partitions  : {NUM_PARTITIONS}")
    print(f"=========================================\n")

    # Esecuzione del calcolo diagnostico
    cluster_diagnostic(rdd_data, champion_centroids, FINAL_RAW_CSV)

    spark.stop()