import os
import gc
import sys
import numpy as np
from dotenv import load_dotenv
import shutil
# S3 objectstore library
import boto3
import urllib3

# Avoid warnings from Cloud Storage connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv("ips.env")       # Load .env file containing IP addresses

import pandas as pd
from sklearn.datasets import fetch_rcv1
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ==========================================
# GENERATE .parquet DATASET AND .parquet EVALUATION FILE
# ==========================================

def generate_data():
    print("Processing RCV1 dataset...")
    rcv1 = fetch_rcv1()

    # Extract true macrolabels
    target_names = rcv1.target_names
    macro_labels = ['CCAT', 'ECAT', 'GCAT', 'MCAT']

    # Saves the index identifying the macro labels (if 'CCAT' is at index 10 is stored as 10: 'CCAT')
    macro_indices = {np.where(target_names == cat)[0][0]: cat for cat in macro_labels}
    targets_csr = rcv1.target
    macro_labels_list = []

    # Iterate over all documents to extract only the macro labels
    for i in range(targets_csr.shape[0]):
        # Finds indices of labels assigned to every single file
        row_indices = targets_csr.indices[targets_csr.indptr[i]:targets_csr.indptr[i+1]]
        # Checks if these labels are in the dictionary. If so, it keeps them
        macros_for_doc = [macro_indices[idx] for idx in row_indices if idx in macro_indices]
        macro_labels_list.append(macros_for_doc)

    print("Truncated SVD Transformation...")
    # Use a decent sample of documents extarcted randomly, to avoid memory usage overload
    # 50000 to 100000 documents should be enough to estimate good fitting parameters
    np.random.seed(42)
    sample_idx = np.random.choice(rcv1.data.shape[0], size=50000, replace=False)
    sorted_idx = np.sort(sample_idx)
    X_sample = rcv1.data[sorted_idx]

    # Reduce to 100 latent dimensions
    svd = TruncatedSVD(n_components=100, random_state=16)
    svd.fit(X_sample)

    # RAM cleaning
    del sample_idx, sorted_idx, X_sample
    gc.collect()

    print("Dataset Chunking...")

    data_dir = "data/rcv1_dataset"
    eval_dir = "data/evaluation_dataset"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    n_docs = rcv1.data.shape[0]
    batch_size = 10000

    for i, start_idx in enumerate(range(0, n_docs, batch_size)):
        end_idx = min(start_idx + batch_size, n_docs)
        print(f"  -> Processing and saving chunk {start_idx} to {end_idx} / {n_docs}")

        # Transofmration and normalization to avoid wrong clustering
        X_batch = rcv1.data[start_idx:end_idx]
        X_dense_batch = svd.transform(X_batch)
        X_normalized_batch = normalize(X_dense_batch, norm='l2', axis=1)
        Y_batch = macro_labels_list[start_idx:end_idx]

        # Temporary DataFrame to store data
        df_chunk = pd.DataFrame({
            'features': [row.tolist() for row in X_normalized_batch],
            'true_labels': Y_batch
        })

        #  Saves data chunk in dedicated diretcory
        chunk_filename = f"data_{i:02d}.parquet"
        df_chunk.to_parquet(os.path.join(data_dir, chunk_filename), engine="pyarrow", index=False)

        # 4Evaluation data saving
        eval_filename = f"eval_{i:02d}.parquet"
        # Takes only single label documents
        mask_single = df_chunk['true_labels'].apply(lambda x: len(x) == 1)
        df_eval_chunk = df_chunk[mask_single].copy()

        if not df_eval_chunk.empty:
            df_eval_chunk['true_labels'] = df_eval_chunk['true_labels'].apply(lambda x: x[0])
            df_eval_chunk.to_parquet(os.path.join(eval_dir, eval_filename), engine="pyarrow", index=False)

        # Cleaning RAM
        del X_batch, X_dense_batch, Y_batch, df_chunk, df_eval_chunk
        gc.collect()

    print("\nData generation complete! Data saved in Parquet partitions.")

# ==========================================
# UPLOAD FILES ON CLOUDVENETO 
# ==========================================

def upload_data_S3():
    """
    Checks if the script is running on a cluster environment.
    If the WORKER_IPS environment variable is detected, it automatically
    uploads data in CloudVeneto memory.
    """
    # Fetch the worker IPs from the environment variables.
    # Uses ips.env file with IPs storend inside thnaks to load_dotenv()
    worker_ips_str = os.getenv("WORKER_IPS", "")

    # If the variable is empty or missing, assume local execution
    if not worker_ips_str:
        print("\n" + "="*40)
        print("\n[INFO] YOU'RE IN A LOCAL ENVIROMENT -> No data distribution required :)\n")
        print("-" * 40)
        print("[WARN] Are you actually on a cluster? Don't worry!")
        print("       You forgot to change WORKER_IPS settings in ips.env file.")
        print("       If you have any doubts, follow the instructions at instructions.md")
        print("="*40 + "\n")
        return

    # Parse the IP addresses, removing any accidental spaces
    workers = [ip.strip() for ip in worker_ips_str.split(",") if ip.strip()]

    print("\n" + "=" * 40)
    print(f"[INFO] CLUSTER ENVIRONMENT DETECTED ({len(workers)} workers).")
    print("\n[INFO] Using the CloudVeneto object store.")
    print("Uploading the generated data to the corresponding container...")

    s3_client = boto3.client("s3",
                             endpoint_url=os.getenv("S3_ENDPOINT"),
                             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                             verify=False)
    
    bucket_name = os.getenv("S3_BUCKET")
    local_folder = "data"

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_path, local_folder)

            try:
                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded: {s3_key}")
            except Exception as e:
                print(f"[ERROR] Failed to upload {s3_key}: {e}")

    print("[INFO] Upload completed.")
    shutil.rmtree(local_folder) # remove data folder from master node
    print("=" * 40 + "\n")

# ==========================================
# EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    generate_data()
    upload_data_S3()