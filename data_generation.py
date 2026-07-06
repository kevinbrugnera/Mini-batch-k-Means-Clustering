import os
import gc
import sys
import subprocess
import getpass
import numpy as np 
from sklearn.utils import shuffle 
from dotenv import load_dotenv

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
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
            'features': list(X_normalized_batch),
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
# DISTRIBUTE FILES ON WORKERS
# ==========================================

def distribute_generated_data():
    """
    Checks if the script is running on a cluster environment.
    If the WORKER_IPS environment variable is detected, it automatically 
    distributes the newly generated Parquet files to all worker nodes.
    """
    # Fetch the worker IPs from the environment variables.
    # Uses ips.env file with IPs storend inside thnaks to load_dotenv()
    worker_ips_str = os.getenv("WORKER_IPS", "")
    
    # If the variable is empty or missing, assume local execution
    if not worker_ips_str:
        print("\n" + "="*40)
        print("\n[INFO] YOU'RE IN A LOCAL ENVIROMENT -> No data distribution required :)\n")
        print("-" * 40)
        print("[WARN] You are indeed on a cluster? Don't worry!")
        print("       You must call ./run_cluster.sh <name of .py scritp> from your terminal.")
        print("       If you have any doubts, follow the instructions at instructions.pdf")
        print("="*40 + "\n")
        return

    # Parse the IP addresses, removing any accidental spaces
    workers = [ip.strip() for ip in worker_ips_str.split(",") if ip.strip()]
    master_user = getpass.getuser()

    #Takes workers user from ips.env file
    worker_user = os.getenv("WORKER_USER", master_user)

    #Takes absolute path of current directory on the master's node
    abs_path = os.getcwd()
    
    # State worker user (always ubuntu on cloudveneto I think) and data path
    #remote_user = "ubuntu"
    #remote_project_path = "~/Project"

    
    print("\n" + "="*40)
    print(f"[INFO] YOU'RE ON A CLUSTER WITH ({len(workers)} WORKERS.")
    print(f"       Starting files distribution...")
    print("="*40 + "\n")
    
    for i, ip in enumerate(workers):
        print(f'-> Transferring data to worker {i+1}: {ip}...')

        #In order for the rest of the code to work the data has to be in home/ubuntu/Project/data/...
        # So we create the dedicated directory in every worker specified
        '''
        create_dir_command = [
            "ssh", 
            "-o", "StrictHostKeyChecking=no", 
            "-o", "BatchMode=yes",  
            "-q",
            f"{remote_user}@{ip}", "mkdir -p ~/Project"]
        subprocess.run(create_dir_command, check=True)
        '''

        #With thi command we copy the data directory inside the newly created Project one
        scp_command = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-q",
            "-r", "data/",
            f"{worker_user}@{ip}:{abs_path}"]

        try:
            # Execute and raise value if something is wrong
            subprocess.run(scp_command, check=True)
            print(f"Successful Data Transfer on {ip}!")
        except subprocess.CalledProcessError:
            print(f"Network error while sending data to {ip}.")

    print("\nData Distriubtion Complete! Ready to work.")

    print("="*40 + "\n")



# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    generate_data()
    
    # Distributes file right after their creation on Master Node
    distribute_generated_data()