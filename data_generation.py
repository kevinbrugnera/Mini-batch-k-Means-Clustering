import os
import sys
import subprocess
import getpass
import numpy as np 
from sklearn.utils import shuffle 

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
from sklearn.datasets import fetch_rcv1
from sklearn.decomposition import TruncatedSVD

# ==========================================
# GENERATE .parquet DATASET AND .parquet EVALUATION FILE
# ==========================================

def generate_data():
    print("Fetching RCV1 dataset...")
    rcv1 = fetch_rcv1()
    
    print("Applying Truncated SVD (reduction to 100 latent dimensions)...")
    svd = TruncatedSVD(n_components=100, random_state=16)
    
    # Compress the sparse matrix into a dense NumPy array
    X_dense = svd.fit_transform(rcv1.data)
    
    
    print("Extracting MACRO true labels...")
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
        
    print("Shuffling data and MACRO labels...")
    # Synchronous shuffle: mixes features and labels while keeping them paired for Spark splits
    X_shuffled, Y_shuffled = shuffle(X_dense, macro_labels_list, random_state=32)
    
    # --- NEW CODE: Creating DataFrames ---
    print("Generating main DataFrame (full dataset)...")
    df_main = pd.DataFrame({
        'features': list(X_shuffled),
        'true_labels': list(Y_shuffled) 
    })
    
    print("Generating evaluation DataFrame (single-label only)...")
    # Filter rows where the length of the 'true_labels' list is exactly 1
    mask_single_label = df_main['true_labels'].apply(lambda x: len(x) == 1)
    df_eval = df_main[mask_single_label].copy()
    
    # Flatten the list in 'true_labels' to a simple string (['ECAT'] -> 'ECAT') 
    df_eval['true_labels'] = df_eval['true_labels'].apply(lambda x: x[0])

    # Prepare the destination directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Saving in parquet files
    parquet_dataset_path = os.path.join(data_dir, "rcv1_dataset.parquet")
    parquet_eval_path = os.path.join(data_dir, "evaluation.parquet")
    
    print(f"Saving full dataset to {parquet_dataset_path}...")
    df_main.to_parquet(parquet_dataset_path, engine="pyarrow", index=False)
    
    print(f"Saving evaluation dataset to {parquet_eval_path}...")
    df_eval.to_parquet(parquet_eval_path, engine="pyarrow", index=False)
    
    print("Data generation complete!")

# ==========================================
# DISTRIBUTE FILES ON WORKERS
# ==========================================

def distribute_generated_data(data_dir="data"):
    """
    Checks if the script is running on a cluster environment.
    If the WORKER_IPS environment variable is detected, it automatically 
    distributes the newly generated Parquet files to all worker nodes.
    """
    # Fetch the worker IPs from the environment variables
    worker_ips_str = os.getenv("WORKER_IPS", "")
    
    # If the variable is empty or missing, assume local execution
    if not worker_ips_str:
        print("\n" + "="*40)
        print("\n[INFO] YOU'RE IN A LOCAL ENVIROMENT -> No data distribution required :)\n")
        print("-" * 40)
        print("[TIP] Did you mean to run this on a cluster?")
        print("      To distribute data automatically, you must define the worker nodes.")
        print("      Stop this script, open your terminal, activate your conda environment")
        print("      and run the following command with your actual IPs before executing:")
        print('      export WORKER_IPS="worker_1_ip, worker_2_ip, ..., worker_n_ip".')
        print("="*40 + "\n")
        return

    # Parse the IP addresses, removing any accidental spaces
    workers = [ip.strip() for ip in worker_ips_str.split(",") if ip.strip()]
    for i, ip in enumerate(workers):
        print(f'Worker {i} IP: {ip}')
    
    
    # Dynamically get the current OS user
    current_user = getpass.getuser()
    
    # Get the absolute path of the data directory to ensure rsync works correctly from any location
    abs_data_dir = os.path.abspath(data_dir)
    
    print("\n" + "="*40)
    print(f"YOU'RE ON A CLUSTER WITH ({len(workers)} WORKERS)")
    print(f"Starting files distribution...")
    
    for ip in workers:
        print(f'-> Transferring data to {ip}...')
        try:
            # Sychronize the entire data/ folder content using rsync over SSH.
            subprocess.run(
                ["rsync", "-avz", f"{abs_data_dir}/", f"{current_user}@{ip}:{abs_data_dir}/"], 
                check=True, 
                stdout=subprocess.DEVNULL, # Hides the verbose rsync output
                stderr=subprocess.DEVNULL  # Hides standard system errors to avoid cluttering the terminal
            )
            print(f"Files successfully synchronized on {ip}")
            
        except subprocess.CalledProcessError:
            # Graceful degradation: if SSH or rsync fails, print a warning but do not crash the script
            print(f"Network error while sending data to {ip}.")
            print(f"(If you are using a shared storage volume, you can safely ignore this warning)")
            
    print("="*40 + "\n")


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    generate_data()
    
    # Distributes file right after their creation on Master Node
    distribute_generated_data(data_dir="data")