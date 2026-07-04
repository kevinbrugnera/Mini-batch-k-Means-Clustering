import os
import sys

# Force spark to use python version used in the environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
from sklearn.datasets import fetch_rcv1
from sklearn.decomposition import TruncatedSVD


def generate_data():
    print("Fetching RCV1 dataset...")
    rcv1 = fetch_rcv1()
    
    print("Applying Truncated SVD (reduction to 100 latent dimensions)...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    
    # Compress the sparse matrix into a dense NumPy array
    X_dense = svd.fit_transform(rcv1.data)
    
    print("Generating DataFrame...")
    # Convert the dense matrix into a list of arrays for the DataFrame
    df = pd.DataFrame({'features': list(X_dense)})
    
    # Prepare the destination directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, "rcv1_dense.parquet")
    
    print(f"Saving binary data to Parquet format at {parquet_path}...")
    # Save using pyarrow to natively preserve the NumPy array structure
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    
    print("Data generation complete!")

if __name__ == "__main__":
    generate_data()