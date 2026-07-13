import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark import StorageLevel
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import ClusteringEvaluator
import pyarrow.parquet as pq
import pyarrow.fs as fs

# ==========================================
# METADATA GENERATOR
# ==========================================

def save_metadata(func_name, duration, params, metrics, base_filepath):
    """Generates a JSON file containing execution metadata."""
    metadata = {
        "execution_info": {
            "function": func_name,
            "duration": duration,
        },
        "parameters": params,
        "metrics_summary": metrics
    }
    
    meta_path = base_filepath.replace('.csv', '_metadata.json')
         
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# ==========================================
# MATH FUNCTIONS
# ==========================================

def closest_idx(point: np.ndarray, centers: np.ndarray) -> int:
    """Finds the index of the closest center."""
    distances = np.sum((centers - point) ** 2, axis=1)
    return int(np.argmin(distances))

def WCSS(rdd_test, centers: list) -> float:
    """Calculates the Within-Cluster Sum of Squares (WCSS) for cost evaluation."""
    centers_np = np.array(centers)
    bc_centers = rdd_test.context.broadcast(centers_np)
    
    wcss = rdd_test.map(
        lambda x: float(np.sum((x - bc_centers.value[closest_idx(x, bc_centers.value)]) ** 2))
    ).sum()
    
    bc_centers.destroy()

    return wcss

def calinski_harabasz(rdd_test, centers: list, k: int) -> float:
    """Calculates the Calinski-Harabasz Index for best_b identification."""
    N = rdd_test.count()
    
    global_sum = rdd_test.reduce(lambda a, b: a + b)
    global_mean = global_sum / float(N)
    
    bc_global_mean = rdd_test.context.broadcast(global_mean)

    tss = rdd_test.map(
        lambda x: float(np.sum((x - bc_global_mean.value) ** 2))
    ).sum()

    bc_global_mean.destroy()
    
    wcss = WCSS(rdd_test, centers)
    
    bcss = tss - wcss
    
    if wcss == 0:
        return float('inf')
        
    ch_score = (bcss / (k - 1)) / (wcss / (N - k))
    
    return ch_score

# ==========================================
# CLUSTERING ALGORITHMS
# ==========================================

def classic_kmeans(rdd_train, k: int, epochs: int, seed: int):
    """Executes the standard K-Means algorithm using Spark."""
    centers = rdd_train.takeSample(False, k, seed)
    
    for _ in range(epochs):
        centers_np = np.array(centers)
        bc_centers = rdd_train.context.broadcast(centers_np)
        
        mapped_points = rdd_train.map(lambda x: (closest_idx(x, bc_centers.value), (x, 1)))
        
        reduced_points = mapped_points.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        
        new_centers_rdd = reduced_points.map(lambda x: (x[0], x[1][0] / x[1][1]))
        new_centers_dict = dict(new_centers_rdd.collect())
        
        centers = [new_centers_dict.get(i, centers[i]) for i in range(k)]
        bc_centers.destroy()
        
    return centers

def process_mini_batch(file_iterator, bc_centers, b_per_worker, s3_creds):
    """Worker mapPartitions logic: fetch from S3, sample locally, map to centroids."""

    # Set environment variables so PyArrow picks them up automatically
    os.environ['AWS_ACCESS_KEY_ID'] = s3_creds['access_key']
    os.environ['AWS_SECRET_ACCESS_KEY'] = s3_creds['secret_key']
    os.environ['AWS_ENDPOINT_URL'] = s3_creds['endpoint']
    # If the endpoint is HTTPS but has no valid cert, PyArrow defaults to insecure if needed
    
    # Initialize using environment variables; no risk of 'unexpected argument'
    s3 = fs.S3FileSystem(
        scheme='https',
        endpoint_override=s3_creds['endpoint']
    )
    
    centers = bc_centers.value
    k = len(centers)
    
    sums = np.zeros_like(centers)
    counts = np.zeros(k)
    
    for file_key in file_iterator:
        full_path = f"{s3_creds['bucket']}/{file_key}"
        try:
            # The filesystem object is correctly passed to pq.ParquetDataset
            dataset = pq.ParquetDataset(full_path, filesystem=s3)
            df = dataset.read(columns=['features']).to_pandas()
            features = np.vstack(df['features'].values).astype(np.float32)
            
            n_rows = features.shape[0]
            sample_size = min(b_per_worker, n_rows)
            
            if sample_size > 0:
                indices = np.random.choice(n_rows, sample_size, replace=False)
                sampled_batch = features[indices]
                
                for point in sampled_batch:
                    distances = np.sum((centers - point) ** 2, axis=1)
                    c_idx = int(np.argmin(distances))
                    sums[c_idx] += point
                    counts[c_idx] += 1
        except Exception:
            continue
            
    for c_idx in range(k):
        if counts[c_idx] > 0:
            yield (c_idx, (sums[c_idx], counts[c_idx]))

def minibatch_kmeans(rdd_train_files, initial_centers, k: int, b: int, epochs: int, seed: int, s3_creds: dict):
    """Executes the Mini-Batch K-Means algorithm distributing the load on Spark workers."""
    centers = np.copy(initial_centers) 
    v = np.zeros(k)
    
    num_partitions = rdd_train_files.getNumPartitions()
    b_per_worker = max(1, b // num_partitions)
    
    np.random.seed(seed)
    
    for epoch in range(epochs):
        bc_centers = rdd_train_files.context.broadcast(centers)
        
        partial_aggregates = rdd_train_files.mapPartitions(
            lambda iterator: process_mini_batch(iterator, bc_centers, b_per_worker, s3_creds)
        )
        
        reduced_batch = partial_aggregates.reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        ).collect()
        
        for c_idx, (sum_x, count) in reduced_batch:
            v[c_idx] += count               
            eta = count / v[c_idx]          
            batch_mean = sum_x / count      
            centers[c_idx] = (centers[c_idx] * (1.0 - eta)) + (batch_mean * eta)
            
        bc_centers.destroy()
        
    return centers.tolist()

# ==========================================
# GET BEST INITIALIZATION PARAMETERS
# ==========================================

def b_search(rdd_train_files, rdd_test, initial_centers, K, b_list, epochs, num_iter, raw_csv, stats_csv, s3_creds):
    """Grid Search to find best b parameter among list of values."""
    start = time.time()
    results = []
    
    for run_id in range(num_iter):
        for b in b_list:
            print(f'Testing Mini-Batch Size={b}, iteration:{run_id}')
            start_time = time.time()
            
            centers = minibatch_kmeans(rdd_train_files, initial_centers, K, b, epochs, seed=run_id, s3_creds=s3_creds)
            
            exec_time = time.time() - start_time
            ch_score = calinski_harabasz(rdd_test, centers, K)
            
            results.append({
                'batch_size': b,
                'iteration_id': run_id,
                'ch_score': ch_score,
                'execution_time': exec_time
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    df_stats = df_results.groupby('batch_size').agg(
        mean_ch=('ch_score', 'mean'),
        std_ch=('ch_score', 'std'),
        mean_time=('execution_time', 'mean'),
        std_time=('execution_time', 'std')
    ).reset_index()
    df_stats.to_csv(stats_csv, index=False)
    
    best_ch_score = df_stats['mean_ch'].max()
    tolerance = 0.025
    accepted_ch = best_ch_score * (1.0 - tolerance)
    mask_ch = df_stats['mean_ch'] >= accepted_ch

    accepted_batch = df_stats[mask_ch]
    mask_b = accepted_batch['mean_time'].idxmin()
    best_b = accepted_batch.loc[mask_b, 'batch_size']
    
    duration = time.time() - start
    
    save_metadata(
        func_name="b_search",
        duration= f"{duration} (s)",
        params={"best_k_used": K, "b_list": b_list, "epochs": epochs, "iterations": num_iter},
        metrics={"optimal_b_found": int(best_b)},
        base_filepath=raw_csv
    )

    return int(best_b), df_stats

# ==========================================
# DIAGNOSTIC/ PERFORMANCE EVALUATION
# ==========================================

def cluster_diagnostic(rdd_data, champion_centers, plot_csv):
    """Evaluates the Champion Model against the ground truth labels."""
    start = time.time()
    centers_np = np.array(champion_centers)
    bc_centers = rdd_data.context.broadcast(centers_np)

    def predict_label(row):
        point = np.array(row['features'])
        pred = closest_idx(point, bc_centers.value)
        return (row['true_labels'], pred)

    labels_rdd = rdd_data.map(predict_label)
    labels_local = labels_rdd.collect()

    labels_true = [x[0] for x in labels_local]
    labels_pred = [x[1] for x in labels_local]
    
    nmi_score = NMI(labels_true, labels_pred)
    evaluated_documents = len(labels_true)
    
    fraction = min(1, 5000/evaluated_documents)
    sampled_rows = rdd_data.sample(False, fraction).collect()

    df_plot = pd.DataFrame([row.asDict() for row in sampled_rows])
    df_plot['predicted_labels'] = df_plot['features'].apply(lambda x: closest_idx(np.array(x), centers_np))
    df_plot['NMI_score'] = nmi_score

    df_plot.to_csv(plot_csv, index=False)
    save_metadata("cluster_diagnostic", f'{time.time() - start} (s)', 'Champion run', {"NMI": nmi_score}, plot_csv)
    bc_centers.destroy()

# ==========================================
# FINAL MINI BATCH
# ==========================================

def mini_batch_run(rdd_train_files, rdd_test, initial_centers, K, best_b, epochs, num_iter, raw_csv, stats_csv, s3_creds):
    """Runs the final comprehensive mini batch k-means."""
    start = time.time()
    results = []

    best_wcss = float('inf')
    best_centers = []
    best_run_id = -1
    
    for run_id in range(num_iter):
        start_time = time.time()
        centers = minibatch_kmeans(rdd_train_files, initial_centers, K, best_b, epochs, seed=run_id, s3_creds=s3_creds)
        exec_time = time.time() - start_time
        
        wcss = WCSS(rdd_test, centers)

        if wcss < best_wcss:
            best_wcss = wcss
            best_centers = centers
            best_run_id = run_id
        
        results.append({'iteration_id': run_id, 'k_value': K, 'batch_size': best_b, 'execution_time': exec_time, 'wcss': wcss})
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    best_centers_check = [c.tolist() if isinstance(c, np.ndarray) else c for c in best_centers]
    stats_dict = {'mean_wcss': df_results['wcss'].mean(), 'std_wcss': df_results['wcss'].std(), 'mean_time': df_results['execution_time'].mean(), 'std_time': df_results['execution_time'].std(), 'champion_run_id': best_run_id, 'champion_wcss_score': best_wcss, 'champion_centroids': json.dumps(best_centers_check)}
    
    df_stats = pd.DataFrame([stats_dict])
    df_stats.to_csv(stats_csv, index=False)
    save_metadata("mini_batch_run", f"{time.time() - start} (s)", {"K": K, "b": best_b, "epochs": epochs, "iterations": num_iter}, {"champion_run_id": best_run_id, "champion_wcss": best_wcss}, raw_csv)
    
    return df_stats