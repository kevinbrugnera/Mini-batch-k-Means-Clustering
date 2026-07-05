import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from kneed import KneeLocator
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI

# ==========================================
# METADATA GENERATOR
# ==========================================

def save_metadata(func_name, duration, params, metrics, base_filepath):
    """Generates a JSON file containing execution logs and metadata."""
    metadata = {
        "execution_info": {
            "function": func_name,
            "duration": duration,
        },
        "parameters": params,
        "metrics_summary": metrics
    }
    
    # use the csv name to generate the .json file
    meta_path = base_filepath.replace('_raw.csv', '_metadata.json')
    if meta_path == base_filepath: 
        meta_path = base_filepath.replace('.csv', '_metadata.json')
        
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# ==========================================
# MATH FUNCTIONS
# ==========================================

def get_closest_center_idx(point: np.ndarray, centers: np.ndarray) -> int:
    """Finds the index of the closest center leveraging NumPy broadcasting."""
    distances = np.sum((centers - point) ** 2, axis=1)
    return int(np.argmin(distances))

def calculate_wcss(rdd_test, centers: list) -> float:
    """Calculates the Within-Cluster Sum of Squares (WCSS) for evaluation."""
    centers_np = np.array(centers)
    bc_centers = rdd_test.context.broadcast(centers_np)
    
    wcss = rdd_test.map(
        # Replaced distance_squared with direct numpy operation for consistency and speed
        lambda x: float(np.sum((x - bc_centers.value[get_closest_center_idx(x, bc_centers.value)]) ** 2))
    ).sum()
    
    bc_centers.destroy()
    return wcss

# ==========================================
# CLUSTERING ALGORITHMS
# ==========================================

def classic_kmeans(rdd_train, k: int, epochs: int):
    """Executes the standard K-Means algorithm using dense NumPy arrays with Spark broadcasting."""
    centers = rdd_train.takeSample(False, k)
    
    for _ in range(epochs):
        centers_np = np.array(centers)
        bc_centers = rdd_train.context.broadcast(centers_np)
        
        # Map using the broadcasted variable
        mapped_points = rdd_train.map(lambda x: (get_closest_center_idx(x, bc_centers.value), (x, 1)))
        
        # Reduce
        reduced_points = mapped_points.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        
        # Update centers
        new_centers_rdd = reduced_points.map(lambda x: (x[0], x[1][0] / x[1][1]))
        new_centers_dict = dict(new_centers_rdd.collect())
        
        # Rebuild centers list
        centers = [new_centers_dict.get(i, centers[i]) for i in range(k)]
        bc_centers.destroy()
        
    return centers

def minibatch_kmeans(rdd_train, k: int, b: int, epochs: int):
    """Executes the Mini-Batch K-Means algorithm distributing the load on Spark workers."""
    centers_list = rdd_train.takeSample(False, k)
    centers = np.array(centers_list) 
    
    v = np.zeros(k)
    
    # Calculate fraction for Spark's .sample()
    total_count = rdd_train.count()
    fraction = float(b) / total_count if total_count > 0 else 1.0
    
    for iteration in range(epochs):
        # Broadcast the centers
        bc_centers = rdd_train.context.broadcast(centers)
        
        # Distributed sampling on workers
        rdd_batch = rdd_train.sample(False, fraction)
        
        # Distributed mapping
        mapped_batch = rdd_batch.map(
            lambda x: (get_closest_center_idx(x, bc_centers.value), (x, 1))
        )
        
        # Distributed reduction, collecting only the k aggregates
        reduced_batch = mapped_batch.reduceByKey(
            lambda a, pt: (a[0] + pt[0], a[1] + pt[1])
        ).collect()
        
        # Local update on the Driver
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

def k_search(rdd_sample, k_list, epochs, num_iter, raw_csv, stats_csv):
    """Grid Search to find best K parameter among list of values."""
    start_time = time.time()
    results = []
    
    for run_id in range(num_iter):
        # Create Train and Test splits
        rdd_train, rdd_test = rdd_sample.randomSplit([0.8, 0.2], seed=run_id)
        rdd_train.cache()
        rdd_test.cache()

        for k in k_list:
            centers = classic_kmeans(rdd_train, k, epochs)
            wcss = calculate_wcss(rdd_test, centers)
            
            results.append({
                'k_value': k,
                'iteration_id': run_id,
                'performance_wcss': wcss
            })
            
        rdd_train.unpersist()
        rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    # Calculate statistics grouping by k_value
    df_stats = df_results.groupby('k_value').agg(
        mean_wcss=('performance_wcss', 'mean'),
        std_wcss=('performance_wcss', 'std')
    ).reset_index()
    df_stats.to_csv(stats_csv, index=False)
    
    # Use KneeLocator to find the mathematical elbow
    kl = KneeLocator(list(df_stats['k_value']), list(df_stats['mean_wcss']), curve="convex", direction="decreasing")
    optimal_k = kl.elbow if kl.elbow else k_list[0]

    duration = time.time() - start_time
    
    # Metadata
    save_metadata(
        func_name="k_search",
        duration= f"{duration} (s)",
        params={"k_list": k_list, "epochs": epochs, "iterations": num_iter},
        metrics={"optimal_k_found": int(optimal_k)},
        base_filepath=raw_csv
    )
    
    return optimal_k, df_stats

def b_search(rdd_sample, best_k, b_list, epochs, num_iter, raw_csv, stats_csv):
    """Grid Search to find best b parameter among list of values."""
    start = time.time()
    results = []
    
    for run_id in range(num_iter):
        rdd_train, rdd_test = rdd_sample.randomSplit([0.8, 0.2], seed=run_id)
        rdd_train.cache()
        rdd_test.cache()

        for b in b_list:
            start_time = time.time()
            centers = minibatch_kmeans(rdd_train, best_k, b, epochs)
            exec_time = time.time() - start_time
            
            wcss = calculate_wcss(rdd_test, centers)
            
            results.append({
                'batch_size': b,
                'iteration_id': run_id,
                'performance_wcss': wcss,
                'execution_time_sec': exec_time
            })
            
        rdd_train.unpersist()
        rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    df_stats = df_results.groupby('batch_size').agg(
        mean_wcss=('performance_wcss', 'mean'),
        std_wcss=('performance_wcss', 'std'),
        mean_time=('execution_time_sec', 'mean'),
        std_time=('execution_time_sec', 'std')
    ).reset_index()
    df_stats.to_csv(stats_csv, index=False)
    
    # Combine normalized scores to find the best batch size
    # Add logic OR to prevent by zero division
    wcss_range = (df_stats['mean_wcss'].max() - df_stats['mean_wcss'].min()) or 1.0
    time_range = (df_stats['mean_time'].max() - df_stats['mean_time'].min()) or 1.0

    # Equal wieghts to wcss and time: worst b value will have a combined score of 2.0, the best 0.0
    df_stats['combined_score'] = ((df_stats['mean_wcss'] - df_stats['mean_wcss'].min()) / wcss_range) + \
                                 ((df_stats['mean_time'] - df_stats['mean_time'].min()) / time_range)
    
    best_b = df_stats.loc[df_stats['combined_score'].idxmin(), 'batch_size']

    duration = time.time() - start
    
    # Metadata
    save_metadata(
        func_name="b_search",
        duration= f'{duration} (s)',
        params={"best_k_used": best_k, "b_list": b_list, "epochs": epochs, "iterations": num_iter},
        metrics={"optimal_b_found": int(best_b)},
        base_filepath=raw_csv
    )
    
    return int(best_b), df_stats

# ==========================================
# DIAGNOSTIC/ PERFORMANCE EVALUATION
# ==========================================

def cluster_diagnostic(evaluation_data_path, champion_centers):
    """
    Evaluates the Champion Model against the ground truth labels 
    using Adjusted and Normalized Mutual Information scores.
    Runs locally (no PySpark).
    """
    df_eval = pd.read_parquet(evaluation_data_path)
    
    # Convert the centers to a numpy array for the new get_closest_center_idx
    champion_centers_np = np.array(champion_centers)
    
    df_eval['predicted_cluster'] = df_eval['features'].apply(lambda x: get_closest_center_idx(x, champion_centers_np))
    
    labels_true = df_eval['true_labels'].tolist()
    labels_pred = df_eval['predicted_cluster'].tolist()
    
    ami_score = AMI(labels_true, labels_pred)
    nmi_score = NMI(labels_true, labels_pred)
    evaluated_documents = len(labels_true)
    
    sample_size = min(5000, len(df_eval))
    df_plot = df_eval.sample(n=sample_size, random_state=32).copy()
    
    df_plot['AMI_score'] = ami_score
    df_plot['NMI_score'] = nmi_score
    
    plot_file_path = evaluation_data_path.replace("evaluation.parquet", "plot_data.parquet")
    df_plot.to_parquet(plot_file_path, engine="pyarrow", index=False)

    return ami_score, nmi_score, evaluated_documents


# ==========================================
# FINAL MINI BATCH
# ==========================================

def mini_batch_run(rdd_data, evaluation_data_path, best_k, best_b, epochs, num_iter, raw_csv, stats_csv):
    """Runs the final comprehensive mini batch k-means using the discovered optimal parameters."""
    start = time.time()
    results = []

    best_wcss = float('inf')
    best_centers = []
    best_run_id = -1
    
    for run_id in range(num_iter):
        rdd_train, rdd_test = rdd_data.randomSplit([0.8, 0.2], seed=run_id)
        rdd_train.cache()
        rdd_test.cache()
        
        start_time = time.time()
        centers = minibatch_kmeans(rdd_train, best_k, best_b, epochs)
        exec_time = time.time() - start_time
        
        wcss = calculate_wcss(rdd_test, centers)

        if wcss < best_wcss:
            best_wcss = wcss
            best_centers = centers
            best_run_id = run_id
        

        
        results.append({
            'iteration_id': run_id,
            'k_value': best_k,
            'batch_size': best_b,
            'execution_time_sec': exec_time,
            'performance_wcss': wcss,
        })
        
        rdd_train.unpersist()
        rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    best_centers_check = [c.tolist() if isinstance(c, np.ndarray) else c for c in best_centers]

    stats_dict = {
        'mean_wcss': df_results['performance_wcss'].mean(),
        'std_wcss': df_results['performance_wcss'].std(),
        'mean_time': df_results['execution_time_sec'].mean(),
        'std_time': df_results['execution_time_sec'].std(),
        'champion_run_id': best_run_id,
        'champion_wcss': best_wcss,
        'best_centroids': json.dumps(best_centers_check) 
    }
    
    df_stats = pd.DataFrame([stats_dict])
    df_stats.to_csv(stats_csv, index=False)


    print("\n" + "="*40)
    print("PERFORMANCE DIAGNOSTIC")
    print("="*40)
    ami_score, nmi_score, evaluated_documents = cluster_diagnostic(evaluation_data_path, best_centers)
    print(f'AMI Score: {ami_score}, \nNMI Score: {nmi_score}, \nDiagnostic on {evaluated_documents} Documents')

    duration = time.time() - start

    save_metadata(
        func_name="mini_batch_run",
        duration= f"{duration} (s)",
        params={"best_k": best_k, "best_b": best_b, "epochs": epochs, "iterations": num_iter},
        metrics={"champion_run_id": best_run_id,
            "champion_wcss": best_wcss,
            "champion_centers": best_centers_check,
            "AMI Score" : ami_score,
            "NMI Score" : nmi_score},
        base_filepath=raw_csv
    )
    
    return df_stats