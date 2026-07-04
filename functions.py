import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from kneed import KneeLocator

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

def distance_squared(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the squared Euclidean distance between two dense NumPy arrays."""
    return float(np.sum((v1 - v2) ** 2))

def get_closest_center_idx(point: np.ndarray, centers: list) -> int:
    """Finds the index of the closest center for a given point."""
    distances = [distance_squared(point, c) for c in centers]
    return int(np.argmin(distances))

def calculate_wcss(rdd_test, centers: list) -> float:
    """Calculates the Within-Cluster Sum of Squares (WCSS) for evaluation."""
    wcss = rdd_test.map(
        lambda x: distance_squared(x, centers[get_closest_center_idx(x, centers)])
    ).sum()
    return wcss

# ==========================================
# CLUSTERING ALGORITHMS
# ==========================================

def classic_kmeans(rdd_train, k: int, epochs: int):
    """Executes the standard K-Means algorithm using dense NumPy arrays."""
    # Initialize random centers from the dataset
    centers = rdd_train.takeSample(False, k)
    
    for _ in range(epochs):
        # Map: assign each point to the closest cluster index -> (cluster_idx, (point_array, 1))
        mapped_points = rdd_train.map(lambda x: (get_closest_center_idx(x, centers), (x, 1)))
        
        # Reduce: sum the arrays and the counts point-wise -> (cluster_idx, (total_coordinate_sum, population))
        reduced_points = mapped_points.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        
        # Calculate new centers by dividing the summed array by the population count -> (cluster_idx, new_center)
        new_centers_rdd = reduced_points.map(lambda x: (x[0], x[1][0] / x[1][1]))
        
        # Collect and update the centers list
        new_centers_dict = dict(new_centers_rdd.collect())
        centers = [new_centers_dict.get(i, centers[i]) for i in range(k)]
    
    # Returns the fianl list of clusters centers    
    return centers

def minibatch_kmeans(rdd_train, k: int, b: int, epochs: int):
    """Executes the Mini-Batch K-Means algorithm using NumPy arrays."""
    centers = rdd_train.takeSample(False, k)
    v = np.zeros(k) # Tracks the number of points assigned to each center for learning rate
    
    for iteration in range(epochs):
        # Extract the mini-batch
        M = rdd_train.takeSample(False, b)
        
        # Cache closest centers for the batch
        d = []
        for x in M:
            d.append(get_closest_center_idx(x, centers))
            
        # Update centers based on the batch
        for i, x in enumerate(M):
            c_idx = d[i]                           
            v[c_idx] += 1                          
            eta = 1.0 / v[c_idx]                   
            
            # Update center with gradient step
            centers[c_idx] = (centers[c_idx] * (1.0 - eta)) + (x * eta)
            
    return centers

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

def mini_batch_run(rdd_data, best_k, best_b, num_iter, raw_csv, stats_csv):
    """Runs the final comprehensive mini batch k-means using the discovered optimal parameters."""
    start = time.time()
    results = []
    
    for run_id in range(num_iter):
        rdd_train, rdd_test = rdd_data.randomSplit([0.8, 0.2], seed=run_id)
        rdd_train.cache()
        rdd_test.cache()
        
        start_time = time.time()
        centers = minibatch_kmeans(rdd_train, best_k, best_b, t=16)
        exec_time = time.time() - start_time
        
        wcss = calculate_wcss(rdd_test, centers)
        
        # Calculate cluster populations for structural validation -> (cluster_idx, population)
        population_dict = rdd_test.map(lambda x: (get_closest_center_idx(x, centers), 1)) \
                                  .reduceByKey(lambda a, b: a + b) \
                                  .collectAsMap()
        
        results.append({
            'iteration_id': run_id,
            'k_value': best_k,
            'batch_size': best_b,
            'execution_time_sec': exec_time,
            'performance_wcss': wcss,
            'cluster_population': str(population_dict)
        })
        
        rdd_train.unpersist()
        rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    # Generate summary statistics
    df_stats = df_results.agg(
        mean_wcss=('performance_wcss', 'mean'),
        std_wcss=('performance_wcss', 'std'),
        mean_time=('execution_time_sec', 'mean'),
        std_time=('execution_time_sec', 'std')
    )
    
    df_stats.to_csv(stats_csv, index=False)

    duration = time.time() - start
    
    # Metadata
    save_metadata(
        func_name="final_benchmark",
        duration= f'{duration} (s)',
        params={"best_k": best_k, "best_b": best_b, "iterations": num_iter},
        metrics={"status": "Success"},
        base_filepath=raw_csv
    )
    
    return df_stats