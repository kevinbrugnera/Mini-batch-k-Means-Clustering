import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import normalized_mutual_info_score as NMI

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
    
    # Broadcasts centers to executors
    bc_centers = rdd_test.context.broadcast(centers_np)
    
    wcss = rdd_test.map(
        lambda x: float(np.sum((x - bc_centers.value[closest_idx(x, bc_centers.value)]) ** 2))
    ).sum()
    
    #Clean executors RAM
    bc_centers.destroy()

    return wcss

def calinski_harabasz(rdd_test, centers: list, k: int) -> float:
    """Calculates the Calinski-Harabasz Index for best_b identification."""
    N = rdd_test.count()
    
    # Compute gloabl centroid by summing all point vectors and dividing by N
    global_sum = rdd_test.reduce(lambda a, b: a + b)
    global_mean = global_sum / float(N)
    
    # Total Sum of Squares. We use total variance rule to get BCSS later (TSS = BCSS + wcss_score)
    bc_global_mean = rdd_test.context.broadcast(global_mean)

    # Sum of Squares between each point of test set and global center found
    tss = rdd_test.map(
        lambda x: float(np.sum((x - bc_global_mean.value) ** 2))
    ).sum()

    #Clean executors RAM
    bc_global_mean.destroy()
    
    #Evaluates wcss_score using designed functions
    wcss = WCSS(rdd_test, centers)
    
    #BCSS Evaluation
    bcss = tss - wcss
    
    # Avoid zero divisions (just in case)
    if wcss == 0:
        return float('inf')
        
    ch_score = (bcss / (k - 1)) / (wcss / (N - k))
    
    return ch_score

# ==========================================
# CLUSTERING ALGORITHMS
# ==========================================

def classic_kmeans(rdd_train, k: int, epochs: int, seed: int):
    """Executes the standard K-Means algorithm using Spark."""

    # Initialize centers randomly from training set
    centers = rdd_train.takeSample(False, k, seed)
    
    for _ in range(epochs):
        #Broadcast
        centers_np = np.array(centers)
        bc_centers = rdd_train.context.broadcast(centers_np)
        
        # Map using the broadcasted variable, returns -> (center_idx, (point,count)) of course count will always be 1
        mapped_points = rdd_train.map(lambda x: (closest_idx(x, bc_centers.value), (x, 1)))
        
        # Reduce by key (center_idx) and gives -> (center_idx, (vectorial_sum, population))
        reduced_points = mapped_points.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        
        # Update centers and gives them to master -> (center_idc, vectorial_sum/population)
        new_centers_rdd = reduced_points.map(lambda x: (x[0], x[1][0] / x[1][1]))
        new_centers_dict = dict(new_centers_rdd.collect())
        
        # Update centers for next iteration
        centers = [new_centers_dict.get(i, centers[i]) for i in range(k)]
        bc_centers.destroy()
        
    return centers

def minibatch_kmeans(rdd_train, k: int, b: int, epochs: int, seed: int):
    """Executes the Mini-Batch K-Means algorithm using Spark."""
    centers_list = rdd_train.takeSample(False, k, seed)
    centers = np.array(centers_list) 
    
    v = np.zeros(k)
    
    # Calculate fraction for Spark's .sample()
    total_count = rdd_train.count()
    fraction = min(1, float(b)/total_count)
    
    for epoch in range(epochs):
        # Broadcast the centers
        bc_centers = rdd_train.context.broadcast(centers)
        
        # Distributed sampling on workers
        rdd_batch = rdd_train.sample(False, fraction)
        
        # Distributed mapping --> (cluster_idx: (point,1))
        mapped_batch = rdd_batch.map(
            lambda x: (closest_idx(x, bc_centers.value), (x, 1))
        )
        
        # Distributed reduction, collecting only the k aggregates
        reduced_batch = mapped_batch.reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
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

def b_search(rdd_sample, K, b_list, epochs, num_iter, raw_csv, stats_csv):
    """Grid Search to find best b parameter among list of values."""
    start = time.time()
    results = []

    rdd_train, rdd_test = rdd_sample.randomSplit([0.8, 0.2], seed=1)
    rdd_train.persist() 
    rdd_test.persist()  

    # Actions to trigger the cache on worker's RAM
    train_size = rdd_train.count() 
    test_size = rdd_test.count()
    
    for run_id in range(num_iter):

        for b in b_list:

            print(f'Testing Mini-Batch Size={b}, iteration:{run_id}')
            start_time = time.time()
            centers = minibatch_kmeans(rdd_train, K, b, epochs, seed=run_id)
            exec_time = time.time() - start_time
            
            ch_score = calinski_harabasz(rdd_test, centers, K)
            
            results.append({
                'batch_size': b,
                'iteration_id': run_id,
                'ch_score': ch_score,
                'execution_time': exec_time
            })
            
    rdd_train.unpersist()
    rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    print("Calculating Final Metrics...")
    
    df_stats = df_results.groupby('batch_size').agg(
        mean_ch=('ch_score', 'mean'),
        std_ch=('ch_score', 'std'),
        mean_time=('execution_time', 'mean'),
        std_time=('execution_time', 'std')
    ).reset_index()
    df_stats.to_csv(stats_csv, index=False)

    
    # Introduce a tolerance of 2% decrease in silhouette score to compute best_b
    best_ch_score = df_stats['mean_ch'].max()
    tolerance = 0.025
    accepted_ch = best_ch_score * (1.0 - tolerance)
    mask_ch = df_stats['mean_ch'] >= accepted_ch

    #Filter the DataFrame with only accepted silhouette scores
    accepted_batch = df_stats[mask_ch]
    # best_b is the one that minimizes run time among these values
    mask_b = accepted_batch['mean_time'].idxmin()
    best_b = accepted_batch.loc[mask_b, 'batch_size']
    

    duration = time.time() - start
    
    # Metadata
    save_metadata(
        func_name="b_search",
        duration= f'{duration} (s)',
        params={"best_k_used": K, "b_list": b_list, "epochs": epochs, "iterations": num_iter},
        metrics={"optimal_b_found": int(best_b)},
        base_filepath=raw_csv
    )

    print("Done!")
    
    return int(best_b), df_stats

# ==========================================
# DIAGNOSTIC/ PERFORMANCE EVALUATION
# ==========================================

def cluster_diagnostic(rdd_data, champion_centers, plot_csv):
    """
    Evaluates the Champion Model against the ground truth labels 
    using Adjusted and Normalized Mutual Information scores.
    Runs locally (no PySpark).
    """

    start = time.time()

    
    # Convert the centers to a numpy array for the new closest_idx
    centers_np = np.array(champion_centers)
    bc_centers = rdd_data.context.broadcast(centers_np)

    def predict_label(row):
        # Converts feature in array
        point = np.array(row['features'])
    
        pred = closest_idx(point, bc_centers.value)
    
        return (row['true_labels'], pred)

    labels_rdd = rdd_data.map(predict_label)
    labels_local = labels_rdd.collect()

    labels_true = [x[0] for x in labels_local]
    labels_pred = [x[1] for x in labels_local]
    
    print("Evaluating Clustering Performance...")
    # Metrics evaluation on master node
    nmi_score = NMI(labels_true, labels_pred)
    evaluated_documents = len(labels_true)
    
    #To visualize clusters we just need a few points. We let this sampling process to the spark session
    fraction = min(1, 5000/evaluated_documents)
    sampled_rows = rdd_data.sample(False, fraction).collect()

    df_plot = pd.DataFrame([row.asDict() for row in sampled_rows])
    
    #Evaluate locally the predicetd labels(easy to do on only 5000 documents)
    df_plot['predicted_labels'] = df_plot['features'].apply(lambda x: closest_idx(np.array(x), centers_np))
    df_plot['NMI_score'] = nmi_score

    df_plot.to_csv(plot_csv, index=False)

    duration = time.time() - start

     # Metadata
    save_metadata(
        func_name="cluster_diagnostic",
        duration= f'{duration} (s)',
        params= 'Champion run centers',
        metrics={"NMI": nmi_score},
        base_filepath=plot_csv
    )

    print("Done!")
    print(f'NMI Score: {nmi_score}')
    
    bc_centers.destroy()

# ==========================================
# FINAL MINI BATCH
# ==========================================

def mini_batch_run(rdd_data, K, best_b, epochs, num_iter, raw_csv, stats_csv):
    """Runs the final comprehensive mini batch k-means using the discovered optimal parameters."""
    start = time.time()
    results = []

    best_wcss = float('inf')
    best_centers = []
    best_run_id = -1

    rdd_train, rdd_test = rdd_data.randomSplit([0.8, 0.2],seed=18)
    rdd_train.persist() 
    rdd_test.persist()  

     # Actions to trigger the cache on worker's RAM
    train_size = rdd_train.count() 
    test_size = rdd_test.count()
    
    for run_id in range(num_iter):
        print(f"MiniBatch K-means, {epochs} epochs -- Iteration: {run_id}")
        
        start_time = time.time()
        centers = minibatch_kmeans(rdd_train, K, best_b, epochs, seed=run_id)
        exec_time = time.time() - start_time
        
        wcss = WCSS(rdd_test, centers)

        if wcss < best_wcss:
            best_wcss = wcss
            best_centers = centers
            best_run_id = run_id
        

        
        results.append({
            'iteration_id': run_id,
            'k_value': K,
            'batch_size': best_b,
            'execution_time': exec_time,
            'wcss': wcss,
        })
        
    rdd_train.unpersist()
    rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    best_centers_check = [c.tolist() if isinstance(c, np.ndarray) else c for c in best_centers]

    stats_dict = {
        'mean_wcss': df_results['wcss'].mean(),
        'std_wcss': df_results['wcss'].std(),
        'mean_time': df_results['execution_time'].mean(),
        'std_time': df_results['execution_time'].std(),
        'champion_run_id': best_run_id,
        'champion_wcss_score': best_wcss,
        'champion_centroids': json.dumps(best_centers_check) 
    }
    
    df_stats = pd.DataFrame([stats_dict])
    df_stats.to_csv(stats_csv, index=False)

    duration = time.time() - start

    save_metadata(
        func_name="mini_batch_run",
        duration= f"{duration} (s)",
        params={"K": K, "b": best_b, "epochs": epochs, "iterations": num_iter},
        metrics={"champion_run_id": best_run_id,
            "champion_wcss": best_wcss,
            },
        base_filepath=raw_csv
    )
    
    return df_stats

def classic_kmeans_run(rdd_data, K, epochs, num_iter, raw_csv, stats_csv):
    """Runs the final comprehensive Classic Full-Batch k-means maintaining identical telemetry."""
    start = time.time()
    results = []

    best_wcss = float('inf')
    best_centers = []
    best_run_id = -1

    # USe the same seed as mini_batch_run for consistency
    rdd_train, rdd_test = rdd_data.randomSplit([0.8, 0.2], seed=18)
    rdd_train.persist() 
    rdd_test.persist()  

     # Actions to trigger the cache on worker's RAM
    train_size = rdd_train.count() 
    test_size = rdd_test.count()
    
    for run_id in range(num_iter):
        print(f"Classic K-means, {epochs} epochs -- Iteration: {run_id}")
        
        start_time = time.time()
        centers = classic_kmeans(rdd_train, K, epochs, seed=run_id)
        exec_time = time.time() - start_time
        
        wcss = WCSS(rdd_test, centers)

        if wcss < best_wcss:
            best_wcss = wcss
            best_centers = centers
            best_run_id = run_id
        
        results.append({
            'iteration_id': run_id,
            'k_value': K,
            'execution_time': exec_time,
            'wcss': wcss,
        })
        
    rdd_train.unpersist()
    rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    best_centers_check = [c.tolist() if isinstance(c, np.ndarray) else c for c in best_centers]

    stats_dict = {
        'mean_wcss': df_results['wcss'].mean(),
        'std_wcss': df_results['wcss'].std(),
        'mean_time': df_results['execution_time'].mean(),
        'std_time': df_results['execution_time'].std(),
        'champion_run_id': best_run_id,
        'champion_wcss_score': best_wcss,
        'champion_centroids': json.dumps(best_centers_check) 
    }
    
    df_stats = pd.DataFrame([stats_dict])
    df_stats.to_csv(stats_csv, index=False)

    duration = time.time() - start

    save_metadata(
        func_name="classic_kmeans_run",
        duration=f"{duration} (s)",
        params={"K": K, "epochs": epochs, "iterations": num_iter},
        metrics={"champion_run_id": best_run_id,
                 "champion_wcss": best_wcss,
                },
        base_filepath=raw_csv
    )

    return df_stats
