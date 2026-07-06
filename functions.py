import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from kneed import KneeLocator
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.evaluation import ClusteringEvaluator

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

def spark_silhouette_score(rdd_test, centers):
    """
    Evaluate Silhouette Score using pySpark implementation.
    """
    centers_np = np.array(centers)
    bc_centers = rdd_test.context.broadcast(centers_np)
    
    # elper function to map feature and prediction labels in MLlib reuqested format
    def map_to_row(x):
        c_idx = get_closest_center_idx(x, bc_centers.value)
        return Row(features=Vectors.dense(x), prediction=int(c_idx))
    
    schema = StructType([
        StructField("features", VectorUDT(), False),
        StructField("prediction", IntegerType(), False)
    ])
    
    # Convert rdd into dataframe using defined schema
    mapped_rdd = rdd_test.map(map_to_row)
    predictions_df = mapped_rdd.toDF(schema=schema)   
    
    # Use pyspark evaluator to compute silhouette score
    evaluator = ClusteringEvaluator(
        predictionCol="prediction", 
        featuresCol="features", 
        metricName="silhouette", 
        distanceMeasure="squaredEuclidean"
    )
    
    score = evaluator.evaluate(predictions_df)
    
    bc_centers.destroy()
    return score

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

        # Actions to trigger the cache on worker's RAM
        train_size = rdd_train.count() 
        test_size = rdd_test.count()

        for k in k_list:
            # Sanity check
            if k < 2:
                print(f'K={k} will have Silhouette Score -1. This metric requires at least two clusters.' )
                continue

            print(f'Testing K={k}, iteration:{run_id}')
            centers = classic_kmeans(rdd_train, k, epochs)
            sil_score = spark_silhouette_score(rdd_test, centers)
            
            results.append({
                'k_value': k,
                'iteration_id': run_id,
                'silhouette_score': sil_score
            })
            
        rdd_train.unpersist()
        rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    
    print("Calculating Final Metrics...")
    # Calculate statistics grouping by k_value
    df_stats = df_results.groupby('k_value').agg(
        mean_silhouette=('silhouette_score', 'mean'),
        std_silhouette=('silhouette_score', 'std')
    ).reset_index()
    df_stats.to_csv(stats_csv, index=False)
    
  # Best K maximizes the silhouette score
    best_row_idx = df_stats['mean_silhouette'].idxmax()
    optimal_k = df_stats.loc[best_row_idx, 'k_value']

    duration = time.time() - start_time
    
    # Metadata
    save_metadata(
        func_name="k_search",
        duration= f"{duration} (s)",
        params={"k_list": k_list, "epochs": epochs, "iterations": num_iter},
        metrics={"optimal_k_found": int(optimal_k)},
        base_filepath=raw_csv
    )
    print("Done!")

    return optimal_k, df_stats

def b_search(rdd_sample, best_k, b_list, docs_volume, num_iter, raw_csv, stats_csv):
    """Grid Search to find best b parameter among list of values."""
    start = time.time()
    results = []
    epochs_values = []
    
    for run_id in range(num_iter):
        rdd_train, rdd_test = rdd_sample.randomSplit([0.8, 0.2], seed=run_id)
        rdd_train.cache()
        rdd_test.cache()

        # Actions to trigger the cache on worker's RAM
        train_size = rdd_train.count() 
        test_size = rdd_test.count()

        for b in b_list:
            #For every batch size the same volume of docs has to be evaluated in order not to falsify time results
            epochs = max(1,int(docs_volume/b))
            epochs_values.append(epochs)

            print(f'Testing Mini-Batch Size={b}, iteration:{run_id}')
            start_time = time.time()
            centers = minibatch_kmeans(rdd_train, best_k, b, epochs)
            exec_time = time.time() - start_time
            
            sil_score = spark_silhouette_score(rdd_test, centers)
            
            results.append({
                'batch_size': b,
                'iteration_id': run_id,
                'silhouette_score': sil_score,
                'execution_time': exec_time
            })
            
        rdd_train.unpersist()
        rdd_test.unpersist()

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_csv, index=False)
    print("Calculating Final Metrics...")
    
    df_stats = df_results.groupby('batch_size').agg(
        mean_silhouette=('silhouette_score', 'mean'),
        std_silhouette=('silhouette_score', 'std'),
        mean_time=('execution_time_sec', 'mean'),
        std_time=('execution_time_sec', 'std')
    ).reset_index()
    df_stats.to_csv(stats_csv, index=False)

    # Introduce a tolerance of 2% decrease in silhouette score to compute best_b
    best_sil = df_stats['mean_silhouette'].max()
    tolerance = 0.02 
    accepted_sil = best_sil * (1.0 - tolerance)
    mask_sil = df_stats['mean_silhouette'] >= accepted_sil

    #Filter the DataFrame with only accepted silhouette scores
    accepted_batch = df_stats[mask_sil]
    # best_b is the one that minimizes run time among these values
    mask_b = accepted_batch['mean_time'].idxmin()
    best_b = accepted_batch.loc[mask_b, 'batch_size']
    

    duration = time.time() - start
    
    # Metadata
    save_metadata(
        func_name="b_search",
        duration= f'{duration} (s)',
        params={"best_k_used": best_k, "b_list": b_list, "epochs": epochs_values, "iterations": num_iter},
        metrics={"optimal_b_found": int(best_b)},
        base_filepath=raw_csv
    )

    print("Done!")
    
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
    
    plot_file_path = evaluation_data_path.replace("evaluation_dataset", "plot_data.parquet")
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
        
        # Actions to trigger the cache on worker's RAM
        train_size = rdd_train.count() 
        test_size = rdd_test.count()
        
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