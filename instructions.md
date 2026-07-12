# Group 2612 Project – Mini-Batch K-Means

## Developers

- Kevin Brugnera
- Riccardo Ferrante
- Marco Lorenzato
- Federico Scianna

This project implements a distributed Mini-Batch K-Means workflow using Apache Spark. The framework is designed to run on both a local machine and a multi-node Linux cluster, while providing a consistent execution environment across all nodes.

> **Important:** All commands must be executed from the root `Project/` directory.

---

# Workflow Overview

The complete workflow consists of five sequential stages:

1. Environment setup
2. Data generation and distribution
3. Grid search for the optimal Mini-Batch size
4. Benchmark execution and champion model selection
5. Model diagnostics and validation

Each stage depends on the output of the previous one, so they should normally be executed in order.

---

# 1. Environment Setup

Before running any script, configure your execution environment.

### 1.1 Configure `ips.env`

Open the `ips.env` file and set the appropriate variables.

### Local execution

```bash
WORKER_IPS=""
```

### Cluster execution

```bash
WORKER_IPS="worker1_ip,worker2_ip,..."
MASTER_IP="master_node_ip"
WORKER_USER="username"
```

We assume that every worker node uses the same username.

### Optional: CloudVeneto S3

If you want to store the generated dataset on CloudVeneto Object Storage, also configure:

```bash
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
S3_ENDPOINT="https://..."
S3_BUCKET="bucket_name"
```

These credentials are used during data generation and by all Spark jobs that read the dataset from object storage.

### 1.2 Run the setup script

Execute

```bash
./setup.sh
```

The setup script:

- creates the Python virtual environment;
- installs all required dependencies;
- configures the master node;
- automatically replicates the environment on every worker node listed in `WORKER_IPS`.

---

# 2. Data Generation and Distribution

### 2.1 Activate the virtual environment

```bash
source pyvenv/bin/activate
```

### 2.2 Generate the dataset

```bash
python data_generation.py
```

The script automatically behaves according to your configuration:

- **Local execution**
  - generates the RCV1 dataset locally.

- **Cluster without S3**
  - generates the dataset and distributes it across the worker nodes.

- **Cluster with S3 configured**
  - generates the dataset and uploads the Parquet files directly to your CloudVeneto S3 bucket.

This step is required before running any clustering experiment.

---

# 3. Grid Search for the Optimal Mini-Batch Size

Before launching Spark jobs, open

```text
run_script.sh
```

and configure the Spark session according to your execution environment (local or cluster).

Then execute

```bash
./run_script.sh grid_search.py
```

This stage performs a hyperparameter search over the values defined in `B_RANGE`.

For each batch size, it computes the Calinski-Harabasz Index and execution time in order to identify the best compromise between clustering quality and computational efficiency.

The selected batch size is saved in

```text
runs/b_search_metadata.json
```

---

# 4. Benchmark and Champion Model Selection

Again, verify that the Spark configuration inside `run_script.sh` matches your environment.

Run

```bash
./run_script.sh benchmark.py
```

Using the optimal batch size found during the previous stage, the benchmark executes **50 independent runs** of Mini-Batch K-Means.

Among all runs, the model with the lowest Within-Cluster Sum of Squares (WCSS) is selected as the **Champion Model**.

The corresponding centroids are exported to

```text
runs/stats_strong_12c.csv
```

---

# 5. Diagnostic and Performance Validation

Verify the Spark configuration if necessary, then run

```bash
./run_script.sh diagnostic.py
```

The diagnostic script loads the Champion Model centroids and evaluates them on the validation dataset.

The following clustering metrics are reported:

- Adjusted Mutual Information (AMI)
- Normalized Mutual Information (NMI)

These scores quantify how well the learned clustering matches the ground-truth labels.

---

# Notes

> **Spark configuration**
>
> Before executing any Spark job (`grid_search.py`, `benchmark.py`, or `diagnostic.py`), ensure that the Spark session inside `run_script.sh` is configured for your current environment. Local and cluster executions typically require different memory allocations, executor settings, and networking parameters.

> **Repeated configuration**
>
> If you are running all experiments using the same environment, you only need to configure `run_script.sh` once. There is no need to modify it before every stage.

> **Execution order**
>
> The recommended execution sequence is:
>
> 1. `./setup.sh`
> 2. `python data_generation.py`
> 3. `./run_script.sh grid_search.py`
> 4. `./run_script.sh benchmark.py`
> 5. `./run_script.sh diagnostic.py`