# Group 2612 Project - MiniBatch K-Means algorithm
## Developers
* Kevin Brugnera
* Riccardo Ferrante
* Marco Lorenzato
* Federico Scianna

In order to implement and use our workflow for the parallelization of K-Means clustering, follow this instructions and use them whenever you have doubts. We tried our best to develop a framework that can be used by anyone using a Linux OS, enforcing consistency and setting up safe enviroments to run the scripts. 

**WARNING: Everything should be run insisde the `Project/` directory.**

## 1. Enviroment Setup
1. Open the `ips.env` file:

   - **LOCAL RUN**
     - Set `WORKER_IPS=""` (empty string).

   - **CLUSTER**
     - Set `WORKER_IPS="worker_1_ip,...,worker_n_ip"`.
     - Set `MASTER_IP="master_node_ip"`.
     - Set `WORKER_USER="username"`.

    We assume that every node uses the same username for convenience.

These variables will be used throughout the rest of the project. **Set them properly!**

**2.** Run `./setup.sh` command.

## 2. Data Generation and Distribution
**1.** Activate enviroment with `source pyvenv/bin/activate` command.

**2.** Run `python data_generation.py` command.

## 3. Grid Search of MiniBatch size (b)
**1.** Open `run_script.sh`

**2.** Adjust spark session specific in the right section (**Local and cluster enviroment have different specs**).

**3.** Run `./run_script.sh batch_initialization.py` command.

## 4. Benchamrk Runs
**1.** Open `run_script.sh`

**2.** Adjust spark session specific in the right section (**Local and cluster enviroment have different specs**).

**3.** Run `./run_script.sh benchmark.py` command.

## 5. Diagnostic
**1.** Open `run_script.sh`

**2.** Adjust spark session specific in the right section (**Local and cluster enviroment have different specs**).

**3.** Run `./run_script.sh diagnostic.py` command.

**INFO: Of course you can skip point **1.** and **2.** of every step if you want to run everything with the same specific.**
