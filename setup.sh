#!/bin/bash

# Load ips.env variables
set -a
source ips.env
set +a

echo "Creating python enviroment in current directory..."

# Creates enviroment
python3 -m venv pyvenv

#Install packages dependencies saved on packages.txt
if [ -f "packages.txt" ]; then
    echo "Installing dependencies..."
    ./pyvenv/bin/pip install -r packages.txt
else
    echo "[WARNING] packages.txt not found! Please make sure to have it."
    exit 0
fi

# Cluster or Local check. If WORKER_IPS="" then we are in local otherwise on cluster
if [ -z "$WORKER_IPS" ]; then
    echo "[INFO] YOU'RE IN A LOCAL ENVIROMENT -> Enviroment generated: ready to work!"
    exit 0
fi

# Takes worker user defined in ips.env, if for any reason does not find it, try with the master's user
WORKER_USER="${WORKER_USER:-$USER}"

# Creates iterable list of ips (Corretto IPs in IFS)
IFS=',' read -r -a WORKER_ARRAY <<< "$WORKER_IPS"

echo "[INFO] YOU'RE IN A CLUSTER WITH ${#WORKER_ARRAY[@]} WORKERS -> Starting Enviroment Distribution..."

# Iterate scp for every IP address
i=1
for IP in "${WORKER_ARRAY[@]}"; do
    
    # Remove accidentally placed spaces between IPs
    IP=$(echo $IP | xargs)
    
    if [ -n "$IP" ]; then
        echo "Copying on Worker $i: $IP..."
        
        # Create directory with the master's path plus S3 (to specify this application version)
        ssh -o StrictHostKeyChecking=no -q "$WORKER_USER@$IP" "mkdir -p \"$PWD\"S3"
        
        rsync -az -e "ssh -o StrictHostKeyChecking=no" pyvenv/ "$WORKER_USER@$IP:${PWD}S3/pyvenv/"
        
        echo "Done!"
        ((i++))
    fi
done

echo "Cluster Setup Completed. You are ready to work!"
