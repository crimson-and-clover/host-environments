#!/usr/bin/env bash
set -euo pipefail

# Configuration - can be overridden by environment variables
USER_NAME="${USER_NAME:-developer}"

# modify here
CONTAINER_NAME="${CONTAINER_NAME:-huyu-cuda11.8}"
HOSTNAME="${HOSTNAME:-3090x8_cu11}"
IMAGE_NAME="${IMAGE_NAME:-huyu/cuda:11.8-ubuntu22.04}"

# SSD and HDD mount arrays
SSD_ROOT=(
    "/home/huyu/ssd1/container"
)

HDD_ROOT=(
    "/home/huyu/hdd1/container"
    "/home/huyu/hdd2/container"
)

SSH_PORT="${SSH_PORT:-22000}"

USER_ID="${USER_ID:-1017}"
GROUP_ID="${GROUP_ID:-${USER_ID}}"

function deploy() {
    # create data directories for SSD arrays
    for ssd_path in "${SSD_ROOT[@]}"; do
        mkdir -p "${ssd_path}/data"
    done
    
    # create data directories for HDD arrays
    for i in "${!HDD_ROOT[@]}"; do
        hdd_path="${HDD_ROOT[$i]}"
        mkdir -p "${hdd_path}/data"
        
        # create home directory only for the first HDD (index 0)
        if [ "$i" -eq 0 ]; then
            mkdir -p "${hdd_path}/home"
        fi
    done
    
    USER_HOME="/home/$USER_NAME"
    
    # Build docker run command array
    DOCKER_CMD=(
        docker run -d
        --name="$CONTAINER_NAME"
        --restart=unless-stopped
        --gpus=all
        --privileged
        --shm-size=4g
        --ulimit memlock=-1
        --ulimit stack=67108864
        --hostname="$HOSTNAME"
        -p "0.0.0.0:$SSH_PORT:22/tcp"
    )
    
    # Add volume mounts
    # mount first HDD home directory to USER_HOME
    DOCKER_CMD+=(-v "${HDD_ROOT[0]}/home:$USER_HOME")
    
    # mount all SSD data directories
    for i in "${!SSD_ROOT[@]}"; do
        DOCKER_CMD+=(-v "${SSD_ROOT[$i]}/data:$USER_HOME/ssd$((i+1))")
    done
    
    # mount all HDD data directories
    for i in "${!HDD_ROOT[@]}"; do
        DOCKER_CMD+=(-v "${HDD_ROOT[$i]}/data:$USER_HOME/hdd$((i+1))")
    done
    
    # Add image name and execute
    DOCKER_CMD+=("$IMAGE_NAME")
    
    # Execute docker run
    "${DOCKER_CMD[@]}"
    
    # TODO add group video render
    # according /dev/dri
    docker exec -it "$CONTAINER_NAME" bash -x /src/init_user.sh "$USER_NAME" "$USER_ID" "$GROUP_ID" || {
        echo "Error: Failed to initialize user. Container may need to be removed manually."
        exit 1
    }
}

deploy
