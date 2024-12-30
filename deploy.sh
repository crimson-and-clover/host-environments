#!/usr/bin/env bash

USER_NAME="developer"

# modify here
CONTAINER_NAME="huyu-cuda11.8"
HOSTNAME="3090x8_cu11"
IMAGE_NAME="huyu/cuda:11.8-ubuntu22.04"
HDD_ROOT=/home/hdd/huyu/container/
SSD_ROOT=/home/huyu/container/
SSH_PORT=22000

USER_ID=1017
GROUP_ID=1017

function deploy() {
    # create data directory
    mkdir -p "$SSD_ROOT/data"
    mkdir -p "$HDD_ROOT/data"
    mkdir -p "$HDD_ROOT/home"
    
    USER_HOME="/home/$USER_NAME"
    
    docker run -d --name="$CONTAINER_NAME" \
    --restart=unless-stopped \
    --gpus=all \
    --privileged \
    --shm-size=2g \
    --hostname="$HOSTNAME" \
    -p "0.0.0.0:$SSH_PORT:22/tcp" \
    -v "$HDD_ROOT/home:$USER_HOME" \
    -v "$SSD_ROOT/data:$USER_HOME/ssd" \
    -v "$HDD_ROOT/data:$USER_HOME/hdd" \
    "$IMAGE_NAME"
    
    docker exec -it "$CONTAINER_NAME" bash -v /src/init_user.sh "$USER_ID" "$GROUP_ID"
}

deploy
