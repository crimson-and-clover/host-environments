#!/usr/bin/env bash

USER_NAME="developer"

# modify here
CONTAINER_NAME="huyu-cuda11.8"
HOSTNAME="3090x8_cu11"
IMAGE_NAME="huyu/cuda:11.8-ubuntu22.04"
HDD_ROOT=/home/hdd/huyu/container/
SSD_ROOT=/home/huyu/container/
SSH_PORT=22000

function deploy() {
    # create data directory
    mkdir -p "$SSD_ROOT/data"
    mkdir -p "$HDD_ROOT/data"
    mkdir -p "$HDD_ROOT/anaconda3/envs"
    mkdir -p "$HDD_ROOT/anaconda3/pkgs"
    mkdir -p "$HDD_ROOT/vscode-server"
    mkdir -p "$HDD_ROOT/cache/huggingface"
    mkdir -p "$HDD_ROOT/cache/pip"
    
    
    USER_HOME="/home/$USER_NAME"
    
    docker run -d --name="$CONTAINER_NAME" \
    --restart=unless-stopped \
    --gpus=all \
    --privileged \
    --shm-size=2g \
    --hostname="$HOSTNAME" \
    -p "0.0.0.0:$SSH_PORT:22/tcp" \
    -v "$SSD_ROOT/data:$USER_HOME/ssd" \
    -v "$HDD_ROOT/data:$USER_HOME/hdd" \
    -v "$HDD_ROOT/anaconda3/envs:$USER_HOME/anaconda3/envs" \
    -v "$HDD_ROOT/anaconda3/pkgs:$USER_HOME/anaconda3/pkgs" \
    -v "$HDD_ROOT/vscode-server:$USER_HOME/.vscode-server" \
    -v "$HDD_ROOT/cache/huggingface:$USER_HOME/.cache/huggingface" \
    -v "$HDD_ROOT/cache/pip:$USER_HOME/.cache/pip" \
    "$IMAGE_NAME"
}

deploy
