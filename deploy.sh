#!/usr/bin/env bash

CONTAINER_NAME="huyu-cuda11.8"
HOSTNAME="cuda_11_8"
IMAGE_NAME="huyu/cuda:11.8-ubuntu22.04"

function deploy() {
    docker run -d --name="$IMAGE_NAME" \
    --restart=unless-stopped \
    --gpus=all \
    --privileged \
    --shm-size=2g \
    --hostname="$HOSTNAME" \
    -p "0.0.0.0:22000:22/tcp" \
    -v "/home/hdd/huyu/container/data:/home/developer/hdd" \
    -v "/home/ssd/huyu/container/data:/home/developer/ssd" \
    -v "/home/hdd/huyu/container/anaconda3/envs:/home/developer/anaconda3/envs" \
    -v "/home/hdd/huyu/container/anaconda3/pkgs:/home/developer/anaconda3/pkgs" \
    -v "/home/hdd/huyu/container/vscode-server:/home/developer/.vscode-server" \
    -v "/home/hdd/huyu/container/cache/huggingface:/home/developer/.cache/huggingface" \
    -v "/home/hdd/huyu/container/cache/pip:/home/developer/.cache/pip" \
    "$IMAGE_NAME"
}

deploy
