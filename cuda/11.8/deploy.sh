#!/usr/bin/env bash

docker run -d --name huyu-cuda11.8 \
    --restart unless-stopped \
    --gpus all \
    --privileged \
    --shm-size=2g \
    --hostname=cuda_11_8 \
    -p "0.0.0.0:22000:22/tcp" \
    -v "/home/hdd/huyu/container/data:/home/developer/hdd" \
    -v "/home/ssd/huyu/container/data:/home/developer/ssd" \
    -v "/home/hdd/huyu/container/anaconda3/envs:/home/developer/anaconda3/envs" \
    -v "/home/hdd/huyu/container/anaconda3/pkgs:/home/developer/anaconda3/pkgs" \
    -v "/home/hdd/huyu/container/vscode-server:/home/developer/.vscode-server" \
    -v "/home/hdd/huyu/container/cache/huggingface:/home/developer/.cache/huggingface" \
    -v "/home/hdd/huyu/container/cache/pip:/home/developer/.cache/pip" \
    huyu/cuda:11.8-ubuntu22.04
