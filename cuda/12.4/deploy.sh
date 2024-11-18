#!/usr/bin/env bash

docker run -d --name huyu-cuda12.4 \
    --restart unless-stopped \
    --gpus all \
    -p 0.0.0.0:22000:22/tcp \
    -v /home/hdd/huyu/shared:/root/hdd \
    -v /home/huyu/shared/:/root/ssd \
    --privileged \
    huyu/cuda:12.4-ubuntu22.04
