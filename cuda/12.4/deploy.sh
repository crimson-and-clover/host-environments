#!/usr/bin/env bash

docker run -d --name huyu-cuda12.4 \
--restart unless-stopped \
--gpus all \
-p 127.0.0.1:22000:22/tcp \
-v /home/hdd/huyu/shared:/root/hdd \
-v /home/huyu/shared/:/root/ssd \
huyu/cuda:12.4-ubuntu22.04
