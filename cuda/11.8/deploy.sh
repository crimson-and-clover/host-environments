#!/usr/bin/env bash

docker run -d --name huyu-cuda11.8 \
--restart unless-stopped \
--gpus all \
-p 0.0.0.0:22000:22/tcp \
-v /home/hdd/huyu/shared:/root/hdd \
-v /home/huyu/shared/:/root/ssd \
huyu/cuda:11.8-ubuntu22.04
