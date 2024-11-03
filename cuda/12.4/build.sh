#!/usr/bin/env bash

docker build \
-t huyu/cuda:12.4-ubuntu22.04 \
--build-arg "BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04" \
-f cuda/12.4/Dockerfile .
