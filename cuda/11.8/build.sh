#!/usr/bin/env bash

docker build \
    --progress=plain \
    -t huyu/cuda:11.8-ubuntu22.04 \
    --build-arg "BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04" \
    -f src/Dockerfile .
