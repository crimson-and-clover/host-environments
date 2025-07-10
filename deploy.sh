#!/usr/bin/env bash

USER_NAME="developer"

# modify here
CONTAINER_NAME="huyu-cuda11.8"
HOSTNAME="a6000x8_cu11"
IMAGE_NAME="huyu/cuda:11.8-ubuntu22.04"

SSD1_ROOT=/home/huyu/ssd1/container/
HDD1_ROOT=/home/huyu/hdd1/container/
HDD2_ROOT=/home/huyu/hdd2/container/

SSH_PORT=22000

USER_ID=1017
GROUP_ID=1017

function deploy() {
    # create data directory
    mkdir -p "$SSD1_ROOT/data"

    mkdir -p "$HDD1_ROOT/data"
    mkdir -p "$HDD1_ROOT/home"
    mkdir -p "$HDD2_ROOT/data"
    mkdir -p "$HDD2_ROOT/home"
    
    USER_HOME="/home/$USER_NAME"
    
    docker run -d --name="$CONTAINER_NAME" \
    --restart=unless-stopped \
    --gpus=all \
    --privileged \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --hostname="$HOSTNAME" \
    -p "0.0.0.0:$SSH_PORT:22/tcp" \
    -v "$HDD1_ROOT/home:$USER_HOME" \
    -v "$SSD1_ROOT/data:$USER_HOME/ssd1" \
    -v "$HDD1_ROOT/data:$USER_HOME/hdd1" \
    -v "$HDD2_ROOT/data:$USER_HOME/hdd2" \
    "$IMAGE_NAME"
    
    # TODO add group video render
    # according /dev/dri
    docker exec -it "$CONTAINER_NAME" bash -v /src/init_user.sh "$USER_ID" "$GROUP_ID"
}

deploy
