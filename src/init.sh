#!/usr/bin/env bash

service ssh start

bash /root/src/create_cuda_env.sh > /root/.cuda_env

echo "source ~/.cuda_env" >> ~/.bashrc

exec pause
