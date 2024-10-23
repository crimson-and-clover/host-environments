#!/usr/bin/env bash

NAMES=($(printenv | grep -v -i conda | grep -E 'NV|NCCL' | awk -F'=' '{ print $1 }'))

for n in ${NAMES[@]}; do
    echo "export $n=\"${!n}\""
done

echo "export PATH=\"\$PATH:/usr/local/cuda/bin\""