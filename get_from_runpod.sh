#!/bin/bash

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_PROJECT_NAME=$3

mkdir -p ./full_fine_tuning/model
scp -P ${RUNPOD_PORT} -r root@${RUNPOD_HOST}:~/app/${RUNPOD_PROJECT_NAME}/full_fine_tuning/model/* ./full_fine_tuning/model/
