#!/bin/bash

set -e

echo "Downloading coco point labels and saving them to .datasets/coco/coco_points"
FILE_NAME="coco_points.zip"

cd ./datasets/coco/
wget -O ${FILE_NAME} "https://huggingface.co/antopost/YOLOPoint/resolve/main/coco_points.zip?download=true"
echo "Unzipping ${FILE_NAME}"
unzip "${FILE_NAME}"
cd ../../