#!/bin/bash

apt-get update
apt-get dist-upgrade -y
apt-get clean
apt-get install git protobuf-compiler ffmpeg libmpeg libsm6 libxext6 -y

pip install --upgrade pip
pip install imutils scikit-learn

# Install Tensorflow Models
if [ ! -d "/workspace/models" ]
then
    git clone https://github.com/tensorflow/models.git
    # Protobuf Compiler
    cd models/research
    protoc object_detection/protos/*.proto --python_out=.
    # Install COCO API
    cd /workspace
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    make
    cp -r pycocotools /workspace/models/research/
else
    echo "TensorFlow Models Repository already exist."
fi

cd /workspace/models/research
cp object_detection/packages/tf1/setup.py .
python -m pip install --use-feature=2020-resolver .

# Test Tensorflow Models is properly installed
python object_detection/builders/model_builder_tf1_test.py

cd /workspace
