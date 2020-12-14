# Tensorflow Faster RCNN

## Clone Project
```bash
mkdir TensorFlow
git clone https://github.com/wcl20/Tensorflow-Faster-RCNN.git
mv Tensorflow-Faster-RCNN/setup.sh .
```

## Download Dataset
```bash
wget http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip
unzip signDatabasePublicFramesOnly.zip -d Tensorflow-Faster-RCNN/lisa
rm signDatabasePublicFramesOnly.zip
```

## Download pretrained model
```bash
mkdir -p Tensorflow-Faster-RCNN/pre-trained-models
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
tar xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz -C Tensorflow-Faster-RCNN/pre-trained-models
rm faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

## Setup
Setup Docker Container
```bash
docker pull nvcr.io/nvidia/tensorflow:20.10-tf1-py3
docker run --gpus all -it --rm -p 6006:6006 -p 6064:6064 -p 8888:8888 -v <PATH TO TensorFlow>:/workspace nvcr.io/nvidia/tensorflow:20.10-tf1-py3
```
If the above command does not work, run the following and try again
```bash
apt-get install nvidia-container-runtime
systemctl restart docker
```
Setup project inside the docker container
```bash
chmod +x setup.sh
./setup.sh
```

## Build TFRecords
```bash 
cd Tensorflow-Faster-RCNN
python build.py
```
The records file will be stored in TensorFlow-Faster-RCNN/lisa/records

## Training
Inside TensorFlow-Faster-RCNN
```bash
cp ../models/research/object_detection/model_main.py .
python model_main.py \
--pipeline_config_path=models/faster_rcnn_resnet101/pipeline.config \
--model_dir=models/faster_rcnn_resnet101 \
--num_train_steps=50000 \
--sample_1_of_n_eval_examples
```



