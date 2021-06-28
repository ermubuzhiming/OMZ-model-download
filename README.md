# mobilenet-yolo-syg model

[![license1](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE)](LICENSE)
[![license2](https://raw.githubusercontent.com/david8862/keras-YOLOv3-model-set/master/LICENSE)](LICENSE)

## Introduction
A YOLOv4-MobileNet object detection pipeline inherited from [keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set) and [keras-yolo3-Mobilenet](https://github.com/Adamdad/keras-YOLOv3-mobilenet). 
Implement with keras, including model training/tuning, model evaluation and on device deployment. The model supports dataset collected by our team and 
including 9 classes 'motor,bike,rider,truck,bus,person,car,traffic_sign,traffic_light'.

## Setup
### Prerequisites
* Python 3.X.X
* Tensorflow 1.X
* Keras 2.X

## Training your own model
### Data preparation
```
wget https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/VOCdevkit0829.zip
unzip -zvf VOCdevkit0829.zip -d 
cd <model_download_dir>
python voc_annotation.py  
```
### Original weights preparation
```
wget https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/last1.h5 ./model_data
```
### Start to train

```
python train.py
```
or you can modify [train.py](https://github.com/ermubuzhiming/OMZ-model-download/blob/main/train.py) to set 
`annotation_path`,`classes_path`,`anchors_path`,`weights_path`,`log_dir`, `Init_epoch`,`Freeze_epoch`,`batch_size`,`learning_rate_base`,
`Freeze_epoch`,`Epoch` and other train parameters, then train it.

## Inference
If you want to test image,modify `FLAG` as `True` in the file [yolo_image.py](https://github.com/ermubuzhiming/OMZ-model-download/blob/main/yolo_image.py),
make dictionary named img and put test images, then
```
python yolo_image.py
```
If you want to test video,make dictionary named video and put test videos, modify `video_path` in the file 
[yolo_image.py](https://github.com/ermubuzhiming/OMZ-model-download/blob/main/yolo_image.py), then
```
python yolo_image.py
```
## Evalution
### Preparation
* modify `all_path` and `save_path` in the file [get_gt_txt1.py](https://github.com/ermubuzhiming/OMZ-model-download/get_gt_txt1.py) 
to get the ground-truth in the test dataset.
* modify `all_path` and `save_path` in the file [get_dr_txt2.py](https://github.com/ermubuzhiming/OMZ-model-download/get_dr_txt2.py)
to get the detection-result and images-optional in the test dataset.
* modify `MINOVERLAP` in the file [get_map3.py](https://github.com/ermubuzhiming/OMZ-model-download/get_map3.py)
to calculate mAP.
### Start to evaluate
```
python get_gt_txt1.py
python get_dr_txt2.py
python get_map3.py
```
