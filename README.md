# YOLOPoint
Joint Keypoint and Object Detection

This is a complimentary repository to our paper [YOLOPoint](https://arxiv.org/abs/2402.03989).
The code is built on top of [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint) and [YOLOv5](https://github.com/ultralytics/yolov5).

![example_output](figures/example.gif)
![filtered_trajectory](figures/filter_points.gif "Removing keypoints on dynamic objects leads to a better trajectory estimation.")

## Installation
### Requirements
- python >= 3.8
- pytorch >= 1.10
- accelerate >= 1.14 (needed only for training)
- rospy (optional, only for deployment with ROS)

```
$ pip install -r requirements.txt
```
Optional step for deployment with ROS:
```
$ pip install -r requirements_ros.txt
```
Huggingface accelerate is a wrapper used mainly for multi-gpu and half-precision training.
You can adjust the settings prior to training with (recommended for faster training) or just skip it:
```
$ accelerate config
```

## Data Organization
```
YOLOPoint/
├── datasets/
│   ├── coco/
│   │   ├── images/
│   │   │    ├── train/
│   │   │    └── val/
│   │   └── labels/
│   │        ├── train/
│   │        └── val/
```

## Pretrained Weights
Download COCO pretrained and KITTI fine-tuned weights:

| COCO pretrained      | [n](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointN.pth.tar?download=true)     | [s](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointS.pth.tar?download=true)          | [m](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointM.pth.tar?download=true) | [l](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointL.pth.tar?download=true) |
|----------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---|---|
| KITTI                | [n](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointN_kitti.pth.tar?download=true) | [s](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointS_kitti.pth.tar?download=true)    | [m](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointM_kitti.pth.tar?download=true) |   |
| COCO (experimental)  |                                                                                                  | [s](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointSv52.pth.tar?download=true)      |   |   |
| KITTI (experimental) |                                                                                                  | [s](https://huggingface.co/antopost/YOLOPoint/resolve/main/YOLOPointSv52_kitti.pth.tar?download=true) |   |   |
**_[New]_** 
Experimental weights follow a lighter YOLOv8-like architecture, were trained with InfoNCE loss and seem to have improved accuracy for keypoint matching.
However, this has not yet been thoroughly evaluated and is not part of the paper.

## Keypoint Labels
Generate your own pseudo ground truth keypoint labels with
```
$ python src/export_homography.py     TODO: make sure this works
```
OR use the shell script to download and place them in the appropriate place:
```
$ sh download_coco_points.sh
```
## Training
1. Adjust your config file as needed before launching the training script.
2. The following command will train YOLOPointS and save weights to logs/my_experiment/checkpoints.
```
$ accelerate launch src/train.py --config configs/coco.yaml --exper_name my_experiment --model YOLOPoint --version s
```
3. Broadcast Tensorboard logs.
```
$ sh logs/my_experiment/run_th.sh
```

## Inference
### Example if you're not using ROS:
```
$ python src/demo.py --config configs/inference.yaml --weights weights/YOLOPointS.pth.tar --source /path/to/image/folder/or/mp4
```

### Example if you are using ROS:
First build the package and start a roscore:
```
$ catkin build yolopoint
$ roscore
```
You can either choose to stream images from a directory or subscribe to a topic.
To visualize object bounding boxes and tracked points, set the --visualize flag.
```
$ rosrun yolopoint demo_ROS.py src/configs/kitti_inference.yaml directory '/path/to/image/folder' --visualize
```
Alternatively, you can publish bounding boxes and points and visualize them in another node.
Before publishing, the keypoint descriptor vectors get flattened into a single vector and then unflattened in a listener node.
```
$ rosrun yolopoint demo_ROS.py src/configs/kitti_inference.yaml ros '/image/message/name' --publish
$ rosrun yolopoint demo_ROS_listener.py '/image/message/name'
```

## Citation
```
@InProceedings{backhaus2023acivs-yolopoint,
  author    = {Anton Backhaus AND Thorsten Luettel AND Hans-Joachim Wuensche},
  title     = {{YOLOPoint: Joint Keypoint and Object Detection}},
  year      = {2023},
  month     = aug,
  pages     = {112--123},
  Note      = {ISBN: 978-3-031-45382-3},
  Publisher = {Springer},
  Series    = {Lecture notes in computer science},
  Volume    = {14124}
}
```
