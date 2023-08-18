# YOLOPoint
Joint Keypoint and Object Detection

This is a complimentary repository to our paper [YOLOPoint](arxiv here).
The code is built on top of [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint) and [YOLOv5](https://github.com/ultralytics/yolov5).

![example_output](figures/0011.gif "Removing keypoints on dynamic objects")

## Installation
### Requirements
- python >= 3.8
- pytorch >= 1.10
- accelerate >= 1.14 (needed only for training)
- rospy (only for deployment with ROS)

```
$ pip install -r requirements.txt
```
Huggingface accelerate is a wrapper used mainly for multi-gpu and half-precision training.
Prior to training, you must configure accelerate with:
```
$ accelerate config
```

## Training
1. Adjust your config files as needed before launching the training script.
2. Use provided weights to generate pseudo-ground truth keypoint labels:
```
$ python export_homography.py
```
3. The following command will train YOLOPointS and save weights to logs/my_experiment/checkpoints.
```
$ accelerate launch train.py --config configs/coco.yaml --exper_name my_experiment --model YOLOPoint --version s
```
4. Broadcast Tensorboard logs.
```
$ sh logs/my_experiment/run_th.sh
```

## Inference
You will first want to configure your config.yaml file.
### Example if you're not using ROS:
```
$ cd src
$ python demo.py --config configs/inference.yaml
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

## Evaluation HPatches
```
$ python evaluation_hpatches.py
```

## TODOs
- [x] Add argument parser to demo.py (config + config-->track_points)
- [ ] Add setup.py
- [ ] Check that export_homography.py works out of the box
- [ ] Write full evaluation pipeline for HPatches