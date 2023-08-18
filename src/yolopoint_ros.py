#!/usr/bin/env python

import argparse
import numpy as np
import cv2
import yaml
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from object_instance_msgs.msg import ObjectInstance2D, ObjectInstance2DArray
from keypoint_msg.msg import KeypointArray
from glob import glob
import os
from demo import YoloPointFrontend
import rospkg
import time


class yolocfg:

    def __init__(self):
        r = rospkg.RosPack()

        with open(r.get_path("yolopoint") + '/' + rospy.get_param('~config', 'configs/campus_inference.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)

        self.weights_path = r.get_path("yolopoint") + '/' + rospy.get_param('~weights_path', 'weights/CampusKitti/checkpoints/CampusKitti_46_2291_ckpt.pth.tar')
        self.filter_pts = rospy.get_param('~filter_pts', True)
        self.visualize = rospy.get_param('~visualize', False)
        self.camera_name = rospy.get_param('~sensor_name', 'surround/front')

        self.publish = True
        self.display_scale = 0.25
        self.source_type = 'ros'

        self.source = '/sensor/camera/' + self.camera_name + '/image_rect_color'

        self.template_paths =  self.config.get('templates')
        if self.template_paths:
            for t in self.template_paths:
                self.template_paths[t] = r.get_path("yolopoint") + '/' + self.template_paths[t]




class YoloPointFrontendROS(YoloPointFrontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, config, yolo_cfg):
        super().__init__(config, yolo_cfg, ros=True)

        # Launch nodes
        self.publish = yolo_cfg.publish
        self.visualize = yolo_cfg.visualize

        self.objects_pub = rospy.Publisher('objects', ObjectInstance2DArray, queue_size=10)
        self.keypoints_pub = rospy.Publisher('keypoints', KeypointArray, queue_size=10)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(yolo_cfg.source, Image, self.callback)


        # Load templates
        template_paths = yolo_cfg.template_paths

        if template_paths:
            print("Loading templates...")
            for rostpc, tp in template_paths.items():
                # Check if tp exists
                if os.path.exists(tp):
                    template = cv2.imread(tp, 0)
                    # crop and resize template
                    template, _, _, _ = self.preprocess(template, interpolation=cv2.INTER_NEAREST)
                    template = cv2.erode(template, np.ones((7, 7), np.uint8), iterations=1)
                    self.templates.update({rostpc: template})
                else:
                    print(f"Template {tp} does not exist")
            print("templates loaded")


    def callback(self, data):

        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        assert img is not None, 'No image is being streamed'


        rostpc = data.header.frame_id
        pts, desc, obj_preds = self.process_img(img, rostpc)

        # Publish messages
        self._publish_messages(pts, desc, obj_preds, data.header)
        if self.visualize:
            self.visualize_objects_and_tracks(img, pts, desc, obj_preds)

    def _publish_messages(self, pts, desc, obj_preds, header=''):
        keypoint_msg, array_msg = self.to_ros_msg(pts, desc, obj_preds, header)
        self.keypoints_pub.publish(keypoint_msg)
        self.objects_pub.publish(array_msg)

    def _publish_image(self, img):
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def to_ros_msg(self, pts, desc, obj_preds, header):
        keypoint_msg = KeypointArray()
        keypoint_msg.header = header
        keypoint_msg.x = pts[1, :].astype(np.uint16)
        keypoint_msg.y = pts[0, :].astype(np.uint16)
        keypoint_msg.score = pts[2, :].astype(np.float32)

        keypoint_msg.desc_len = np.array(desc.shape[0], dtype=np.uint8)
        keypoint_msg.desc_flat = desc.flatten().astype(float)

        # Process predictions
        array_msg = []
        for det in obj_preds:  # per image

            # Instantiate empty ObjectInstance2DArray object
            # must be in this loop so multiple detections in one image possible
            # header must also be in this loop so that the image continues to move when there are no objects present
            array_msg = ObjectInstance2DArray()
            array_msg.header = header
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    msg = ObjectInstance2D()
                    msg.class_name = self.names[c]
                    msg.class_index = c
                    msg.class_count = len(self.names)
                    msg.class_probabilities = [float(conf)]
                    msg.is_instance = True
                    msg.bounding_box_min_x = int(xyxy[0])
                    msg.bounding_box_min_y = int(xyxy[1])
                    msg.bounding_box_max_x = int(xyxy[2])
                    msg.bounding_box_max_y = int(xyxy[3])

                    array_msg.instances += [msg]
        return keypoint_msg, array_msg


if __name__ == '__main__':



    print("Initializing ROS node.")
    rospy.init_node('yolopoint_node')
    conf = yolocfg()

    print("Initializing YOLOPoint.")



    fe = YoloPointFrontendROS(config=conf.config, yolo_cfg=conf)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

# rosrun yolopoint demo_ROS.py configs/kitti_inference.yaml ros '/sensor/camera/surround/front/image_rect_color' --visualize --crop_resize_front