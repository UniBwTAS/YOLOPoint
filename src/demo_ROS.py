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


class YoloPointFrontendROS(YoloPointFrontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, config, args):
        super().__init__(config, args, ros=True)

        # Launch nodes
        self.publish = args.publish
        self.visualize = args.visualize

        if self.publish:
            self.objects_pub = rospy.Publisher('objects', ObjectInstance2DArray, queue_size=10)
            self.keypoints_pub = rospy.Publisher('keypoints', KeypointArray, queue_size=10)

        if args.source_type == 'ros':
            self.bridge = CvBridge()
            self.image_sub = rospy.Subscriber(args.source, Image, self.callback)
        else:
            img_paths = sorted(glob(os.path.join(args.source, '*.png'))) + \
                        sorted(glob(os.path.join(args.source, '*.jpg')))
            assert len(img_paths) > 0, f"Directory {args.source} either does not " \
                                       f"exist or no images with extension png or jpg found"
            self.stream_imgs(img_paths)

        # Load templates
        template_paths = self.config.get('templates')

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

    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        assert img is not None, 'No image is being streamed'
        rostpc = data.header.frame_id
        pts, desc, obj_preds = self.process_img(img, rostpc)

        # Publish messages
        if self.publish:
            self._publish_messages(pts, desc, obj_preds, data.header)
        if self.visualize:
            self.visualize_objects_and_tracks(img, pts, desc, obj_preds)

    def _publish_messages(self, pts, desc, obj_preds, header=''):
        keypoint_msg, array_msg = self.to_ros_msg(pts, desc, obj_preds, header)
        self.keypoints_pub.publish(keypoint_msg)
        self.objects_pub.publish(array_msg)

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

    def stream_imgs(self, img_paths):
        # Usage: stream images from directory
        for p in img_paths:
            img = cv2.imread(p)
            pts, desc, obj_preds = self.process_img(img)
            if self.publish:
                self._publish_messages(pts, desc, obj_preds)
            if self.visualize:
                self.visualize_objects_and_tracks(img, pts, desc, obj_preds)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='YOLOPoint Demo.')
    parser.add_argument('config', type=str, default='configs/kitti_inference.yaml',
                        help='path/to/config (default: configs/kitti_inference.yaml)')
    parser.add_argument('source_type', type=str, default='',
                        help="Choose 'directory' or 'ros'")
    parser.add_argument('source', type=str, default='',
                        help='path/to/directory or ros/image/topic/name')
    parser.add_argument('--weights_path', type=str, default='logs/YOLOPointS_kitti_nomo/checkpoints'
                                                            '/YOLOPointS_kitti_nomo_110_11111_last_ckpt.pth.tar',
                        help='Path to pretrained weights file '
                             '(default: logs/YOLOPointS_kitti_nomo/checkpoints'
                             '/YOLOPointS_kitti_nomo_110_11111_last_ckpt.pth.tar).')
    parser.add_argument('--filter_pts', action='store_false',
                        help='Filter out dynamic points by default')
    parser.add_argument('--visualize', action='store_true',
                        help='Opencv visualization of points and tracks. Only for demo purposes!!')
    parser.add_argument('--publish', action='store_true',
                        help='Publish keypoints and bounding boxes')
    parser.add_argument('--display_scale', type=float, default=1.,
                        help='Factor to scale output visualization (default: 1, requires --visualize argument).')
    args = parser.parse_args()

    args.source_type = args.source_type.lower()
    assert args.source_type in {'ros', 'directory'}, "source_type must be either 'ros' or 'directory'"

    r = rospkg.RosPack()
    path_to_pkg = r.get_path("yolopoint")

    print("Getting config.")
    with open(os.path.join(path_to_pkg, args.config), 'r') as f:
        config = yaml.safe_load(f)

    print("Initializing ROS node.")
    rospy.init_node('yolopoint_node')

    print("Initializing YOLOPoint.")
    fe = YoloPointFrontendROS(config=config, args=args)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

# rosrun yolopoint demo_ROS.py configs/kitti_inference.yaml ros '/sensor/camera/surround/front/image_rect_color' --visualize --crop_resize_front