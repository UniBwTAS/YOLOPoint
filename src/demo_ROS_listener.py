#!/usr/bin/env python

import argparse
import numpy as np
import rospy
from keypoint_msg.msg import KeypointArray
from demo import PointTracker
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import message_filters


class KeypointListener:
    """ Example subscriber node that gets keypoints and images and displays keypoints and tracks. """

    def __init__(self, args):
        # Launch nodes
        self.keypoint_sub = message_filters.Subscriber('/keypoints', KeypointArray)
        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber(args.source, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.keypoint_sub, self.image_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.callback)

        # tracker
        self.nn_thresh = 0.7
        self.min_length = 2
        self.max_length = 4
        self.tracker = PointTracker(max_length=4, nn_thresh=0.7)

        self.img = None

        self.display_scale = args.display_scale

    def callback(self, kp_data, img_data):
        coords = np.array([kp_data.y, kp_data.x])
        desc_len = kp_data.desc_len
        desc_flat = np.array(kp_data.desc_flat)
        desc = desc_flat.reshape(desc_len, -1)

        try:
            self.img = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        assert self.img is not None, 'No image is being streamed'
        print(coords.shape)
        print(desc.shape)
        self._visualize_tracks(coords, desc)


    def _visualize_tracks(self, pts, desc):

        self.tracker.update(pts, desc)

        # Get tracks for points which were matched successfully across all frames.
        tracks = self.tracker.get_tracks(self.min_length)
        out = self.img.copy()

        tracks[:, 1] /= float(self.nn_thresh)  # Normalize track scores to [0,1].
        self.tracker.draw_tracks(out, tracks)
        # annotate bboxes and draw tracks
        if self.display_scale != 1.:
            new_shape = (np.array(out.shape[:2][::-1]) * self.display_scale).astype(int)
            out = cv2.resize(out, new_shape)
        cv2.imshow("Visualization", out)
        cv2.waitKey(1)

def main(args):
    rospy.init_node('keypoint_example_node', anonymous=True)
    kl = KeypointListener(args)
    rospy.spin()

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Keypoint example listener')
    parser.add_argument('source', type=str, default='', help='ros/image/message/name')
    parser.add_argument('--display_scale', type=float, default=1.,
                             help='Factor to scale output visualization (default: 1).')
    # don't use together with --crop_resize_front
    args = parser.parse_args()

    main(args)