from math import pi
import cv2
import numpy as np
from numpy.random import normal, uniform
from scipy.stats import truncnorm
import random as rndm
from random import random, randint
from utils.general_yolo import xywhn2xyxy, xyxy2xywhn, clip_boxes
from utils.augmentations_yolo import box_candidates
from utils.utils import warp_points, homography_scaling
import torch

def sample_homography(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=1.0, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0., crop_pts=None, seed=None):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
        r: enlarging factor of cropped image to non-cropped (<1)

    Returns:
        A `Tensor` of shape `[3, 3]`.
    """
    if seed is not None:
        np.random.seed(seed)
        rndm.seed(seed)
    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])

    # Random perspective and affine perturbations
    std_trunc = 2
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        # scales = np.concatenate( (np.ones((n_scales,1)), scales[:,np.newaxis]), axis=1)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            # valid = np.where((scaled >= 0.) * (scaled < 1.))
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 = pts1 * shape[np.newaxis,:] + shift
    pts2 = pts2 * shape[np.newaxis,:] + shift

    homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))

    # r = 0.75
    # if r:   # r < 1
    #     pts3 = pts1.copy()  # yx
    #     pts3 *= r
    #     pts4 = (pts2 - pts1) * r + pts3
    #     homography_big2 = cv2.getPerspectiveTransform(np.float32(pts3), np.float32(pts4))

    if crop_pts is not None:
        W = crop_pts["W_crop"]
        H = crop_pts["H_crop"]
        tl = np.array(crop_pts["tl"])
        bl = tl + np.array([H, 0])
        br = tl + np.array([H, W])
        tr = tl + np.array([0, W])

        im_corners = np.array([tl, bl, br, tr])
        im_corners_norm = (im_corners / np.array(crop_pts['HW_full'])) * 2 - 1
        H_32_1 = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(im_corners_norm))   # p^tilde --> p
        pts42 = warp_points(torch.from_numpy(pts2.copy()).float(), torch.from_numpy(H_32_1).float()).numpy()
        homography_big = cv2.getPerspectiveTransform(np.flip(np.float32(im_corners_norm), axis=1), np.flip(np.float32(pts42), axis=1))

        return homography, homography_big

    return homography



def flip(image, points, labels=(), horizontal=0., vertical=0.):
    """
    Flips input image and points given a probability
    :param image: tensor [H, W]
    :param points: [x, y, score]
    :param labels: [class, x, y, w, h] , 0 < x, y < 1
    :param horizontal: 0 <= probability <= 1
    :param vertical: 0 <= probability <= 1
    :return:
    """
    y, x = image.shape[:2]

    nl = len(labels)
    np = len(points)
    if horizontal and horizontal > random():
        if np:
            points[:, 0] = x - points[:, 0] # check if 0 is correct
        if nl:
            labels[:, [1, 3]] = x - labels[:, [1, 3]]
            labels[:, [1, 3]] = labels[:, [3, 1]]
        image = cv2.flip(image, 1)

    if vertical and vertical > random():
        if np:
            points[:, 1] = y - points[:, 1]
        if nl:
            labels[:, [2, 4]] = y - labels[:, [2, 4]]
            labels[:, [2, 4]] = labels[:, [4, 2]]
        image = cv2.flip(image, 0)

    return image, points, labels

class RandomFixedSizeCrop:
    """
    Crop an image to a fixed size.
    """
    def __init__(self, cropHW, center_crop=False): # TODO: add argument that crop is always in centre (for val)
        self.cropHW = cropHW
        self.center_crop = center_crop

    @staticmethod
    def filter_keypoints(points, shape):
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) & \
               (points[:, 1] >= 0) & (points[:, 1] < shape[0])
        return points[mask, :]

    def __call__(self, img, pts=None, bbox_labels=None):
        height, width = img.shape[:2]
        assert width >= self.cropHW[1] and height >= self.cropHW[0], \
            f"Given crop dimensions {self.cropHW} must be <= image dimensions {img.shape[:2]}."

        # Calculate the maximum x and y coordinates for the top-left corner of the crop
        max_x = width - self.cropHW[1]
        max_y = height - self.cropHW[0]

        if not self.center_crop:
            x = randint(0, max_x)
            y = randint(0, max_y)
        else:
            x = (width - self.cropHW[1]) // 2
            y = (height - self.cropHW[0]) // 2

        # if pts is not None:
        #     pts[:, 0] -= x
        #     pts[:, 1] -= y
        #     pts = self.filter_keypoints(pts, (self.cropHW[0], self.cropHW[1]))
        # if bbox_labels is not None:
        #     bbox_labels[:, 1:] = xywhn2xyxy(bbox_labels[:, 1:], self.cropHW[1], self.cropHW[0], -x, -y)
        #     # filter out small bboxes
        #     labels_old = bbox_labels.clone()
        #     clip_boxes(bbox_labels[:, 1:], (self.cropHW[0], self.cropHW[1]))
        #     i = box_candidates(box1=bbox_labels[:, 1:], box2=labels_old[:, 1:], area_thr=25)
        #     bbox_labels = bbox_labels[i]
        #     bbox_labels[:, 1:] = xyxy2xywhn(bbox_labels[:, 1:], h=self.cropHW[0], w=self.cropHW[1], clip=True)
        # img = img[y:y + self.cropHW[0], x:x + self.cropHW[1]]
        top_left = (y,x)    # top left corner pixel
        # return img, pts, bbox_labels, top_left
        return top_left

if __name__ == "__main__":
    crop_size = (288, 288)
    img = cv2.imread("/home/anba/catkin_ws/src/tas_dev/dev/anba/superpoint/figures/0.png")
    rfs_crop = RandomFixedSizeCrop(crop_size, center_crop=True)
    for _ in range(10):
        crop = rfs_crop(img)
        cv2.imshow("crop", crop)
        cv2.waitKey(0)