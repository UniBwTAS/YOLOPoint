"""Script for descriptor evaluation

Updated by You-Yi from https://github.com/eric-yyjau/image_denoising_matching
Date: 2020/08/05

"""

import numpy as np
import cv2
from os import path as osp
from glob import glob
import torch
from utils.utils import homography_scaling


def draw_matches_cv(data, matches, plot_points=True):
    # if plot_points:
    keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]

    def to3dim(img):
        img = np.rint(img[:, :, np.newaxis]*255.).astype(np.uint8)
        return img

    img1 = data['image1']
    img2 = data['image2']

    if data['image1'].ndim == 2:
        img1 = to3dim(img1)
        img2 = to3dim(img2)
        img1 = np.concatenate([img1, img1, img1], axis=2)
        img2 = np.concatenate([img2, img2, img2], axis=2)
    else:
        img1 = np.rint(img1*255.).astype(np.uint8)
        img2 = np.rint(img2*255.).astype(np.uint8)
    matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, singlePointColor=(0, 0, 255))
    return matches_img.transpose((2,0,1))

def homography_scaling_np(homography, H, W):
    trans = np.array([[2./W, 0., -1], [0., 2./H, -1], [0., 0., 1.]])
    homography = (np.linalg.inv(trans) @ homography @ trans)
    return homography

def compute_homography(data, keep_k_points=300, correctness_thresh=3, orb=False, return_mimg=True,
                       verbose=False, visualize=False):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    def print_(*input_string):
        if verbose:
            print(input_string)

    real_H = data['inv_homography']
    shape = data['image'].shape[:2]

    # Keeps only the points shared between the two views
    keypoints = data['prob'][:keep_k_points,[1, 0]]
    warped_keypoints = data['warped_prob'][:keep_k_points,[1, 0]]

    desc = data['desc'][:keep_k_points]
    warped_desc = data['warped_desc'][:keep_k_points]

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    try:
        cv2_matches = bf.match(desc, warped_desc)
        cv2_matches = sorted(cv2_matches, key=lambda x: x.distance)
        matches_idx = np.array([m.queryIdx for m in cv2_matches])
        m_keypoints = keypoints[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in cv2_matches])
        m_dist = np.array([m.distance for m in cv2_matches])
        m_warped_keypoints = warped_keypoints[matches_idx, :]
        matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
        m_dist_norm = m_dist / (m_dist.max())
    except Exception as e:
        print(e)
        cv2_matches, matches = [], []
        m_dist_norm = None

    if return_mimg:
        # Draw first n matches.
        draw_input = {
            'image1': data['image'],
            'image2': data['warped_image'],
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints
        }
        matches_img = draw_matches_cv(draw_input, cv2_matches)
    else:
        matches_img = None

    if visualize:
        cv2.imshow("matches", matches_img.transpose(1, 2, 0))

    # Estimate the homography between the matches using RANSAC
    if len(matches) >= 4:
        H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]], cv2.RANSAC)
        inliers = inliers.flatten()
    else:
        H = inliers = None

    # Compute correctness
    if H is None:
        correctness = 0
        H = np.identity(3)
        mean_dist = None
        print_("no valid estimation")
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])
        print_("corner: ", corners)
        real_H = homography_scaling_np(real_H, *shape)
        real_warped_corners = real_H@np.transpose(corners)
        real_warped_corners = np.transpose(real_warped_corners)
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        print_("real_warped_corners: ", real_warped_corners)

        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        print_("warped_corners: ", warped_corners)

        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        correctness = mean_dist <= correctness_thresh

    if visualize:
        cv2.waitKey(0)

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,
            'cv2_matches': cv2_matches,
            'mscores': m_dist_norm, # normalized distance
            'inliers': inliers,
            'homography': H,
            'mean_dist': mean_dist,
            'matches_img': matches_img
            }

def sample_desc_from_points(coarse_desc, pts, device, cell_size=8):
    """
    :param coarse_desc: torch descriptor tensor [D, H, W]
    :param pts:
    :param cell_size: 8
    :return: sparse desc
    """
    if len(coarse_desc.shape) != 4:
        coarse_desc = coarse_desc.view(*(1,) * (4 - coarse_desc.ndim), *coarse_desc.shape)

    H, W = coarse_desc.shape[2]*cell_size, coarse_desc.shape[3]*cell_size
    D = coarse_desc.shape[1]

    if len(pts.shape) == 2:
        if pts.shape[1] == 0:
            desc = np.empty((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(device)  # is to device really necessary? it's probably faster doing it on cpu if only one tensor
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)   # output of grid_sample and shape of pts?
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    else:
        desc = np.empty((D, 0))
        print(f"_________________!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\npts.shape = {pts.shape},\ntype = {type(pts)},\n"
              f"pts: {pts}")

    return desc


if __name__=='__main__':
    from utils.loader import dataLoader
    from utils.utils import dict_update, squeezeToNumpy, load_model, flattenDetection, toNumpy, \
        getPtsFromHeatmap, load_model
    import yaml
    import os
    from time import time
    from tqdm import tqdm

    os.chdir('/home/anba/catkin_ws/src/tas_dev/dev/anba/superpoint')
    with open('configs/coco_rgb.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = 'YOLOPoint_M'
    device = 'cuda'
    weights_path = 'logs/YOLOPoint_M_color/checkpoints/YOLOPoint_M_color_100_1729862_last_checkpoint.pth.tar'

    input_channels = 3

    model = load_model(model).to(device)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    dataset = config['data']['dataset']
    bs = {'model': {
        'train_batch_size': 1,
        'val_batch_size': 1}}
    config = dict_update(config, bs)
    print(config)

    data = dataLoader(config, dataset=dataset, DEBUG=False)
    train_loader, val_loader = data['train_loader'], data['val_loader']
    nms_dist = config['model']['nms']
    conf_thresh = config['model']['detection_threshold']
    top_k = config['model']['top_k']

    correctness_list = []

    with torch.no_grad():  # not sure if we need this
        t1 = time()
        for idx, sample in tqdm(enumerate(train_loader)):

            img = sample['image']
            img_warp = sample['warped_image']

            outs = model(img.to(device))
            semi, desc = outs['semi'], outs['desc']

            semi_flat_tensor = flattenDetection(semi).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, (0, 1))
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)

            outs_warp = model(img_warp.to(device))
            semi_warp, desc_warp = outs_warp['semi'], outs_warp['desc']

            semi_flat_tensor_warp = flattenDetection(semi_warp).detach()
            semi_flat_warp = toNumpy(semi_flat_tensor_warp)
            semi_thd_warp = np.squeeze(semi_flat_warp, (0, 1))
            pts_nms_warp = getPtsFromHeatmap(semi_thd_warp, conf_thresh, nms_dist)

            desc = sample_desc_from_points(desc, pts_nms, 'cuda', cell_size=8)
            desc_warp = sample_desc_from_points(desc_warp, pts_nms_warp, 'cuda', cell_size=8)

            if input_channels == 3:
                img = img.squeeze().transpose(0, 2).transpose(0, 1)
                img_warp = img_warp.squeeze().transpose(0, 2).transpose(0, 1)

            data = {
                'image': squeezeToNumpy(img),
                'warped_image': squeezeToNumpy(img_warp),
                'homography': squeezeToNumpy(sample['homographies']),
                'inv_homography': squeezeToNumpy(sample['inv_homographies']),
                'prob': pts_nms.transpose(),
                'warped_prob': pts_nms_warp.transpose(),
                'desc': desc.transpose(),
                'warped_desc': desc_warp.transpose()
            }

            hom_data = compute_homography(data, return_mimg=True, visualize=False)
            correctness_list.append(hom_data['correctness'])

            if idx >= 250:
                break

    print(np.mean(correctness_list))