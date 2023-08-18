import os

import numpy as np
from utils.utils import warp_image, warp_points, homography_scaling
import torch
import cv2


def batch_precision_recall(batch_pred, batch_labels):
    """
    Computes precisions and recalls of a batch
    :param batch_pred: batch of heatmaps (0 <= label <= 1) [batch_size, *, H, W]
    :param batch_labels: batch of 2D labels (0s or 1s) [batch_size, *, H, W]
    :return: dict of precision and recall batch (concatenated scalars, torch tensor)
    """
    if len(batch_pred.size()) == 4:
        batch_pred = torch.squeeze(batch_pred, 1)
    if len(batch_labels.size()) == 4:
        batch_labels = torch.squeeze(batch_labels, 1)
    offset = 10 ** -6
    assert batch_pred.size() == batch_labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    batch_sum = torch.sum(batch_pred * batch_labels, (1, 2))
    precision = batch_sum / (torch.sum(batch_pred, (1, 2)) + offset)
    recall = batch_sum / (torch.sum(batch_labels, (1, 2)) + offset)
    return {"precision": precision, "recall": recall}

def warp_keypoints(keypoints, Hom, shape, scale=True):
    '''
    :param keypoints:
    points:
        numpy (N, (x,y))
    :param Hom:
    :return:
    '''
    num_points = keypoints.shape[0]
    if scale:
        Hom = homography_scaling(Hom, *shape[:2])
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(Hom))
    return warped_points[:, :2] / warped_points[:, 2:]

def compute_repeatability(data, keep_k_points=300, distance_thresh=3, verbose=False, scale=True):
    '''
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    '''

    def filter_keypoints(points, shape, margin=2):
        ''' Keep only the points whose coordinates are
        inside the dimensions of shape. '''
        '''
        points:
            numpy (N, (x,y))
        shape:
            (y, x)
        margin:
            valid border margin
        '''
        mask = (points[:, 0] >= margin) & (points[:, 0] < shape[1]-margin) &\
               (points[:, 1] >= margin) & (points[:, 1] < shape[0]-margin)

        return points[mask, :]

    def keep_true_keypoints(points, H, shape, margin=2, scale=True):
        ''' Keep only the points whose warped coordinates by H
        are still inside shape. '''
        '''
        input:
            points: numpy (N, (x,y))
            shape: (y, x)
        return:
            points: numpy (N, (x,y))
        '''
        warped_points = warp_keypoints(points[:, [0, 1]], H, shape[:2], scale)

        mask = (warped_points[:, 0] >= margin) & (warped_points[:, 0] < shape[1]-margin) &\
               (warped_points[:, 1] >= margin) & (warped_points[:, 1] < shape[0]-margin)

        return points[mask, :]

    def select_k_best(points, k):
        ''' Select the k most probable points (and strip their proba).
        points have shape (num_points, 3) where the last coordinate is the proba. '''
        sorted_prob = points
        if points.shape[1] > 2:
            sorted_prob = points[points[:, 2].argsort(), :2]
            start = min(k, points.shape[0])
            sorted_prob = sorted_prob[-start:, :]
        return sorted_prob

    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []

    shape = data['image'].shape
    if shape[0] == 3:
        # [ch,y,x]
        shape = (*shape[1:], 3)

    H = data['homography']
    H_inv = data['inv_homography']

    keypoints = data['prob'].copy()

    warped_keypoints = data['warped_prob'].copy()
    # filter detections outside image boundaries
    # TODO: include homography in data to avoid extra inverse calculation
    # plot_points(img_warp, warped_keypoints)
    warped_keypoints = keep_true_keypoints(warped_keypoints, H, shape, scale)   # doesn't do anything
    # plot_points(img_warp, warped_keypoints)
    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, :2], H_inv, shape)
    # plot_points(img_warp, true_warped_keypoints)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)
    # plot_points(img_warp, true_warped_keypoints)
    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)

    # shapes are broadcast to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None

    if N2 != 0: # found warped keypoints
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        local_err1 = min1[min1 <= distance_thresh]

    if N1 != 0: # true keypoints
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)

    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += (local_err1.sum())/ (count1 + count2)
        if local_err2 is not None:
            localization_err += (local_err2.sum())/ (count1 + count2)
    else:
        repeatability = 0
    if verbose:
        print('Average number of points in the first image: ' + str(np.mean(N1s)))
        print('Average number of points in the second image: ' + str(np.mean(N2s)))

    return repeatability, localization_err

def plot_points(img_test, pts, winname="test"):
    # used for debugging
    if torch.is_tensor(img_test):
        img_test = squeezeToNumpy(img_test)
    if img_test.shape[0] == 3:
        img_test = img_test.transpose((1,2,0))
    img_new = img_test.copy()
    pts = np.round(pts.transpose()[:2,:300].transpose()).astype(int)
    for p in pts:
        cv2.circle(img_new, p, 1, (255, 0, 0), -1)
    cv2.imshow(winname, img_new)
    cv2.waitKey(0)

if __name__ == '__main__':
    from utils.loader import dataLoader
    from utils.utils import dict_update, squeezeToNumpy, load_model, flattenDetection, toNumpy, \
        getPtsFromHeatmap, load_model
    import yaml
    from time import time
    from tqdm import tqdm

    def visualize(img, pts):

        img *= 255
        img = img.astype(np.uint8)
        if img.shape[0] == 3:
            img = img.transpose((1,2,0))
        img = np.ascontiguousarray(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for point in pts[:, :2]:
            point = point.astype(int)
            img = cv2.circle(img, point, 1, (0, 255, 0), -1)
        cv2.imshow("test", img)
        cv2.waitKey(0)

    os.chdir('/home/anba/catkin_ws/src/tas_dev/dev/anba/superpoint')
    with open('configs/coco_rgb.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = 'YOLOPointMFull'
    device = 'cuda'

    weights_path = config['pretrained']

    model = load_model(model_name=model, names=config['names']).to(device)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test = model.state_dict()
    model.eval()

    dataset = config['data']['dataset']
    bs = {'model': {
        'train_batch_size': 1,
        'val_batch_size': 1}}
    config = dict_update(config, bs)
    print(config)

    # data and params
    data = dataLoader(config, dataset=dataset, DEBUG=False)
    train_loader, val_loader = data['train_loader'], data['val_loader']
    nms_dist = config['model']['superpoint']['nms']
    conf_thresh = config['model']['superpoint']['detection_threshold']
    # top_k = config['model']['top_k']

    t1 = time()
    sample_size = 500
    repeatability_list = []
    loc_error_list = []
    with torch.inference_mode(): # not sure if we need this
        for idx, sample in tqdm(enumerate(val_loader)):

            img = sample['image']
            img_warp = sample['warped_image']

            outs = model(img.to(device))
            semi, desc = outs['semi'][0].unsqueeze(0), outs['desc'][0].unsqueeze(0)

            semi_flat_tensor = flattenDetection(semi).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, (0, 1))
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)

            outs_warp = model(img_warp.to(device))
            semi_warp, desc_warp = outs_warp['semi'][0].unsqueeze(0), outs_warp['desc'][0].unsqueeze(0)

            semi_flat_tensor_warp = flattenDetection(semi_warp).detach()
            semi_flat_warp = toNumpy(semi_flat_tensor_warp)
            semi_thd_warp = np.squeeze(semi_flat_warp, (0, 1))
            pts_nms_warp = getPtsFromHeatmap(semi_thd_warp, conf_thresh, nms_dist)

            data = {
                'image': squeezeToNumpy(img[0].unsqueeze(0)),
                'inv_homography': squeezeToNumpy(sample['inv_homographies'][0].unsqueeze(0)),
                'prob': pts_nms.transpose(),
                'warped_prob': pts_nms_warp.transpose(),
            }

            vis = False
            if vis:
                im_warp = squeezeToNumpy(img_warp)
                # heatmap2Dnorm = (semi_thd - np.min(semi_thd))
                # heatmap2Dnorm /= np.max(heatmap2Dnorm)
                #
                # cv2.imshow('heatmap', heatmap2Dnorm)
                # cv2.waitKey(0)

                for image, points in (data['image'], data['prob']), (im_warp, data['warped_prob']):
                    visualize(image, points)

            repeatability, loc_error = compute_repeatability(data, keep_k_points=300, distance_thresh=3, verbose=False)
            # print(f"repeatability: {repeatability}, loc_error: {loc_error}")
            repeatability_list.append(repeatability)
            loc_error_list.append(loc_error)

            if idx >= sample_size:
                break

    print("rep", np.mean(repeatability_list))
    print("loc err", np.mean(loc_error_list))
    print(time()-t1)
