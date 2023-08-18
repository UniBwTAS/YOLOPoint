"""Script for evaluation
This is the evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

import matplotlib
matplotlib.use('Agg') # solve error of tk

import numpy as np
# from evaluations.descriptor_evaluation import compute_homography
# from evaluations.detector_evaluation import compute_repeatability
import cv2
import matplotlib.pyplot as plt

import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs

# from evaluations.descriptor_evaluation import compute_homography

def warp_keypoints(keypoints, H):
    """
    :param keypoints:
    points:
        numpy (N, (x,y))
    :param H:
    :return:
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def draw_matches_cv_new(data, matches):
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
        img1 = img1.transpose((1, 2, 0))
        img2 = img2.transpose((1, 2, 0))
    matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None)
    return matches_img.transpose((2,0,1))

def compute_repeatability(data, keep_k_points=300,
                          distance_thresh=3, verbose=False):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        """
        points:
            numpy (N, (x,y))
        shape:
            (y, x)
        """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) & \
               (points[:, 1] >= 0) & (points[:, 1] < shape[0])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        """
        input:
            points: numpy (N, (x,y))
            shape: (y, x)
        return:
            points: numpy (N, (x,y))
        """
        # warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points = warp_keypoints(points[:, [0, 1]], H)
        # warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[1]) & \
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[0])
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points
        if points.shape[1] > 2:
            sorted_prob = points[points[:, 2].argsort(), :2]
            start = min(k, points.shape[0])
            sorted_prob = sorted_prob[-start:, :]
        return sorted_prob

    # paths = get_paths(exper_name)
    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []
    # for path in paths:
    # data = np.load(path)
    shape = data['image'].shape[1:] if len(data['image'].shape) > 2 else data['image'].shape
    H = data['homography']

    # Filter out predictions
    # keypoints = np.where(data['prob'] > 0)
    # prob = data['prob'][keypoints[0], keypoints[1]]
    # keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    # warped_keypoints = np.where(data['warped_prob'] > 0)
    # warped_prob = data['warped_prob'][warped_keypoints[0], warped_keypoints[1]]
    # warped_keypoints = np.stack([warped_keypoints[0],
    #                              warped_keypoints[1],
    #                              warped_prob], axis=-1)
    # keypoints = data['prob'][:, :2]
    keypoints = data['prob']
    # warped_keypoints = data['warped_prob'][:, :2]
    warped_keypoints = data['warped_prob']

    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),
                                           shape)

    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    # true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints[:, :2] = warp_keypoints(keypoints[:, :2], H)  # make sure the input fits the (x,y)
    # true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
    #                                   true_warped_keypoints[:, 0],
    #                                   prob], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    print('true_warped_keypoints: ', true_warped_keypoints[:2, :])
    N2 = warped_keypoints.shape[0]
    print('warped_keypoints: ', warped_keypoints[:2, :])
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                          ord=None, axis=2)
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        # print("count1: ", count1)
        local_err1 = min1[min1 <= distance_thresh]
        # print("local_err1: ", local_err1)
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        # repeatability.append((count1 + count2) / (N1 + N2))
        repeatability = (count1 + count2) / (N1 + N2)
    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += (local_err1.sum()) / (count1 + count2)
        if local_err2 is not None:
            localization_err += (local_err2.sum()) / (count1 + count2)
    else:
        repeatability = 0
    if verbose:
        print("Average number of points in the first image: " + str(np.mean(N1s)))
        print("Average number of points in the second image: " + str(np.mean(N2s)))
    # return np.mean(repeatability)
    return repeatability, localization_err


def keep_shared_points(keypoint_map, H, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    """
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = keep_true_keypoints(keypoints, H, keypoint_map.shape)
    keypoints = select_k_best(keypoints, keep_k_points)

    return keypoints.astype(int)

def compute_homography(data, keep_k_points=600, correctness_thresh=3, orb=False, shape=(240, 320)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    # shape = data['prob'].shape
    print("shape: ", shape)
    real_H = data['homography']

    # Keeps only the points shared between the two views

    keypoints = data['prob'][:keep_k_points,[1, 0]]
    warped_keypoints = data['warped_prob'][:keep_k_points,[1, 0]]

    desc = data['desc'][:keep_k_points]
    warped_desc = data['warped_desc'][:keep_k_points]

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    # def get_matches():
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    print("desc: ", desc.shape)
    print("w desc: ", warped_desc.shape)
    cv2_matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in cv2_matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in cv2_matches])
    m_dist = np.array([m.distance for m in cv2_matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]
    matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
    print(f"matches: {matches.shape}")
    # get_matches()
    # from export_classical import get_sift_match
    # data = get_sift_match(sift_kps_ii=keypoints, sift_des_ii=desc,
    # sift_kps_jj=warped_keypoints, sift_des_jj=warped_desc, if_BF_matcher=True)
    # matches_pts = data['match_quality_good']
    # cv_matches = data['cv_matches']
    # print(f"matches: {matches_pts.shape}")

    show_mimg = True
    if show_mimg:
        # Draw first n matches.
        draw_input = {
            'image1': data['image'],
            'image2': data['warped_image'],
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints
        }
        matches_img = draw_matches_cv_new(draw_input, cv2_matches)
        cv2.imshow("matches", matches_img.transpose(1, 2, 0))
        cv2.waitKey(0)
    else:
        matches_img = None


    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                    m_warped_keypoints[:, [1, 0]],
                                    cv2.RANSAC)

    # H, inliers = cv2.findHomography(matches_pts[:, [1, 0]],
    #                                 matches_pts[:, [3, 2]],
    #                                 cv2.RANSAC)

    inliers = inliers.flatten()
    # print(f"cv_matches: {np.array(cv_matches).shape}, inliers: {inliers.shape}")

    # Compute correctness
    if H is None:
        correctness = 0
        H = np.identity(3)
        print("no valid estimation")
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])
        print("corner: ", corners)
        # corners = np.array([[0, 0, 1],
        #             [0, shape[1] - 1, 1],
        #             [shape[0] - 1, 0, 1],
        #             [shape[0] - 1, shape[1] - 1, 1]])
        real_warped_corners = np.dot(corners, np.transpose(real_H))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        print("real_warped_corners: ", real_warped_corners)

        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        print("warped_corners: ", warped_corners)

        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        # correctness = float(mean_dist <= correctness_thresh)
        correctness = mean_dist <= correctness_thresh

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,  # cv2.match
            'cv2_matches': cv2_matches,
            'mscores': m_dist / (m_dist.max()),  # normalized distance
            'inliers': inliers,
            'homography': H,
            'mean_dist': mean_dist
            }

def draw_matches_cv(data, matches, plot_points=True):

    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    else:
        matches_pts = data['matches']
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")
        # keypoints1, keypoints2 = [], []

    inliers = data['inliers'].astype(bool)

    def to3dim(img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img
    img1 = to3dim(data['image1'])
    img2 = to3dim(data['image2'])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    img1 *= 255.
    img2 *= 255.
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def find_files_with_ext(directory, extension='.npz', if_int=True):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files

def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

def evaluate(args):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = args.path
    files = find_files_with_ext(path)
    correctness = []
    repeatability = []
    mscore = []
    mAP = []
    localization_err = []
    rep_thd = 3
    save_file = path + "/result.txt"
    inliers_method = 'cv'
    compute_map = True
    verbose = True
    top_K = 300
    print("top_K: ", top_K)

    reproduce = True
    if reproduce:
        logging.info("reproduce = True")
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)})")

    # create output dir
    if args.outputImg:
        path_warp = path+'/warping'
        os.makedirs(path_warp, exist_ok=True)
        path_match = path + '/matching'
        os.makedirs(path_match, exist_ok=True)
        path_rep = path + '/repeatibility' + str(rep_thd)
        os.makedirs(path_rep, exist_ok=True)

    print(f"file: {files[0]}")
    files.sort(key=lambda x: int(x[:-4]))
    from numpy.linalg import norm
    from utils.draw import draw_keypoints

    for f in tqdm(files):
        f_num = f[:-4]
        data = dict(np.load(path + '/' + f))
        print("load successfully. ", f)

        # unwarp
        # prob = data['prob']
        # warped_prob = data['prob']
        # desc = data['desc']
        # warped_desc = data['warped_desc']
        # homography = data['homography']
        real_H = data['homography']
        image = data['image']
        warped_image = data['warped_image']
        keypoints = data['prob'][:, [1, 0]]
        print("keypoints: ", keypoints[:3,:])
        warped_keypoints = data['warped_prob'][:, [1, 0]]
        print("warped_keypoints: ", warped_keypoints[:3,:])
        # print("Unwrap successfully.")

        if args.repeatibility:
            data.update({'inv_homography': np.linalg.inv(real_H)})
            data.update({'homography': real_H})
            rep, local_err = compute_repeatability(data, keep_k_points=top_K, distance_thresh=rep_thd, verbose=False)
            repeatability.append(rep)
            print("repeatability: %.2f"%(rep))
            if local_err > 0:
                localization_err.append(local_err)
                print('local_err: ', local_err)
            if args.outputImg:
                # img = to3dim(image)
                img = image
                pts = data['prob']
                img1 = draw_keypoints(img*255, pts.transpose())

                # img = to3dim(warped_image)
                img = warped_image
                pts = data['warped_prob']
                img2 = draw_keypoints(img*255, pts.transpose())

                plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['img1', 'img2'], dpi=200)
                plt.title("rep: " + str(repeatability[-1]))
                plt.tight_layout()

                plt.savefig(path_rep + '/' + f_num + '.png', dpi=300, bbox_inches='tight')
                pass

        if args.homography:
            # estimate result
            homography_thresh = [1,3,5,10,20,50]
            shape = image.shape[:2]
            from numpy.linalg import inv    # TODO: show fewer keypoints in matches image
            data.update({'inv_homography': inv(data['homography'])})
            data.update({'inv_homography': data['homography']})
            result = compute_homography(data, correctness_thresh=homography_thresh, shape=shape)#, visualize=True)
            correctness.append(result['correctness'])
            # est_H_mean_dist.append(result['mean_dist'])
            # compute matching score
            def warpLabels(pnts, homography, H, W):
                import torch
                """
                input:
                    pnts: numpy
                    homography: numpy
                output:
                    warped_pnts: numpy
                """
                from utils.utils import warp_points
                from utils.utils import filter_points
                pnts = torch.tensor(pnts).long()
                homography = torch.tensor(homography, dtype=torch.float32)
                warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                          homography)  # check the (x, y)
                warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
                return warped_pnts.numpy()

            from numpy.linalg import inv
            H, W = image.shape[:2]
            unwarped_pnts = warpLabels(warped_keypoints, inv(real_H), H, W)
            score = (result['inliers'].sum() * 2) / (keypoints.shape[0] + unwarped_pnts.shape[0])
            print("m. score: ", score)
            mscore.append(score)
            # compute map
            if compute_map:
                def getMatches(data):
                    from models.model_wrap import PointTracker

                    desc = data['desc']
                    warped_desc = data['warped_desc']

                    nn_thresh = 1.2
                    print("nn threshold: ", nn_thresh)
                    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                    # matches = tracker.nn_match_two_way(desc, warped_desc, nn_)
                    tracker.update(keypoints.T, desc.T)
                    tracker.update(warped_keypoints.T, warped_desc.T)
                    matches = tracker.get_matches().T
                    mscores = tracker.get_mscores().T

                    # mAP
                    # matches = data['matches']
                    print("matches: ", matches.shape)
                    print("mscores: ", mscores.shape)
                    print("mscore max: ", mscores.max(axis=0))
                    print("mscore min: ", mscores.min(axis=0))

                    return matches, mscores

                def getInliers(matches, H, epi=3, verbose=False):
                    """
                    input:
                        matches: numpy (n, 4(x1, y1, x2, y2))
                        H (ground truth homography): numpy (3, 3)
                    """
                    from evaluations.detector_evaluation import warp_keypoints
                    # warp points
                    warped_points = warp_keypoints(matches[:, :2], H) # make sure the input fits the (x,y)

                    # compute point distance
                    norm = np.linalg.norm(warped_points - matches[:, 2:4],
                                            ord=None, axis=1)
                    inliers = norm < epi
                    if verbose:
                        print("Total matches: ", inliers.shape[0], ", inliers: ", inliers.sum(),
                                          ", percentage: ", inliers.sum() / inliers.shape[0])

                    return inliers

                def getInliers_cv(matches, H=None, epi=3, verbose=False):
                    import cv2
                    # count inliers: use opencv homography estimation
                    # Estimate the homography between the matches using RANSAC
                    H, inliers = cv2.findHomography(matches[:, [0, 1]],
                                                    matches[:, [2, 3]],
                                                    cv2.RANSAC)
                    inliers = inliers.flatten()
                    print("Total matches: ", inliers.shape[0],
                          ", inliers: ", inliers.sum(),
                          ", percentage: ", inliers.sum() / inliers.shape[0])
                    return inliers


                def computeAP(m_test, m_score):
                    from sklearn.metrics import average_precision_score

                    average_precision = average_precision_score(m_test, m_score)
                    print('Average precision-recall score: {0:0.2f}'.format(
                        average_precision))
                    return average_precision

                def flipArr(arr):
                    return arr.max() - arr

                if args.sift:
                    assert result is not None
                    matches, mscores = result['matches'], result['mscores']
                else:
                    matches, mscores = getMatches(data)

                real_H = data['homography']
                if inliers_method == 'gt':
                    # use ground truth homography
                    print("use ground truth homography for inliers")
                    inliers = getInliers(matches, real_H, epi=3, verbose=verbose)
                else:
                    # use opencv estimation as inliers
                    print("use opencv estimation for inliers")
                    inliers = getInliers_cv(matches, real_H, epi=3, verbose=verbose)

                ## distance to confidence
                if args.sift:
                    m_flip = flipArr(mscores[:])  # for sift
                else:
                    m_flip = flipArr(mscores[:,2])

                if inliers.shape[0] > 0 and inliers.sum()>0:
#                     m_flip = flipArr(m_flip)
                    # compute ap
                    ap = computeAP(inliers, m_flip)
                else:
                    ap = 0

                mAP.append(ap)


            if args.outputImg:
                # draw warping
                output = result
                img1 = image
                img2 = warped_image

                img1 = to3dim(img1)
                img2 = to3dim(img2)
                H = output['homography']
                warped_image1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                img1 = np.concatenate([img1, img1, img1], axis=2)
                warped_image1 = np.stack([warped_image1, warped_image1, warped_image1], axis=2)
                img2 = np.concatenate([img2, img2, img2], axis=2)
                plot_imgs([img1, img2, warped_image1], titles=['img1', 'img2', 'warped_image1'], dpi=200)
                plt.tight_layout()
                plt.savefig(path_warp + '/' + f_num + '.png')

                ## plot filtered image
                img1, img2 = data['image'], data['warped_image']
                warped_image1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                plot_imgs([img1, img2, warped_image1], titles=['img1', 'img2', 'warped_image1'], dpi=200)
                plt.tight_layout()
                # plt.savefig(path_warp + '/' + f_num + '_fil.png')
                # plt.savefig(path_warp + '/' + f_num + '.png')

                plt.show()

                # draw matches
                result['image1'] = image
                result['image2'] = warped_image
                matches = np.array(result['cv2_matches'])
                ratio = 0.2
                ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))

                img = draw_matches_cv(result, matches[ran_idx], plot_points=True)
                # filename = "correspondence_visualization"
                plot_imgs([img], titles=["Two images feature correspondences"], dpi=200)
                plt.tight_layout()
                plt.savefig(path_match + '/' + f_num + 'cv.png', bbox_inches='tight')
                plt.close('all')
                plt.Imshow(img)

        if args.plotMatching:
            matches = result['matches'] # np [N x 4]
            if matches.shape[0] > 0:
                from utils.draw import draw_matches
                filename = path_match + '/' + f_num + 'm.png'
                ratio = 0.1
                inliers = result['inliers']

                matches_in = matches[inliers == True]
                matches_out = matches[inliers == False]

                def get_random_m(matches, ratio):
                    ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))
                    return matches[ran_idx], ran_idx
                image = data['image']
                warped_image = data['warped_image']
                ## outliers
                matches_temp, _ = get_random_m(matches_out, ratio)
                # print(f"matches_in: {matches_in.shape}, matches_temp: {matches_temp.shape}")
                draw_matches(image, warped_image, matches_temp, lw=0.5, color='r',
                            filename=None, show=False, if_fig=True)
                ## inliers
                matches_temp, _ = get_random_m(matches_in, ratio)
                draw_matches(image, warped_image, matches_temp, lw=1.0,
                        filename=filename, show=False, if_fig=False)

    if args.repeatibility:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print("repeatability: ", repeatability_ave)
        print("localization error over ", len(localization_err), " images : ", localization_err_m)
    if args.homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        # est_H_mean_dist = np.array(est_H_mean_dist)
        print("homography estimation threshold", homography_thresh)
        print("correctness_ave", correctness_ave)
        # print(f"mean est H dist: {est_H_mean_dist.mean()}")
        mscore_m = np.array(mscore).mean(axis=0)
        print("matching score", mscore_m)
        if compute_map:
            mAP_m = np.array(mAP).mean()
            print("mean AP", mAP_m)

        print("end")

    # # save to files
    # with open(save_file, "a") as myfile:
    #     myfile.write("path: " + path + '\n')
    #     myfile.write("output Images: " + str(args.outputImg) + '\n')
    #     if args.repeatibility:
    #         myfile.write("repeatability threshold: " + str(rep_thd) + '\n')
    #         myfile.write("repeatability: " + str(repeatability_ave) + '\n')
    #         myfile.write("localization error: " + str(localization_err_m) + '\n')
    #     if args.homography:
    #         myfile.write("Homography estimation: " + '\n')
    #         myfile.write("Homography threshold: " + str(homography_thresh) + '\n')
    #         myfile.write("Average correctness: " + str(correctness_ave) + '\n')
    #
    #         # myfile.write("mean est H dist: " + str(est_H_mean_dist.mean()) + '\n')
    #
    #         if compute_map:
    #             myfile.write("nn mean AP: " + str(mAP_m) + '\n')
    #         myfile.write("matching score: " + str(mscore_m) + '\n')
    #
    #     if verbose:
    #         myfile.write("====== details =====" + '\n')
    #         for i in range(len(files)):
    #
    #             myfile.write("file: " + files[i])
    #             if args.repeatibility:
    #                 myfile.write("; rep: " + str(repeatability[i]))
    #             if args.homography:
    #                 myfile.write("; correct: " + str(correctness[i]))
    #                 # matching
    #                 myfile.write("; mscore: " + str(mscore[i]))
    #                 if compute_map:
    #                     myfile.write(":, mean AP: " + str(mAP[i]))
    #             myfile.write('\n')
    #         myfile.write("======== end ========" + '\n')
    #
    # dict_of_lists = {
    #     'repeatability': repeatability,
    #     # 'localization_err': localization_err,
    #     # 'correctness': np.array(correctness),
    #     # 'homography_thresh': homography_thresh,
    #     # 'mscore': mscore,
    #     # 'mAP': np.array(mAP),
    #     # 'est_H_mean_dist': est_H_mean_dist
    # }
    #
    # filename = f'{save_file[:-4]}.npz'
    # logging.info(f"save file: {filename}")
    # np.savez(
    #     filename,
    #     **dict_of_lists,
    # )


if __name__ == '__main__':
    import argparse
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sift', action='store_true', help='use sift matches')
    parser.add_argument('-o', '--outputImg', action='store_true')
    parser.add_argument('-r', '--repeatibility', action='store_true')
    parser.add_argument('-homo', '--homography', action='store_true')
    parser.add_argument('-plm', '--plotMatching', action='store_true')
    args = parser.parse_args()
    evaluate(args)
