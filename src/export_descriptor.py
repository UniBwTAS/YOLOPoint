"""
This script exports detection/ description using pretrained model.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

## basic
import argparse
from utils.general_yolo import LOGGER
import yaml
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
## torch
import torch
import torch.utils.data
from evaluations.descriptor_evaluation import sample_desc_from_points
## other functions
from utils.utils import data_size, load_model, squeezeToNumpy, getPtsFromSemi, warp_image_batch
from utils.loader import dataLoader
from models.model_wrap import PointTracker
import cv2


@torch.no_grad()
def export_descriptor(config, output_dir, model_info):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    # basic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # save_dir = os.path.join(output_dir, 'desc_export_hpatches')
    # os.makedirs(save_dir, exist_ok=True)

    ## parameters
    outputMatches = True

    # data loading
    dataset = config['data']['dataset'] # HPatches
    val_loader = dataLoader(config, dataset=dataset, action='val')
    data_size(val_loader, config, tag="val")

    # model loading
    model = load_model(**model_info).to(device)
    try:
        state_dict = torch.load(config['pretrained'])['model_state_dict']
    except KeyError:
        state_dict = torch.load(config['pretrained'])
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    ## tracker
    nn_thresh = config['model']['superpoint']['nn_thresh']
    det_thresh = config['model']['superpoint']['detection_threshold']
    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)

    count = 0
    nms_dist = config["model"]["superpoint"]["nms"]

    for i, sample in tqdm(enumerate(val_loader)):
        img_0, img_1 = sample["image"].to(device), sample["warped_image"].to(device)

        # first image, no matches
        outs = model(img_0)
        semi, desc = outs["semi"], outs["desc"]
        pts = getPtsFromSemi(semi[0, :, :, :], conf_thresh=det_thresh, nms_dist=nms_dist)
        desc = sample_desc_from_points(desc[0, :, :, :], pts, device)

        if outputMatches:
            tracker.update(pts, desc)

        # first image, no matches
        outs = model(img_1)
        semi_warp, desc_warp = outs["semi"], outs["desc"]
        pts_warp = getPtsFromSemi(semi_warp[0, :, :, :], conf_thresh=det_thresh, nms_dist=nms_dist)

        # ### debug
        # def plot_points(im, pts, name):
        #     img_new = im.copy()
        #     pts = np.round(pts[:]).astype(int)
        #     for p in pts:
        #         cv2.circle(img_new, p, 1, (255, 0, 0), -1)
        #     cv2.imshow(name, img_new)
        #     cv2.waitKey(0)
        # plot_points(squeezeToNumpy(img_1), pts[:2].transpose(), "test")

        # imgwarop = warp_image_batch(img_0.type(torch.FloatTensor), sample["homography"].inverse().type(torch.FloatTensor))
        # cv2.imshow("test", squeezeToNumpy(imgwarop))
        # cv2.waitKey(0)

        desc_warp = sample_desc_from_points(desc_warp[0, :, :, :], pts_warp, device)

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0), "prob": pts.transpose(), "desc": desc.transpose()}

        if outputMatches == True:
            tracker.update(pts_warp, desc_warp)

        pred.update(
            {
                "warped_image": squeezeToNumpy(img_1),
                "warped_prob": pts_warp.transpose(),
                "warped_desc": desc_warp.transpose(),
                "homography": squeezeToNumpy(sample["homography"]),
            }
        )

        if outputMatches == True:
            matches = tracker.get_matches()
            print("matches: ", matches.transpose().shape)
            pred.update({"matches": matches.transpose()})
        print("pts: ", pts.shape, ", desc: ", desc.shape)

        # clean last descriptor
        tracker.clear_desc()

        filename = str(count)
        path = Path(output_dir, f"{filename}.npz")
        np.savez_compressed(path, **pred)
        count += 1
    print("output pairs: ", count)


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)

    # add parser
    parser = argparse.ArgumentParser()

    # export command
    parser.add_argument("--config", type=str)
    parser.add_argument("--exper_name", type=str, default=None)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(config)

    pretrained = Path(config['pretrained'])
    chkpt_name = Path(str(pretrained.stem).split('.')[0])
    exper_config = pretrained.parents[1] / 'config.yml'
    # get model name
    with open(exper_config, "r") as f:
        exper_config = yaml.safe_load(f)
    # model_name = exper_config['model']['name']
    model_name = 'YOLOPoint' #'SuperPointNet'           #exper_config['model']['name']
    model_version =  's' #    None                                 #exper_config['model'].get('version')
    inp_ch = exper_config['model']['input_channels']
    model_info = {'model_name': model_name, 'version': model_version, 'inp_ch': inp_ch}

    alteration = config['data']['alteration']
    output_dir = pretrained.parents[1] / 'hpatches_eval' / chkpt_name /alteration
    os.makedirs(output_dir, exist_ok=True)

    export_descriptor(config, output_dir, model_info)
