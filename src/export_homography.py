"""
This script exports detection/ description using pretrained model.
"""

## basic
import argparse
import yaml
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
from utils.utils import squeezeToNumpy

## torch
import torch
import torch.optim
import torch.utils.data
from utils.loader import dataLoader
from utils.utils import flattenDetection, data_size, load_model, warp_image_batch, getPtsFromHeatmap


@torch.no_grad()
def homographic_export(config, output_dir, export_task):
    """
    input 1 images, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    TODO: do homography adaptation for cropped images to enable training on larger images
    """

    # basic setting
    dataset = config["data"]["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Export {dataset} {export_task} data on device: {device}")

    ## parameters
    nms_dist = config["model"]["nms"]  # 4
    top_k = config["model"]["top_k"]
    conf_thresh = config["model"]["detection_threshold"]

    ## save data
    save_path = Path(output_dir)
    save_output = save_path / export_task
    print(f"=> will save everything to {save_output}")
    save_output.mkdir(parents=True, exist_ok=True)

    # data loading
    loader = dataLoader(config, export=True, action=export_task)
    data_size(loader, config, tag=export_task)

    # model loading
    weights_path = config["pretrained"]
    try:
        print("==> Loading pre-trained network.")
        print("path:", weights_path)
        model_conf = config["model"]
        ch = model_conf["input_channels"]
        version = model_conf["version"]
        names = config["names"]
        model_name = config["model"]["name"]
        model = load_model(inp_ch=ch, names=names, version=version, model_name=model_name)
        model.load_state_dict(torch.load(weights_path)['model_state_dict'])
        model.eval().to(device)
        print("==> Successfully loaded pre-trained network.")

    except Exception:
        print(f"load model: {weights_path} failed!")
        raise

    # loop through all images
    for i, sample in enumerate(tqdm(loader, position=0)):

        name = sample["name"][0]
        print(f"name: {name}")

        p = Path(save_output, f"{name}.npz")
        # if p.exists():
        #     print(f"File {name} exists. Skipping sample.")
        #     continue

        img, mask_2D = sample["image"], sample["valid_mask"]
        img = img.squeeze().to(device)

        # for i in range(img.shape[0]):
        #     im = img[i].cpu().numpy()
        #     im = im.transpose(1, 2, 0)
        #     cv2.imshow("im", im)
        #     cv2.waitKey(0)

        mask_2D = mask_2D.transpose(0, 1).to(device)

        inv_homographies = sample["inv_homographies"].to(device)

        # pass through network
        output = model(img)
        semi = output['semi']
        heatmap = flattenDetection(semi)
        ### combine heatmap
        heatmap = heatmap * mask_2D
        heatmap = warp_image_batch(heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear")
        mask_2D = warp_image_batch(mask_2D, inv_homographies[0, :, :, :], device=device, mode="bilinear")

        height, width = heatmap.shape[2:]

        if pad := sample.get('pad'):
            if pad[1]:
                heatmap = heatmap[:, :, pad[0]:width - pad[1], :]
                mask_2D = mask_2D[:, :, pad[0]:width - pad[1], :]
            if pad[3]:
                heatmap = heatmap[:, :, :, pad[2]:height - pad[3]]
                mask_2D = mask_2D[:, :, :, pad[2]:height - pad[3]]

        debug = False
        if debug:
            inv_warp_image = warp_image_batch(img, inv_homographies[0, :, :, :], device=device, mode="bilinear")
            for i in range(img.shape[0]):
                hm = squeezeToNumpy(heatmap[i, :, :, :])
                img_ = squeezeToNumpy(inv_warp_image[i, :, :, :])*255.
                img_warped = squeezeToNumpy(img[i, :, :, :]*255.).astype(np.uint8).transpose((1, 2, 0))
                img_ = img_.astype(np.uint8).transpose((1, 2, 0))
                img_ = np.ascontiguousarray(img_)
                pts_ = getPtsFromHeatmap(hm, conf_thresh, nms_dist).transpose()[:300,:2].astype(int)
                pts_ += np.array([squeezeToNumpy(pad[2]), squeezeToNumpy(pad[0])])
                for p in pts_:
                    # xy
                    cv2.circle(img_, p, 2, (255, 0, 0), -1)
                cv2.imshow("unwarped", img_)
                cv2.imshow("original", img_warped)
                k = cv2.waitKey(0)
                if k == ord('s'):
                    # save images
                    cv2.imwrite(f"img_{i}.png", img_)
                    cv2.imwrite(f"img_warped{i}.png", img_warped)

        heatmap = torch.sum(heatmap, dim=0)
        mask_2D = torch.sum(mask_2D, dim=0)
        outputs = heatmap / mask_2D

        pts = getPtsFromHeatmap(outputs.detach().cpu().squeeze(), conf_thresh, nms_dist)  # (x,y, prob)

        ## top K points
        pts = pts.transpose()
        if top_k and pts.shape[0] > top_k:
            pts = pts[:top_k, :]

        if debug:
            # plot for debugging
            # for i in range(img.shape[0]):
            top = int(pad[0].cpu().numpy())
            bottom = int(pad[1].cpu().numpy())
            left = int(pad[2].cpu().numpy())
            right = int(pad[3].cpu().numpy())
            img_np = squeezeToNumpy(img[0]).transpose((1,2,0))
            # crop image
            img_np = img_np[top:img_np.shape[0]-bottom, left:img_np.shape[1]-right, :]
            points = pts[:, :2].astype(int)
            img_np = np.ascontiguousarray(img_np)
            print(img_np.shape)
            for p in points:
                cv2.circle(img_np, p, 2, (255, 0, 0), -1)
            cv2.imshow('test', img_np)
            cv2.waitKey(0)

        if config.get('normalize_points'):
            (H, W) = sample['dims'][1]
            H = H.numpy()
            W = W.numpy()
            pts[:,0] /= W
            pts[:,1] /= H

        ## save keypoints
        pred = {"pts": pts}

        ## - make directories
        filename = str(name)

        path = Path(save_output, f"{filename}.npz")
        np.savez_compressed(path, **pred) # comment this in later
        print(path)


if __name__ == "__main__":
    # from glob import glob
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # using homography adaptation to export detection psuedo ground truth
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", type=str)
    # parser.add_argument('--model', type=str)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--tasks', type=str, nargs='+', default=['train', 'val'], help="'train' or 'val' or both")

    args = parser.parse_args()

    # Ensure that all relative paths start from project root
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_directory, '..'))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # ######## for automatic execute ######## ---> comment out later
    # config['pretrained'] = glob(os.path.join('logs/Magicpoint_YOLO_M_det/checkpoints/*_best.pth.tar'))[0]
    print("config:", config)

    output_dir = os.path.join('logs', args.exper_name)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)

    for export_task in args.tasks:
        print(f"Exporting {export_task} set")
        homographic_export(config, output_dir, export_task)
    #python src/export_homography.py configs/coco_export.yaml test --tasks val