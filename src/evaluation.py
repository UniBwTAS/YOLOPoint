import torch
from evaluations.detector_evaluation import compute_repeatability
from evaluations.descriptor_evaluation import compute_homography, sample_desc_from_points
from utils.loader import dataLoader
from utils.utils import squeezeToNumpy, getPtsFromSemi, load_model, dict_update, load_pretrained_yolo
import yaml
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
from time import time
import traceback


def evaluate(config, ss=None, device='cpu', dset=None):
    """
    Crawls through all checkpoint files in given directory and performs an evaluation with sample size 200.
    Then performs full evaluation on best n.
    config['pretrained'] --> list: exper_name or single logs/exper_name/checkpoint.pth.tar file
    ss: sample size (for fast evaluation)
    dset: dataset you want to evaluate on (defaults to val dataset of data the model was trained on)
    """

    nms_dist = config["model"]["nms"]
    conf_thresh = config["model"]["detection_threshold"]

    model_name_old = ''
    dataset_old = ''
    if not isinstance(eval_targets := config['pretrained'], list):
        # evaluate single target
        eval_targets = [eval_targets]

    for eval_target in eval_targets:
        weights_paths = glob(eval_target)
        w_total = len(weights_paths)

        for w, weights_path in enumerate(weights_paths):
            try:
                weights_path = Path(weights_path)

                # # check if already evaluated
                # metrics_file = weights_path.parent.parent / 'metrics.yml'
                # try:
                #     with open(metrics_file, 'r') as f:
                #         exper_metrics = yaml.safe_load(f)
                # except FileNotFoundError:
                #     pass

                print(f'[{w+1}/{w_total}] Evaluating {weights_path}')
                weights = torch.load(weights_path)

                with open(weights_path.parent.parent / 'config.yml', 'r') as f:
                    train_config = yaml.safe_load(f)

                # load data
                dataset = dset or train_config['data']['dataset']
                bs = {"model": {"val_batch_size": 1}}
                config = dict_update(config, bs)
                if dset is None and not dataset == dataset_old:
                    data = dataLoader(config, dataset=dataset, DEBUG=False)
                    val_loader = data['val_loader']
                    ss = ss or len(val_loader)

                # load model
                model_name = train_config['model']['name']
                input_channels = train_config['model'].get('input_channels', 1)
                if not model_name == model_name_old:
                    model = load_model(model_name, inp_ch=input_channels).to(device)
                model.load_state_dict(weights['model_state_dict'])
                model.eval()

                repeatability_list = []
                loc_error_list = []
                # mean_dist_list = []
                correctness_list = []
                inference_time = []

                with torch.no_grad():
                    for batch_idx, sample in enumerate(tqdm(val_loader)):

                        img, labels_2D, mask_2D = (
                            sample["image"],
                            sample["labels_2D"],
                            sample["valid_mask"]
                        )
                        img_warp, labels_2D_warp, mask_2D_warp = (
                            sample["warped_image"],
                            sample["warped_labels"],
                            sample["warped_valid_mask"]
                        )
                        t0 = time()
                        outs = model(img.to(device))
                        inference_time.append(time() - t0)
                        semi, desc = outs["semi"], outs["desc"]

                        outs_warp = model(img_warp.to(device))
                        semi_warp, desc_warp = outs_warp["semi"], outs_warp["desc"]

                        pts_nms = getPtsFromSemi(semi[0, :, :, :], conf_thresh, nms_dist)
                        pts_nms_warp = getPtsFromSemi(semi_warp[0, :, :, :], conf_thresh, nms_dist)
                        inv_homography = sample['inv_homographies'][0, :, :]
                        inp_img = img[0, :, :, :]
                        inp_img_warp = img_warp[0, :, :, :]

                        sparse_desc = sample_desc_from_points(desc[0, :, :, :], pts_nms, device)
                        sparse_desc_warp = sample_desc_from_points(desc_warp[0, :, :, :], pts_nms_warp, device)

                        if input_channels == 3:
                            inp_img = inp_img.transpose(0,2).transpose(0,1)
                            inp_img_warp = inp_img_warp.transpose(0,2).transpose(0,1)

                        data = {
                            'image': squeezeToNumpy(inp_img),
                            'warped_image': squeezeToNumpy(inp_img_warp),
                            'inv_homography': squeezeToNumpy(inv_homography),
                            'prob': pts_nms.transpose(),
                            'warped_prob': pts_nms_warp.transpose(),
                            'desc': sparse_desc.transpose(),
                            'warped_desc': sparse_desc_warp.transpose()
                        }

                        repeatability, loc_error = compute_repeatability(data)
                        repeatability_list.append(repeatability)
                        loc_error_list.append(loc_error)
                        hom_metrics = compute_homography(data)
                        # mean_dist_list.append(np.mean(hom_metrics['mscores']))
                        correctness_list.append(float(hom_metrics['correctness']))

                        if batch_idx >= ss:
                            break
                model_name_old = model_name
                dataset_old = dataset
                del inference_time[0]
                inference_time = round(float(np.mean(inference_time)), 4)
                metrics = {
                    str(weights_path.name):
                        {'repeatability': round(float(np.mean(repeatability_list)), 4),
                         'location_error': round(float(np.mean(loc_error_list)), 4),
                         # 'mean_distance': round(float(np.mean(mean_dist_list)), 4),
                         'correctness': round(float(np.mean(correctness_list)), 4),
                         'criterion': round(float(np.mean(correctness_list)+np.mean(repeatability_list)), 4),
                         'inference_time': inference_time}
                }
                if ss:
                    metrics[str(weights_path.name)].update({'sample_size': ss})
                metrics_file = weights_path.parent.parent / 'metrics.yml'
                try:
                    with open(metrics_file, 'r') as f:
                        exper_metrics = yaml.safe_load(f)
                        exper_metrics.update(metrics)
                except FileNotFoundError:
                    exper_metrics = metrics
                with open(metrics_file, 'w') as f:
                    yaml.dump(exper_metrics, f)
            except:
                print(f'\nAn error occured during evaluation of {eval_target}:')
                print(traceback.format_exc())

        sort_by_criterion(metrics_file)


def sort_by_criterion(metrics_path):
    # update metrics.yml with sorted
    with open(metrics_path, 'r') as f:
        metrics = yaml.safe_load(f)
        sorted_metrics = dict(reversed(sorted(metrics.items(), key=lambda kv: kv[1]['criterion'])))
    with open(metrics_path, 'w') as f:
        yaml.dump(sorted_metrics, f, sort_keys=False)


if __name__ == "__main__":
    with open("configs/evaluation.yaml", 'r') as f:
        config = yaml.safe_load(f)

    evaluate(config, device='cuda')
