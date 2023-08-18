import torch
import os
import argparse
import yaml
from utils.loader import get_save_path, dataLoader
# from utils.loss_functions import descriptor_loss_dense, descriptor_loss_sparse, ComputeDetectorLoss, ComputeObjectLoss, \
#     infonce_loss_2

from utils.loss_functions import  ComputeDetectorLoss, ComputeObjectLoss
from utils.loss_functions import infonce as descriptor_loss_sparse



from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from utils.utils import flattenDetection, toNumpy, img_overlap, data_size, load_model, labels2Dto3D, \
    getMasks, squeezeToNumpy, getPtsFromSemi, dict_update, filter_invalid_pts, parse_str_slice
import numpy as np
from time import time
from glob import glob
from evaluations.detector_evaluation import compute_repeatability, batch_precision_recall
from evaluations.descriptor_evaluation import compute_homography, sample_desc_from_points
from evaluations.yolo_evaluation import process_batch
import cv2
# from utils.torch_utils_yolo import smart_optimizer, ModelEMA
from utils.general_yolo import non_max_suppression, scale_boxes, xywh2xyxy, LOGGER, check_version
from utils.metrics_yolo import ConfusionMatrix, ap_per_class
from utils.metrics_yolo import fitness as yolo_fitness
from utils.plots_yolo import Annotator, colors
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

class TrainAgent:

    def __init__(self, config, output_dir, args):

        self.config = config
        self.args = args
        nbs = 64    # nominal batch size
        bs = self.config['training_params']['train_batch_size']
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.gas = max(round(nbs / (bs * num_devices)), 1)
        if self.gas > 1:
            LOGGER.info(f'Doing gradient accumulation with {self.gas} steps')
        # set broacast_buffers to False to avoid issues with batch norm layers and DDP
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gas,
                                       kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
        self.device = self.accelerator.device

        LOGGER.info(f'Training on {self.accelerator.num_processes} device(s)')

        self.epoch = 1
        self.start = 1
        self.step = 1
        self.best = False

        # Init network
        self.names = self.config['names']
        if 'dontcare' in (name.lower() for name in self.names):
            assert self.names[-1].lower() == 'dontcare', 'DontCare class must be at the end of the class list or else labels' \
                                                 'will get mixed up!'
            LOGGER.info("Removing 'don't care' regions")
            self.names = self.names[:-1]
        self.nc = len(self.names)
        self.input_channels = self.config['model'].get('input_channels', 3)
        self.version = args.version
        self.model_name = args.model
        self.model = load_model(model_name=args.model, inp_ch=self.input_channels, names=self.names, version=args.version).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=config['optimizer']['lr'])

        # Scheduler
        lrf = self.config.get('scheduler', {}).get('lrf', 1)
        lf = lambda x: (1 - x / self.config['epochs']) * (1.0 - lrf) + lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        if weights_path := self.config.get('pretrained'):
            ckpt = torch.load(weights_path)
            if ckpt.get('names') == self.names:
                # continue with training
                self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
                if self.config.get('resume'):
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    self.best = True
                    self.start = ckpt['epoch']+1
                    self.step = ckpt['global_step']
                    self.fitness = ckpt['fitness']
                LOGGER.info(f'Continuing training from {self.start} epoch')
            else:
                # load all but Detect layers
                self.model.load_state_dict(ckpt['model_state_dict'], verbose=False, strict=True)
                # Warm Start
                if ws := self.config.get('warm_start'):
                    self._shrink_perturb(**ws)
            LOGGER.info(f'Loaded weights: {weights_path}')

        # Superpoint sparse loss
        self.sparse_loss_params = self.config['model']['superpoint'].get('sparse_loss', {})

        # fitness
        self.fitness = 0.
        self.save_best = self.config['save_best']

        if to_freeze := config.get('freeze_layers'):
            to_freeze = parse_str_slice(to_freeze)
            self.model.freeze_layers(to_freeze)

        self.gradclip = self.config.get('gradient_clip')

        self.save_path = get_save_path(output_dir)

        # data loading
        # check for multiple datasets

        self.train_loader = dataLoader(deepcopy(self.config), DEBUG=args.debug, action='train')
        self.val_loader = dataLoader(deepcopy(self.config), DEBUG=args.debug, action='val')
        self.train_loader, self.val_loader = self.accelerator.prepare(self.train_loader, self.val_loader)
        # logging info
        data_size(self.train_loader, self.config, tag='train')
        data_size(self.val_loader, self.config, tag='val')

        tb_output_dir = os.path.join(output_dir, "tensorboard")
        self.writer = SummaryWriter(tb_output_dir)

        self.valid_epoch = 1
        self.debug = args.debug
        self.lambda_loss = self.config['model']['lambda_loss']
        self.lambda_loss_obj = self.config['model']['lambda_loss_obj']

        self.nms_dist = self.config["model"]["superpoint"]["nms"]
        self.conf_thresh = self.config["model"]["superpoint"]["detection_threshold"]

        self.joint_training = self.config["joint_training"]
        if not self.joint_training:
            LOGGER.info(f'Training point detector only')

        # yolo loss scaling attributes
        nl = 3
        self.config['model']['yolo']['box'] *= 3 / nl  # scale to layers
        self.config['model']['yolo']['cls'] *= self.nc / 80
        try:
            img_size = self.config['data']['preprocessing']['img_size']
        except KeyError:
            img_size = self.config['data']['augmentation']['homographic']['cropHW'][1]
        self.config['model']['yolo']['obj'] *= (img_size / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.compute_object_loss = ComputeObjectLoss(self.model, self.config['model']['yolo'], self.device)  # init loss class

        self.compute_det_loss = ComputeDetectorLoss(self.device)

        # # torch.compile
        # if check_version(torch.__version__, '2.0.0'):
        #     self.model = torch.compile(self.model)

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

    def train(self):
            try:
                # train network
                LOGGER.info(f"\nBeginning training on {self.device}\n")
                for self.epoch in tqdm(range(self.start, self.config['epochs']+1), desc="epoch", position=0):
                    self.accelerator.wait_for_everyone()
                    train_losses = []
                    self.model.train()

                    current_lr = {f'lr{i}': x for i, x in enumerate(self.scheduler.get_last_lr())}
                    if self.accelerator.is_main_process:
                        self.writer.add_scalars("training/LR", current_lr, global_step=self.step)
                    fetch_start = time()
                    for batch_idx, sample in enumerate(tqdm(self.train_loader, desc="batch", position=1, leave=False)):
                        with self.accelerator.accumulate(self.model):
                            self.optimizer.zero_grad()

                            # Fetch data
                            img, labels_2D, mask_2D, box_targets = (
                                sample["image"].to(self.device),
                                sample["labels_2D"].to(self.device),
                                sample["valid_mask"].to(self.device),
                                sample["box_labels"].to(self.device)
                            )

                            img_warp, labels_2D_warp, mask_2D_warp = (
                                sample["warped_image"].to(self.device),
                                sample["warped_labels"].to(self.device),
                                sample["warped_valid_mask"].to(self.device)
                            )

                            # Forward pass
                            outs = self.model(img)
                            semi, desc, obj = outs["semi"], outs["desc"], outs["objects"]

                            # loss yolo
                            loss_obj, loss_obj_items = self.compute_object_loss(obj, box_targets.to(self.device))

                            # loss superpoint
                            labels_3D = labels2Dto3D(labels_2D).to(self.device)
                            mask_3D_flattened = getMasks(mask_2D, device=self.device)

                            loss_det = self.compute_det_loss(semi, labels_3D, mask_3D_flattened)

                            outs_warp = self.model(img_warp.to(self.device))
                            semi_warp, desc_warp = outs_warp["semi"], outs_warp["desc"]
                            labels_3D_warp = labels2Dto3D(labels_2D_warp).to(self.device)
                            mask_3D_flattened_warp = getMasks(mask_2D_warp, device=self.device)

                            loss_det_warp = self.compute_det_loss(semi_warp, labels_3D_warp, mask_3D_flattened_warp)
                            if self.joint_training:
                                t0 = time()
                                # loss_desc = descriptor_loss_dense(desc, desc_warp, sample["inv_homographies"], mask_3D_flattened.unsqueeze(1),
                                #                             lambda_d=config['model']["superpoint"]['lambda_d'], device=self.device)
                                loss_desc = descriptor_loss_sparse(desc, desc_warp, mask_2D_warp, sample['inv_homographies'],
                                                        **self.sparse_loss_params, device=self.device)
                                # loss_desc = info_nce_loss(desc, desc_warp, mask_2D_warp, sample['inv_homographies'],
                                #                         **self.sparse_loss_params, device=self.device)
                                loss_time = time() - t0
                            else:
                                loss_desc, POS, NEG = torch.tensor([0.]).to(self.device), 0., 0.

                            loss_desc *= self.lambda_loss
                            loss_det_sum = loss_det + loss_det_warp
                            loss_obj *= self.lambda_loss_obj
                            loss = loss_det_sum + loss_desc + loss_obj

                            # Backward pass
                            t0 = time()
                            self.accelerator.backward(loss)
                            backpass_time = time() - t0

                            train_losses.append(loss.item() * self.gas)
                            if self.gradclip:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradclip)  # clip gradients

                            self.optimizer.step()

                            tb_losses = {"training/Detector loss": loss_det_sum.item(),
                                         "training/Descriptor loss": loss_desc.item(),
                                         "training/Object detector loss": loss_obj.item(),
                                         "training/Training loss": loss.item() * self.gas
                            }
                            self._gather_and_write_to_tb(tb_losses, ret=False, task='training')

                            # self.writer.add_scalar("training/Detector loss", loss_det_sum.item(), global_step=self.step)
                            # self.writer.add_scalar("training/Descriptor loss", loss_desc.item(), global_step=self.step)
                            # self.writer.add_scalar("training/Object detector loss", loss_obj.item(), global_step=self.step)
                            # self.writer.add_scalar("training/Training loss", loss.item()*self.gas, global_step=self.step)

                            # self.writer.add_scalars("time/time", {
                            #     "contrastive": loss_time,
                            #     "backpass": backpass_time,
                            #     "optimize": optim_time,
                            #     "iteration": iter_time,
                            #     "fetch": fetch_time,
                            #     "forward": forward_time
                            # }, global_step=self.step)
                            self.step += 1
                        # end batch ------------------------------------------------------------------------------------
                    # end epoch ----------------------------------------------------------------------------------------
                    self.scheduler.step()
                    train_loss = np.mean(train_losses)
                    LOGGER.info(f"train loss: {train_loss:.4f}")

                    if self.epoch % self.config["validation_interval"] == 0:
                        val_stats = self._validate()

                    if self.epoch % self.config["save_interval"] == 0:
                        if self.valid_epoch != self.epoch:
                            val_stats = self._validate()
                        model_stats = {
                            'train_loss': round(train_loss, 4),
                            **val_stats
                        }
                        # save model
                        if self.best and self.save_best:
                            self._save_model(model_stats, filename='best.pth.tar', delete_old=True)
                            self.best = False
                        elif not self.save_best:
                            self._save_model(model_stats)
                # end training -----------------------------------------------------------------------------------------

            except KeyboardInterrupt:
                if self.epoch > 1:
                    LOGGER.info("Training interrupted, saving model")
                    if self.valid_epoch == 1:
                        model_stats = {'train_loss': round(train_loss, 4), 'val_loss': -1.}

                    self._save_model(model_stats, filename='last.pth.tar')

            LOGGER.info("Training complete")

    def _validate(self):
        self.valid_epoch = self.epoch
        self.model.eval()
        precision = torch.Tensor().to(self.device)
        recall = torch.Tensor().to(self.device)
        val_losses = []
        det_losses = []
        desc_losses = []
        obj_losses = []
        extended_val_metrics = {}
        repeatability_list = []
        loc_error_list = []
        correctness_list = []
        num_tb_img = min(self.config['training_params']['val_batch_size'], 6) # number of images to tensorboard
        sample_size = self.config.get('extended_val_sample_size')   # extended validation size (repeatability and homography estimation)
        count = 0
        objects_seen = 0
        yolo_stats = []
        confusion_matrix = ConfusionMatrix(nc=self.nc)

        with torch.inference_mode():

            for batch_idx, sample in enumerate(tqdm(self.val_loader, desc="val", position=2, leave=False)):

                img, labels_2D, mask_2D, box_targets = (
                sample["image"],
                sample["labels_2D"],
                sample["valid_mask"],
                sample["box_labels"]
                )

                img_warp, labels_2D_warp, mask_2D_warp = (
                    sample["warped_image"],
                    sample["warped_labels"],
                    sample["warped_valid_mask"]
                )

                # forward
                outs = self.model(img.to(self.device))
                semi, desc, (preds_batch, obj) = outs["semi"], outs["desc"], outs["objects"]

                outs_warp = self.model(img_warp.to(self.device))
                semi_warp, desc_warp, obj_warp = outs_warp["semi"], outs_warp["desc"], outs_warp["objects"]

                labels_3D = labels2Dto3D(labels_2D).to(self.device)
                mask_3D_flattened = getMasks(mask_2D, device=self.device)
                loss_det = self.compute_det_loss(semi, labels_3D, mask_3D_flattened)

                preds_batch = non_max_suppression(preds_batch,
                                            self.config['model']['yolo']['conf_thres_box'],
                                            self.config['model']['yolo']['iou_thres_box'],
                                            labels=[],
                                            multi_label=True,
                                            agnostic=self.nc==1,
                                            max_det=self.config['model']['yolo']['max_det'])

                # extended validation
                if sample_size and count < sample_size:

                    for idx in range(sample['image'].shape[0]):
                        pts_nms = getPtsFromSemi(semi[idx, :, :, :], conf_thresh=self.conf_thresh, nms_dist=self.nms_dist)
                        pts_nms_warp = getPtsFromSemi(semi_warp[idx, :, :, :], conf_thresh=self.conf_thresh, nms_dist=self.nms_dist)

                        # filter points
                        pts_nms = filter_invalid_pts(pts_nms, squeezeToNumpy(mask_2D[idx]))
                        pts_nms_warp = filter_invalid_pts(pts_nms_warp, squeezeToNumpy(mask_2D_warp[idx]))

                        homography = sample['homographies'][idx, :, :]
                        inv_homography = sample['inv_homographies'][idx, :, :]
                        inp_img = img[idx, :, :, :]
                        inp_img_warp = img_warp[idx, :, :, :]

                        if self.joint_training:
                            sparse_desc = sample_desc_from_points(desc[idx, :, :, :], pts_nms, self.device)
                            sparse_desc_warp = sample_desc_from_points(desc_warp[idx, :, :, :], pts_nms_warp, self.device)
                        else:
                            sparse_desc = np.zeros(1) # dummy
                            sparse_desc_warp = np.zeros(1)    # dummy

                        if self.input_channels == 3:
                            inp_img = inp_img.transpose(0,2).transpose(0,1)
                            inp_img_warp = inp_img_warp.transpose(0,2).transpose(0,1)

                        inp_img = squeezeToNumpy(inp_img)
                        inp_img = np.ascontiguousarray(inp_img)
                        data = {
                            'image': inp_img,
                            'warped_image': squeezeToNumpy(inp_img_warp),
                            'homography': squeezeToNumpy(homography),
                            'inv_homography': squeezeToNumpy(inv_homography),
                            'prob': pts_nms.transpose(),
                            'warped_prob': pts_nms_warp.transpose(),
                            'desc': sparse_desc.transpose(),
                            'warped_desc': sparse_desc_warp.transpose()
                        }

                        repeatability, loc_error = compute_repeatability(data)
                        repeatability_list.append(repeatability)
                        loc_error_list.append(loc_error)

                        return_img = (idx < num_tb_img and batch_idx == 0 and self.accelerator.is_local_main_process)
                        if self.joint_training:
                            hom_metrics = compute_homography(data, return_mimg=return_img)
                            correctness_list.append(float(hom_metrics['correctness']))

                        # write qualitative results to tensorboard
                        if return_img:
                            preds = preds_batch[idx]
                            annotated_img = inp_img.copy()
                            # inp_img soll ndarray H,W,C
                            annotator = Annotator(annotated_img, line_width=2, example=str(self.names))
                            if len(preds):
                                preds[:, :4] = scale_boxes(annotated_img.shape[:2], preds[:, :4], annotated_img.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(preds):
                                # Add bbox to image
                                c = int(cls)  # integer class
                                label = f'{self.names[c]} {conf:.2f}'
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                # Stream results
                                annotated_img = annotator.result()
                            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            if self.accelerator.is_main_process:
                                self.writer.add_image(f"yolo_detections/{idx}", annotated_img, self.step, dataformats='HWC')

                            if self.joint_training and hom_metrics:
                                # match image
                                matches_img = hom_metrics['matches_img'].transpose(1,2,0)
                                matches_img = cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB)
                                self.writer.add_image(f'matches/{idx}', matches_img, self.step, dataformats='HWC')

                            if self.input_channels == 3:
                                data['image'] = cv2.cvtColor(data['image'], cv2.COLOR_BGR2GRAY)

                            # kp detections image
                            semi_thd_nms_sample = np.zeros_like(data['image'])
                            semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1

                            result_overlap = img_overlap(
                                toNumpy(labels_2D[idx, :, :, :]),
                                np.expand_dims(semi_thd_nms_sample, 0),
                                np.expand_dims(data['image'], 0)
                            )
                            if self.accelerator.is_main_process:
                                self.writer.add_image(f"detector_output_overlay-NMS/{idx}", result_overlap, self.step)

                        count += 1

                labels_3D_warp = labels2Dto3D(labels_2D_warp).to(self.device)
                mask_3D_flattened_warp = getMasks(mask_2D_warp, device=self.device)
                loss_det_warp = self.compute_det_loss(semi_warp, labels_3D_warp, mask_3D_flattened_warp)
                if self.joint_training:
                    # loss_desc = descriptor_loss_dense(desc, desc_warp, sample["inv_homographies"], mask_3D_flattened.unsqueeze(1),
                    #                                 lambda_d=config['model']["superpoint"]['lambda_d'], device=self.device)
                    loss_desc = descriptor_loss_sparse(desc, desc_warp, mask_2D_warp, sample['inv_homographies'],
                                            device=self.device, **self.sparse_loss_params)
                else:
                    loss_desc = torch.tensor([0.]).to(self.device)

                loss_det_sum = loss_det + loss_det_warp
                loss_desc = self.lambda_loss * loss_desc
                loss_obj, loss_obj_items = self.compute_object_loss(obj, box_targets.to(self.device))  # loss scaled by batch_size
                loss = loss_det_sum + loss_desc + loss_obj
                det_losses.append(loss_det_sum.item())
                val_losses.append(loss.item())
                desc_losses.append(loss_desc.item())
                obj_losses.append(loss_obj.item())
                semi_flat_tensor = flattenDetection(semi)

                # YOLO eval
                nb, _, height, width = img.shape
                box_targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels

                for si, pred in enumerate(preds_batch):
                    labels = box_targets[box_targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    shape = sample['shapes'][si][0]
                    iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou vector for mAP@0.5:0.95
                    niou = iouv.numel()
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=self.device)  # init
                    objects_seen += 1

                    if npr == 0:
                        if nl:
                            yolo_stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                            # confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    else:
                        # Predictions
                        if self.nc==1:
                            pred[:, 5] = 0
                        predn = pred.clone()
                        # scale_boxes(img[si].shape[1:], predn[:, :4], shape, sample['shapes'][si][1])  # native-space pred

                        # Evaluate
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        # scale_boxes(img[si].shape[1:], tbox, shape, sample['shapes'][si][1])  # native-space labels
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)
                        confusion_matrix.process_batch(predn, labelsn)
                        yolo_stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                # Compute PR of batch (not mean)
                precision_recall = batch_precision_recall(semi_flat_tensor, labels_2D.to(self.device))
                precision = torch.concat((precision, precision_recall['precision']))
                recall = torch.concat((recall, precision_recall['recall']))

        # Compile metrics
        if sample_size:
            extended_val_metrics = {'Repeatability': np.mean(repeatability_list)}
            if self.joint_training:
                extended_val_metrics.update({'Correctness': np.mean(correctness_list),})

        extended_val_metrics = self._gather_and_write_to_tb(extended_val_metrics)

        precision = torch.mean(precision).item()
        recall = torch.mean(recall).item()
        self.dice = 2 * recall * precision / (recall + precision)

        yolo_stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*yolo_stats)]  # to numpy
        if len(yolo_stats) and yolo_stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*yolo_stats, plot=True, save_dir=self.save_path, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean() # precision, recall, meanAP50?
            yolo_fi = yolo_fitness(np.array((mp, mr, map, map50)).reshape(1, -1))
        else:
            yolo_fi = 0.0
            mp, mr, map50, map = 0.0,0.0,0.0,0.0

        val_metrics = {"Dice": self.dice,
                       "Precision (yolo)": mp,
                       "Recall (yolo)": mr,
                       "MAP50 (yolo)": map50,
                       "MAP (yolo)": map,
                       "Fitness (yolo)": yolo_fi,
                       "Detector loss": np.mean(det_losses),
                       "Descriptor loss": np.mean(desc_losses),
                       "Object loss": np.mean(obj_losses),
                       "val_loss": np.mean(val_losses)}

        val_metrics = self._gather_and_write_to_tb(val_metrics)

        repeatability = extended_val_metrics.get('Repeatability', torch.tensor([0]))
        homography = extended_val_metrics.get('Correctness', torch.tensor([0]))
        yolo_fi = val_metrics['Fitness (yolo)']

        repeatability, homography = repeatability.item(), homography.item()
        superpoint_fi = 0.55 * repeatability + 0.45 * homography
        fi = 0.3 * superpoint_fi + 0.7 * yolo_fi
        if fi > self.fitness:
            self.fitness = fi
            self.best = True

        LOGGER.info(f"val_loss: {val_metrics['val_loss']:.5f} | "
             f"fitness: {self.fitness:.4f} | "
              f"repeatability: {repeatability:.4f} | "
              f"homography: {homography:.4f} | "
              f"MAP50: {map50:.4f}")

        return {**val_metrics, **extended_val_metrics}

    def _save_model(self, stats, filename='ckpt.pth.tar', delete_old=False):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            last = 'last_' if self.config['epochs'] == self.epoch else ''
            deb = 'debug_' if self.debug else ''
            filename = f'{self.args.exper_name}_{self.epoch}_{self.step}_{deb}{last}{filename}'
            full_path = os.path.join(self.save_path, filename)
            LOGGER.info(f"Saving checkpoint to {filename}; current loss: {stats['val_loss']:.3f}")
            if delete_old:
                old_checkpoints = glob(os.path.join(self.save_path, '*.pth*'))
                for cp in old_checkpoints:
                    os.remove(cp)
            torch.save({
                "epoch": self.epoch,
                "global_step": self.step,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "fitness": self.fitness,
                "scheduler_state_dict": self.scheduler.state_dict(),
                "names": self.names,
                "version": self.version,
                "model_name": self.model_name,
                **stats
            },
                full_path)

    def _shrink_perturb(self, lamda=0.5, sigma=0.01):
        # use this when fine-tuning on new data
        LOGGER.info("Shrinking and perturbing pretrained weights")
        for (name, param) in self.model.named_parameters():
            if 'weight' in name:  # just weights
                mu = torch.zeros_like(param.data)
                param.data = param.data * lamda + torch.normal(mean=mu, std=sigma)

    def _gather_and_write_to_tb(self, metrics_dict, task='validation', ret=True):
        # convenience function for writing scalars to tensorboard during distributed training
        for key in metrics_dict:
            metrics_dict[key] = torch.tensor(metrics_dict[key]).to(self.device)
        gathered_val_metrics = self.accelerator.gather(metrics_dict)
        for key in gathered_val_metrics:
            gathered_val_metrics[key] = torch.mean(gathered_val_metrics[key])
            if self.accelerator.is_main_process:
                self.writer.add_scalar(task + '/' + key, gathered_val_metrics[key].item(), global_step=self.step)
        if ret:
            return gathered_val_metrics

def main(config, output_dir, args):
    TA = TrainAgent(config, output_dir, args)
    TA.train()

if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--exper_name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--debug', action='store_true', default=False, help='turn on debugging mode')
    parser.add_argument('--version', type=str, help="Choose one of s, m, l, x")
    args = parser.parse_args()

    # Ensure that all relative paths start from project root
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_directory, '..'))

    output_dir = os.path.join('logs', args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        model_info = {'model': {'name': args.model, 'version': args.version}}
        config = dict_update(config, model_info)

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(os.path.join(output_dir, 'run_tb.sh'), 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    echo "Tensorboard: {args.exper_name}"
    tensorboard --logdir ../{args.exper_name} --window {args.exper_name}
    ''')

    main(config, output_dir, args)

    # CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file {configs/accelerate_config.yaml} {train.py} {--config configs/superpoint_coco_train_YOLO.yaml} {--exper_name YOLOPoint_M} {--model YOLOPoint_M}
