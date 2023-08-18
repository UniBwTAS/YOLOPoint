import torch
from torch.utils.data import Dataset
import os
from glob import glob
import numpy as np
from utils.utils import warp_image, warp_points, compute_valid_mask, warpLabels, squeezeToNumpy, \
    homography_scaling, get_labels, warp_image_batch
from utils.homographies import sample_homography, flip, RandomFixedSizeCrop
from utils.photometric import imgPhotometric
import copy
from utils.general_yolo import xywhn2xyxy, xyxy2xywhn, clip_boxes, LOGGER, make_divisible
import cv2
from utils.augmentations_yolo import box_candidates, letterbox
import random
import pickle
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path
try:
    import tensorflow as tf
except ImportError:
    pass
#from utils.debug import timeit


class DataBaseClass(Dataset, ABC):

    def __init__(self, transform=None, action="train", DEBUG=False, export=False, return_points=False, **config):

        self.config = config
        self.transform = transform
        self.img_size = self.config['preprocessing']['img_size']
        self.mosaic = self.config['augmentation'].get('mosaic')
        self.augment = self.config['augmentation'] is not None
        self.return_points = return_points
        self.cache_labels = self.config.get('labels', {}).get('cache')
        self.inp_ch = self.config.get('input_channels', 3)

        dataset=config['dataset'].lower()
        suffix = self.config['suffix']
        if isinstance(suffix, str):
            suffix = [suffix]

        debug_len = 512

        if "train" in action:
            self.action = "train" if not DEBUG else "val"  # overfitting test
        elif "val" in action:
            self.action = "val"

        self._do_homographic_augm = self.config["augmentation"]["homographic"][f"enable_{self.action}"] if not export else False

        data_dir = os.path.join("datasets", dataset)
        siz = '' if not os.path.isdir(os.path.join(data_dir, 'images'+str(self.img_size))) else str(self.img_size)
        self.img_paths = []
        for s in suffix:
            self.img_paths += glob(os.path.join(data_dir, 'images'+siz, self.action, '*'+s))
        self.img_paths = sorted(self.img_paths)
        # self.img_paths = self.img_paths[::-1]       # delete this afterwards!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        n = len(self.img_paths)  # number of images
        assert n > 0, f"Zero images found in {data_dir}/images{str(self.img_size)}/{self.action}"

        # Load labels or just paths
        if not export:
            if os.path.isdir(labels_dir:=os.path.join(data_dir, 'labels' + siz, self.action)) or \
                    os.path.isdir(labels_dir:=os.path.join(data_dir, 'labels', self.action)):
                self.labels_dir = labels_dir
                if self.cache_labels:
                    cache_path = os.path.join(labels_dir, 'cache')
                    LOGGER.info(f"Loading labels from {labels_dir}")
                    if os.path.isfile(cache_path):
                        with open(cache_path, 'rb') as labels:
                            self.labels = pickle.load(labels)
                            self.labels = [x.astype(np.float32) for x in self.labels]
                    else:
                        LOGGER.info("Creating labels cache in", labels_dir)
                        self.labels = self._cache_object_labels(ret_labels=True)
                else:
                    self.labels = [None] * n

            else:
                raise FileNotFoundError(f"Labels directories {labels_dir} and {labels_dir + siz} do not exist!")

        self.indices = range(n) if not DEBUG else range(debug_len)

        # points
        self.point_paths = [None] * n
        if points_dir := self.config.get('labels', {}).get('points'):
            points_dir = os.path.join(points_dir, self.action)
            if os.path.isfile(cache_path := os.path.join(points_dir, 'cache')) and self.cache_labels:
                LOGGER.info(f"Loading labels from {points_dir}")
                with open(cache_path, 'rb') as points:
                    self.points = pickle.load(points)
            else:
                LOGGER.info(f"Loading labels from {points_dir}")
                self.point_paths = sorted(glob(os.path.join(points_dir, "*.npz")))
                # self.point_paths = self.point_paths[::-1]       # delete this line afterwards!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if self.cache_labels:
                    LOGGER.info('Creating labels cache in:', cache_path)
                    all_points = []
                    for point_path in tqdm(self.point_paths, desc='Creating point labels cache...'):
                        points = np.load(point_path)
                        all_points.append(points['pts'])
                    with open(cache_path, 'wb') as f:
                        pickle.dump(all_points, f)
                self.points = [None] * n
                # assert len(self.point_paths) == len(self.img_paths) and len(self.img_paths) > 0, \
                #     f"Zero or non-matching number of images ({len(self.img_paths)}) and labels ({len(self.point_paths)})"

        if config.get('cache_images'):
            self.imgs = []
            mem = 0
            pbar = tqdm(self.img_paths, desc='Loading images into cache...')
            for im_path in pbar:
                img = cv2.imread(im_path)
                self.imgs.append(img)
                mem += img.nbytes
                mem_str = f'{mem / 1E9:.2f}'+' Gb'
                pbar.set_postfix({'memory': mem_str})
        else:
            self.imgs = [None] * n

        if not export:
            self.enable_photo_train = self.config["augmentation"]["photometric"][f"enable_{self.action}"]

        self._do_only_crop = False
        if cropHW := self.config["augmentation"]["homographic"].get('cropHW'):
            center_crop = self.action == "val"
            self.random_crop = RandomFixedSizeCrop(cropHW, center_crop=center_crop)
            self.img_size = cropHW[1]
            self.cropHW = cropHW
            self._do_only_crop = self.action == "val"
        else:
            self.random_crop = None
            self.cropHW = None

        if DEBUG:
            self.imgs = self.imgs[:debug_len]
            self.img_paths = self.img_paths[:debug_len]
            self.point_paths = self.point_paths[:debug_len]
            self.points = self.points[:debug_len]
            self.labels = self.labels[:debug_len]
            self.indices = self.indices[:debug_len]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        if self._do_homographic_augm and self.mosaic >= random.random():
            return self._mosaic_augmentation(idx)

        return self._get_single_sample(idx)

    def _get_single_sample(self, index, mosaic_piece=False):
        image, (H0, W0), (H, W) = self.load_image(index)   # HW0 original scale, HW rescaled to fit self.img_size
        H_, W_ = H, W # no letterbox dims
        shape = (H, W)
        pad = [0]*6
        ratio = [1, 1]

        if not mosaic_piece and self.random_crop is None and self._letterbox:
            image, ratio, pad = letterbox(image, self.img_size, auto=False, scaleup=self.augment, color=(0, 0, 0))
            H += pad[0] + pad[1]
            W += pad[2] + pad[3]

        shapes = (H0, W0), ((H_ / H0, W_ / W0), (pad[4], pad[5]))
        name = self._get_basename(self.img_paths[index])

        sample = {"name": name, "index": index, "HW": (H_, W_, pad[4], pad[5]), "shapes": shapes}   # HW and shapes might break for homography adaptation

        if self.config.get('homography_adaptation'):    # probably deprecated by now
            return self._do_homographic_adaptation(image, shape, pad, sample, H0, W0, H, W)

        if self.points[index] is not None:
            points = self.points[index]
        elif self.point_paths:
            point_path = self.point_paths[index]
            points = np.load(point_path)["pts"]  # (x, y)
            # warp points to rescaled image scale (if at original scale)
        else:
            points = np.empty([0,3])
        points[:, 0] += pad[2]
        points[:, 1] += pad[0]
        points = torch.from_numpy(points[:, :2]).float()
        if self.labels[index] is not None:
            labels = torch.tensor(self.labels[index].copy())
        else:
            name = self._get_basename(self.img_paths[index])
            label_path = os.path.join(self.labels_dir, name+'.txt')
            labels = self._read_object_label(label_path).astype(np.float32)
            labels = torch.tensor(labels)
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * W_, ratio[1] * H_, padw=pad[4], padh=pad[5])

        crop_corner = None
        if (self._do_homographic_augm or (self.action == 'val' and self.random_crop)) and (self.labels or self.point_paths):
            if self.config["augmentation"]["homographic"].get('flipping') is not None and self.action != 'val':
                flipping = self.config["augmentation"]["homographic"].get('flipping')
                image, points, labels = flip(image, points, labels, **flipping)
            if self.random_crop is not None and not mosaic_piece:
                crop_corner = self.random_crop(image.copy(), points, labels)

        ### augmentation part here ###
        if self.config["augmentation"]["photometric"][f"enable_{self.action}"]:
            ### photometric augmentation ###
            # image.shape = HWC
            aug_light = copy.deepcopy(self.config['augmentation'])
            aug_light['photometric']['params'] = aug_light['photometric']['params_light']
            img_warped_pair = imgPhotometric(image, **aug_light)
            image = imgPhotometric(img_warped_pair, **self.config['augmentation'])
        else:
            img_warped_pair = image.copy()
        points_warped_pair = torch.clone(points)

        if (self._do_homographic_augm or self._do_only_crop) and not mosaic_piece:   # not implemented for mosaic pieces
            if self.cropHW:
                H, W = self.cropHW
            warped_image, labels_2D, valid_mask, points, homography, labels = \
                self._homographic_augmentation(image, points, H, W, pad_=pad, labels=labels, crop_yx=crop_corner, idx=index)
            points = torch.from_numpy(points)
            if self.transform:
                warped_image = self.transform(warped_image)

            # filter candidates
            clip_boxes(labels[:, 1:], (self.img_size, self.img_size))
            i = box_candidates(box1=labels[:, 1:], box2=labels[:, 1:], area_thr=25)
            labels = labels[i]

            # transform labels back to normed before outputting
            labels = self._extend_and_xyxy2xywhn(labels, H=H, W=W, clip=True)
            dont_care, labels = self._rmv_dont_care(labels)

            sample.update({
                "image": warped_image.contiguous(),
                "labels_2D": labels_2D.contiguous(),
                "valid_mask": valid_mask.contiguous(),
                "box_labels": labels,
                "dont_care": dont_care})
            if self.return_points or mosaic_piece:
                sample.update({"points": points})

        else:   # no homographic augmentation
            if self.transform and not mosaic_piece:
                image = self.transform(image).contiguous()

            labels_2D = get_labels(points, H, W)
            sample.update({"labels_2D": labels_2D})

            homography = torch.eye(3)
            valid_mask = compute_valid_mask(torch.tensor([H, W]), homography=homography, pad=pad)

            if not mosaic_piece:
                labels = self._extend_and_xyxy2xywhn(labels, H=H, W=W, clip=True)
                # pop dont_care labels (do only for KITTI!!!)
                dont_care, labels = self._rmv_dont_care(labels)
                sample.update({"dont_care": dont_care})
            else:
                labels_out = labels
                labels_out[:, 1:] = xyxy2xywhn(labels[:, 1:], h=H, w=W, clip=True)

            sample.update({
                "image": image,
                "points": points,
                "valid_mask": valid_mask.contiguous(),
                "box_labels": labels,
                })

        if self.config["warped_pair"]["enable"] and not mosaic_piece:
            # img input must be torch tensor
            sample = self._get_warped_pair(torch.from_numpy(img_warped_pair), points_warped_pair, sample, homography,
                                           H, W, pad, valid_mask=sample["valid_mask"], crop_yx=crop_corner, idx=index)

        return sample

    def _homographic_augmentation(self, image, points, H, W, valid_mask=None, pad_=(0,0,0,0), labels=(), mosaic=False,
                                  crop_yx=None, cropHW=None, idx=None):
        """
        Random homographic augmentation
        :param image: ndarray (H, W, 3)
        :param points: Tensor (N, 2)
        :param H: int (of cropped img)
        :param W: int (of cropped img)
        :param valid_mask: Tensor ? (H, W)
        :param pad_: tuple (top, bottom, left, right, tb, lr)
        :param labels: Tensor (M, 5) as xyxy
        :param crop: (h_crop, w_crop) top left image coordinate of crop
        :return: warped image, valid_mask, points, homography describing the warp, labels
        """
        if self._do_only_crop:
            config_hom_aug = {'translation': False, 'rotation': False, 'scaling': False, 'perspective': False}
        elif not mosaic:
            config_hom_aug = self.config["augmentation"]["homographic"]["params"]
        else:
            config_hom_aug = self.config["augmentation"]["homographic"]["mosaic"]["params"]
        # information for crop:, tl_corner
        cropHW = self.cropHW or cropHW
        if crop_yx is not None:
            crop = {"tl": crop_yx, "H_crop": cropHW[0], "W_crop": cropHW[1], "HW_full": image.shape[:2]}
        else:
            crop = None
        seed = idx if self.action == 'val' else None
        homography = sample_homography(np.array([2, 2]), shift=-1, crop_pts=crop, seed=seed, **config_hom_aug)

        if crop_yx:
            homography, homography_big = homography[:]
            homography_big = torch.tensor(homography_big).float()
            # inv_homography_big = homography_big.inverse()

        homography = torch.tensor(homography).float()
        inv_homography = homography.inverse()

        image = torch.from_numpy(image)  # image.shape = HWC
        # warp small image or (large image, then crop)
        hom = homography if crop_yx is None else homography_big
        warped_image = warp_image(image, hom, mode="bilinear")
        warped_image = warped_image.numpy()  # image.shape = CHW

        if crop_yx:
            # crop large warped_image (W and H are small image dims); get bottom right corner
            crop_yx_br = torch.tensor(crop_yx) + torch.tensor(cropHW)
            warped_image = warped_image[crop_yx[0]:crop_yx_br[0], crop_yx[1]:crop_yx_br[1]]

        if valid_mask is not None:
            if crop_yx:
                valid_mask = warp_image(valid_mask, homography_big, mode='nearest')
                valid_mask = valid_mask[crop_yx[0]:crop_yx_br[0], crop_yx[1]:crop_yx_br[1]]
            else:
                valid_mask = warp_image(valid_mask, homography, mode='nearest')
        else:
            valid_mask = compute_valid_mask(
                        torch.tensor(image.shape[:2]),   # could be a problem when using crop
                        homography=hom,
                        erosion_radius=self.config["augmentation"]["homographic"]["valid_border_margin"],
                        pad=pad_)
            if crop_yx:
                valid_mask = valid_mask[:, crop_yx[0]:crop_yx_br[0], crop_yx[1]:crop_yx_br[1]]

        if crop_yx and not mosaic:  # TODO: test just crop (no mosaic); might be wrong here
            points[:, 0] -= crop_yx[1]
            points[:, 1] -= crop_yx[0]
        warped_set = warpLabels(points, *warped_image.shape[:2], inv_homography)   # warped_set = warpLabels(points, *image.shape[:2], inv_homography_big)
        labels_2D = warped_set['labels']
        points = warped_set['warped_pnts']

        if mosaic:
            points = torch.from_numpy(points)

        if labels.size():
            if crop_yx and not mosaic:
                labels[:, [1,3]] -= crop_yx[1]
                labels[:, [2,4]] -= crop_yx[0]
            labels[:, 1:3] = warp_points(labels[:, 1:3], homography_scaling(inv_homography, *warped_image.shape[:2]))
            labels[:, 3:5] = warp_points(labels[:, 3:5], homography_scaling(inv_homography, *warped_image.shape[:2]))

            # filter candidates
            labels_old = labels.clone()
            clip_boxes(labels[:, 1:], warped_image.shape[:2])
            i = box_candidates(box1=labels[:, 1:], box2=labels_old[:, 1:], area_thr=25)
            labels = labels[i]
        if crop_yx:
            homography = (homography, homography_big)

        return warped_image, labels_2D, valid_mask, points, homography, labels

    def _get_warped_pair(self, vanilla_img, vanilla_points, sample, homography, H, W, pad=(0,0,0,0), valid_mask=None,
                         mosaic=False, crop_yx=None, cropHW=None, idx=None):

        config = self.config["warped_pair"]["params"] if not mosaic else self.config["warped_pair"]["mosaic_params"]
        if crop_yx is not None:
            cropHW = cropHW or self.cropHW
            crop = {"tl": crop_yx, "H_crop": cropHW[0], "W_crop": cropHW[1], "HW_full": vanilla_img.shape[:2]}
        else:
            crop = None

        seed = idx if self.action == 'val' else None
        homography_warped = sample_homography(np.array([2, 2]), shift=-1, crop_pts=crop, seed=seed, **config)
        if crop_yx:
            # vanilla_img is big unaugmented image
            homography, homography_big = homography[:]
            homography_warped, homography_warped_big = homography_warped[:]
            homography_warped_big = torch.tensor(homography_warped_big).float()
            homography_big = homography_big @ homography_warped_big # homography after homography augmentation
            # inv_homography_big = homography_big.inverse()

        homography_warped = torch.tensor(homography_warped).float()
        inv_homography_warped = homography_warped.inverse()

        # warp original (cropped) image
        homography = homography @ homography_warped
        inv_homography = homography.inverse()

        hom = homography if crop_yx is None else homography_big
        warped_image = warp_image(vanilla_img, hom, mode="bilinear")

        if crop_yx:
            crop_yx_br = torch.tensor(crop_yx) + torch.tensor(cropHW)
            warped_image = warped_image[crop_yx[0]:crop_yx_br[0], crop_yx[1]:crop_yx_br[1]]

        if self.enable_photo_train and self.action == "train":
            warped_image = imgPhotometric(warped_image, **self.config['warped_pair'])  # numpy array (H, W, 3)
        else:
            warped_image = warped_image.numpy()

        if crop_yx and not mosaic:
            vanilla_points[:, 0] -= crop_yx[1]
            vanilla_points[:, 1] -= crop_yx[0]
        warped_set = warpLabels(vanilla_points, H, W, inv_homography)
        #warped_set = warpLabels(vanilla_points, *vanilla_img.shape[:2], hom.inverse())
        warped_labels_2D = warped_set["labels"]
        # if crop_yx is not None:
        #     warped_labels_2D = warped_labels_2D[:, crop_yx[0]:crop_yx_br[0], crop_yx[1]:crop_yx_br[1]]

        if valid_mask is not None:
            if batch_dim := valid_mask.shape[0] == 1:
                valid_mask = valid_mask.squeeze()
            warped_valid_mask = warp_image(valid_mask, homography_warped, mode='nearest')
            if batch_dim:
                # valid_mask = valid_mask.unsqueeze(0)
                warped_valid_mask = warped_valid_mask.unsqueeze(0)
        else:
            warped_valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                homography=homography,
                erosion_radius=self.config["warped_pair"]["valid_border_margin"],
                pad=pad
            )

        if mosaic:
            self._remove_borders(warped_valid_mask, *warped_valid_mask.shape[:2], bm=4)
            warped_valid_mask = warped_valid_mask.unsqueeze(0)

        if self.transform:
            warped_image = self.transform(warped_image)

        sample.update({
            "homographies": homography_warped,
            "inv_homographies": inv_homography_warped,
            "warped_valid_mask": warped_valid_mask.contiguous(),
            "warped_image": warped_image.contiguous(),
            "warped_labels": warped_labels_2D.contiguous(),
        })

        if self.return_points:
            warped_points = torch.from_numpy(warped_set['warped_pnts']) # always return points as torch tensor
            if crop_yx and not mosaic:
                warped_points[:, 0] -= crop_yx[1]
                warped_points[:, 1] -= crop_yx[0]
            sample.update({"warped_points": warped_points})

        return sample

    def _do_homographic_adaptation(self, image, shape, pad, sample, H0, W0, H, W):
        homoAdapt_iter = self.config['homography_adaptation']['num']
        homographies = np.stack([sample_homography(np.array([2, 2]), shift=-1,
                                                   **self.config['homography_adaptation']['homographies']['params'])
                                 for _ in range(homoAdapt_iter)])

        homographies = np.stack([homography for homography in homographies])
        homographies[0, :, :] = np.identity(3)

        homographies = torch.tensor(homographies, dtype=torch.float32)
        inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)])

        # images
        image = torch.tensor(image, dtype=torch.float32)
        warped_image = warp_image_batch(image.repeat(homoAdapt_iter, 1, 1, 1), homographies, mode='bilinear')
        warped_image = warped_image.squeeze()

        # masks
        # print(pad)
        er = self.config['homography_adaptation']['valid_border_margin']
        valid_mask = compute_valid_mask(image.shape[:2], homography=homographies, erosion_radius=er, pad=pad)

        if len(warped_image.shape) == 3:
            warped_image = warped_image.unsqueeze(-1)
        warped_image = warped_image.transpose(1, 3).transpose(2, 3)

        sample.update({'image': warped_image.contiguous(),
                       'valid_mask': valid_mask.contiguous(),
                       'homographies': homographies,
                       'inv_homographies': inv_homographies,
                       'pad': pad,
                       'dims': ((H0, W0), (H, W))
                       })
        return sample

    @property
    @abstractmethod
    def _letterbox(self):
        pass

    @abstractmethod
    def _rmv_dont_care(self, labels):
        pass

    @abstractmethod
    def _mosaic_augmentation(self, idx):
        # implement for individual data classes
        pass

    def load_image(self, i):
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        if self.imgs[i] is None:
            path = self.img_paths[i]
            im = cv2.imread(path).astype(np.float32) / 255.  # BGR
            assert im is not None, f'Image Not Found {path}'
        else:
            im = self.imgs[i].astype(np.float32) / 255.
        h0, w0 = im.shape[:2]  # orig hw
        if self.cropHW is None: # rescale and crop are currently not implemented
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (round(w0 * r), round(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        elif self.cropHW[0] > h0 or self.cropHW[1] > w0:
            r = max(self.cropHW[0]/h0, self.cropHW[1]/w0)
            im = cv2.resize(im, (round(w0 * r), round(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        if self.inp_ch != 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.expand_dims(im, axis=2)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


    @staticmethod
    def _get_basename(path):
        # returns name of file without extension (e.g. '.../000123.jpg' --> '000123')
        name = os.path.basename(path)
        return os.path.splitext(name)[0]

    @staticmethod
    def _remove_borders(img, h, w, bm=2):
        img[0:bm, :] = 0.
        img[-bm:h, :] = 0.
        img[:, 0:bm] = 0.
        img[:, -bm:w] = 0.
        return img

    @staticmethod
    def _extend_and_xyxy2xywhn(labels, H, W, clip=False):
        nl = len(labels)
        labels_out = torch.zeros(nl, 6).float() if torch.is_tensor(labels) else np.zeros((nl, 6), dtype=np.float32)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=W, h=H, clip=clip, eps=1E-3)
            labels_out[:, 1:] = labels
        return labels_out

    @staticmethod
    def _pop_labels(labels, idx):
        popped = labels[labels[:, 1] == idx]
        labels = labels[labels[:, 1] != idx]
        return popped, labels

    @staticmethod
    def _read_object_label(label_path):
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                text = f.readlines()
            label = []
            for t in text:
                t = t.split('\n')
                if '' in t:
                    t.remove('')
                t = t[0].split(' ')
                for i, item in enumerate(t):
                    t[i] = float(item)

                t = np.array(t, dtype=np.float16)   # save in half precision to save disk space
                label.append(t)
            if len(label):
                # label file is not empty
                label = np.array(label)
            else:
                label = np.empty((0, 6), dtype=np.float16)
        else:
            label = np.empty((0, 6), dtype=np.float16)
        return label

    def _cache_object_labels(self, ret_labels=False):
        all_labels = []
        for im_path in tqdm(self.img_paths, desc='Creating object labels cache...'):
            name = self._get_basename(im_path)
            label_path = os.path.join(self.labels_dir, name+'.txt')
            label = self._read_object_label(label_path)
            all_labels.append(label)

        with open(os.path.join(self.labels_dir, 'cache'), 'wb') as f:
            pickle.dump(all_labels, f)

        if ret_labels:
            all_labels = [lbl.astype(np.float32) for lbl in all_labels]
            return all_labels


class Kitti(DataBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _letterbox(self):
        return False

    def _rmv_dont_care(self, labels):
        # removes "don't care" regions
        return self._pop_labels(labels, 20)

    def _mosaic_augmentation(self, idx):
        mosaic_pieces = []
        indices = [idx]
        indices += random.choices(self.indices, k=3)
        random.shuffle(indices)
        labels4 = []
        for index in indices:
            mosaic_pieces.append(self._get_single_sample(index, mosaic_piece=True))
        sy, sx = mosaic_pieces[0]['image'].shape[:2]
        xc, yc = sx, sy
        for i, mosaic_piece in enumerate(mosaic_pieces):
            img = (mosaic_piece['image'] * 255.).astype(np.uint8)
            valid_mask = squeezeToNumpy(mosaic_piece['valid_mask'])

            (h, w) = img.shape[:2]
            valid_mask = self._remove_borders(valid_mask, h, w, bm=2)
            if i == 0:  # top left
                img4 = np.full((sy * 2, sx * 2, img.shape[2]), dtype=np.uint8,
                               fill_value=(114, 114, 114))  # base image with 4 tiles
                valid_mask4 = np.zeros((sy * 2, sx * 2), dtype=np.float32)
                sample = {"name": mosaic_piece['name'], "index": mosaic_piece['index'], "HW": mosaic_piece['HW']}
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                         0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                        y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, sx * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(sy * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, sx * 2), min(sy * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            valid_mask4[y1a:y2a, x1a:x2a] = valid_mask[y1b:y2b, x1b:x2b]
            # remove borders
            valid_mask4 = self._remove_borders(valid_mask4, 2 * sy, 2 * sx, bm=2)

            padw = x1a - x1b
            padh = y1a - y1b

            # Points
            points = mosaic_piece['points']
            if points.shape[0] > 0:
                points[:, 0] += padw
                points[:, 1] += padh

            # Labels
            labels = mosaic_piece['box_labels'].numpy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format

            labels4.extend(labels)

        top = sy // 2
        bot = int(sy * 1.5)
        left = sx // 2
        right = int(sx * 1.5)

        # create new mosaic sample
        mosaic_points = np.concatenate([x['points'] for x in mosaic_pieces])
        padx = left
        pady = top
        if mosaic_points.size:
            mosaic_points[:, 0] -= padx
            mosaic_points[:, 1] -= pady
        labels4 = np.array(labels4)
        if labels4.size:
            labels4[:, [1, 3]] -= padx
            labels4[:, [2, 4]] -= pady
        else:
            labels4 = np.empty((0, 6), dtype=np.float32)

        mosaic_points = torch.from_numpy(mosaic_points)

        valid_mask4 = torch.from_numpy(valid_mask4)

        img4 = img4.astype(np.float32) / 255
        vanilla_img = img4.copy()
        vanilla_pts = torch.clone(mosaic_points)
        vanilla_mask = torch.clone(valid_mask4)

        if self._do_homographic_augm and self.config['augmentation']['homographic']['mosaic']['enable']:
            labels4 = torch.from_numpy(labels4)

            img4, labels_2D, valid_mask4, mosaic_points, homographies, labels4 = \
                self._homographic_augmentation(img4, mosaic_points, sy, sx, labels=labels4,
                                               valid_mask=valid_mask4, mosaic=True, crop_yx=[pady, padx], idx=idx)

            homography, homography_big = homographies[:]
            homography = (homography, homography_big)
        else:
            print("Doing mosaic augmentation without homographic augmentation is deprecated!")
            homography = torch.eye(3)
            labels4 = torch.from_numpy(labels4)
            labels_2D = get_labels(mosaic_points, sy, sx)
            img4 = img4.astype(np.float32) / 255.

        sample = self._get_warped_pair(torch.from_numpy(vanilla_img), vanilla_pts, sample, homography, sy, sx,
                                       valid_mask=vanilla_mask, mosaic=True, crop_yx=[pady, padx], roi=(top, bot, left, right))

        labels4 = self._extend_and_xyxy2xywhn(labels4, sy, sx, clip=True)
        dont_care, labels4 = self._rmv_dont_care(labels4)

        valid_mask4 = valid_mask4.unsqueeze(0)

        if self.transform:
            img4 = self.transform(img4)
        sample.update({'image': img4.contiguous(),
                       'valid_mask': valid_mask4.contiguous(),
                       'box_labels': labels4,
                       'dont_care': dont_care,
                       'labels_2D': labels_2D.contiguous(),
                       'shapes': None})
        if self.return_points:
            sample.update({'points': mosaic_points})
        return sample


class Coco(DataBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _letterbox(self):
        return True

    def _rmv_dont_care(self, labels):
        return torch.empty((0, 6)).float(), labels

    # deprecated
    def _mosaic_augmentation(self, idx):
        mosaic_pieces = []
        indices = [idx]
        s = self.img_size
        indices += random.choices(self.indices, k=3)
        yc, xc = s, s  # mosaic center
        random.shuffle(indices)
        labels4 = []
        for index in indices:
            mosaic_pieces.append(self._get_single_sample(index, mosaic_piece=True))
        # get edges of mosaic to minimize boundaries
        left = []
        right = []
        top = []
        bot = []
        for i, mosaic_piece in enumerate(mosaic_pieces):
            img = (mosaic_piece['image'] * 255.).astype(np.uint8)
            valid_mask = squeezeToNumpy(mosaic_piece['valid_mask'])

            (h, w) = img.shape[:2]
            valid_mask = self._remove_borders(valid_mask, h, w, bm=2)
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), dtype=np.uint8,
                               fill_value=(114, 114, 114))  # base image with 4 tiles
                valid_mask4 = np.zeros((s * 2, s * 2), dtype=np.float32)
                name = self._get_basename(self.img_paths[idx])
                sample = {"name": name}
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                         0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                left.append(x1a)
                top.append(y1a)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                        y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                right.append(x2a)
                top.append(y1a)
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bot left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                left.append(x1a)
                bot.append(y2a)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bot right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                right.append(x2a)
                bot.append(y2a)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            valid_mask4[y1a:y2a, x1a:x2a] = valid_mask[y1b:y2b, x1b:x2b]
            # remove borders        # I think this should go further down!
            valid_mask4 = self._remove_borders(valid_mask4, 2 * s, 2 * s, bm=2)

            padw = x1a - x1b
            padh = y1a - y1b

            # Points
            points = mosaic_piece['points']
            if points.shape[0] > 0:
                points[:, 0] += padw
                points[:, 1] += padh

            # Labels
            labels = mosaic_piece['box_labels'].numpy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format

            labels4.extend(labels)

        right = min(right)
        left = max(left)
        top = max(top)
        bot = min(bot)

        # move center of largest full area to center
        transx = s - (right + left) // 2
        transy = s - (bot + top) // 2
        translation_matrix = np.float32([[1, 0, transx], [0, 1, transy]])
        img4 = cv2.warpAffine(img4, translation_matrix, img4.shape[:2],
                              borderValue=(114, 114, 114))  # alternatively choose a larger mosaic size
        valid_mask4 = cv2.warpAffine(valid_mask4, translation_matrix, img4.shape[:2])
        edges = [s//2, int(1.5 * s)]*2

        # create new mosaic sample
        mosaic_points = np.concatenate([x['points'] for x in mosaic_pieces])
        padx = transx - edges[0]
        pady = transy - edges[0]
        if mosaic_points.size:
            mosaic_points[:, 0] += padx
            mosaic_points[:, 1] += pady

        labels4 = np.array(labels4)
        if labels4.size:
            labels4[:, [1, 3]] += padx
            labels4[:, [2, 4]] += pady
        else:
            labels4 = np.empty((0, 6), dtype=np.float32)

        mosaic_points = torch.from_numpy(mosaic_points)

        valid_mask4 = torch.from_numpy(valid_mask4)

        img4 = img4.astype(np.float32) / 255.
        vanilla_img = img4.copy()
        vanilla_pts = torch.clone(mosaic_points)
        # vanilla_mask = torch.clone(valid_mask4)

        labels4 = torch.from_numpy(labels4)
        img4, labels_2D, valid_mask4, mosaic_points, homography, labels4 = \
            self._homographic_augmentation(img4, mosaic_points, s, s, labels=labels4,
                                           valid_mask=valid_mask4, mosaic=True, crop_yx=(s//2, s//2), cropHW=(s,s))

        sample = self._get_warped_pair(torch.from_numpy(vanilla_img), vanilla_pts, sample, homography, s, s,
                                       valid_mask=valid_mask4, mosaic=True, crop_yx=(s//2, s//2), cropHW=(s,s))

        labels4 = self._extend_and_xyxy2xywhn(labels4, s, s, clip=True)

        # remove borders
        valid_mask4 = self._remove_borders(valid_mask4, 2 * s, 2 * s, bm=4)

        valid_mask4 = valid_mask4.unsqueeze(0)

        if self.transform:
            img4 = self.transform(img4)
        sample.update({'image': img4.contiguous(),
                       'valid_mask': valid_mask4.contiguous(),
                       'box_labels': labels4,
                       'labels_2D': labels_2D.contiguous(),
                       'shapes': None})
        if self.return_points:
            sample.update({'points': mosaic_points})

        return sample


class Campus(Kitti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _rmv_dont_care(self, labels):
        return torch.empty((0, 6)).float(), labels


class HPatches(Dataset):

    def __init__(self, transform=None, **config):
        self.config = config
        self.files = self._init_dataset(**self.config)
        sequence_set = []
        for (img, img_warped, mat_hom) in zip(self.files['image_paths'], self.files['warped_image_paths'], self.files['homography']):
            sample = {'image': img, 'warped_image': img_warped, 'homography': mat_hom}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform

        # self.sizer = np.array(config['resize'])   # 640

    def __getitem__(self, index):
        """
        :param index:
        :return:
            image:
                tensor (3,H,W)
            warped_image:
                tensor (3,H,W)
        """
        def _read_image(path, rgb):
            input_image = cv2.imread(path)
            if not rgb:
                input_image = np.expand_dims(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), -1)
            return input_image

        def _preprocess(image):

            image = tf.convert_to_tensor(image, dtype=tf.float32)
            # tf.Tensor.set_shape(image, [None, None, 3])
            # image = tf.image.rgb_to_grayscale(image)

            image = ratio_preserving_resize(image)
            return tf.cast(image, tf.float32)

        def ratio_preserving_resize(image):
            target_size = tf.convert_to_tensor([480, 640])
            scales = tf.cast(tf.divide(target_size, tf.shape(image)[:2]), tf.float32)
            new_size = tf.cast(tf.shape(image)[:2], tf.float32) * tf.reduce_max(scales)
            image = tf.image.resize(image, tf.cast(new_size, tf.int32), method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])

        def _adapt_homography_to_preprocessing(zip_data):
            H = tf.cast(zip_data['homography'], tf.float32)
            source_size = tf.cast(zip_data['shape'], tf.float32)
            source_warped_size = tf.cast(zip_data['warped_shape'], tf.float32)
            target_size = tf.cast(tf.convert_to_tensor([480, 640]), tf.float32)

            # Compute the scaling ratio due to the resizing for both images
            s = tf.reduce_max(tf.divide(target_size, source_size))
            up_scale = tf.linalg.diag(tf.stack([1. / s, 1. / s, tf.constant(1.)]))
            warped_s = tf.reduce_max(tf.divide(target_size, source_warped_size))
            down_scale = tf.linalg.diag(tf.stack([warped_s, warped_s, tf.constant(1.)]))

            # Compute the translation due to the crop for both images
            pad_y = tf.cast(((source_size[0] * s - target_size[0]) / tf.constant(2.0)), tf.int32)
            pad_x = tf.cast(((source_size[1] * s - target_size[1]) / tf.constant(2.0)), tf.int32)
            translation = tf.stack([tf.constant(1), tf.constant(0), pad_x,
                                    tf.constant(0), tf.constant(1), pad_y,
                                    tf.constant(0),tf.constant(0), tf.constant(1)])
            translation = tf.cast(tf.reshape(translation, [3,3]), tf.float32)
            pad_y = tf.cast(((source_warped_size[0] * warped_s - target_size[0])
                                 / tf.constant(2.0)), tf.int32)
            pad_x = tf.cast(((source_warped_size[1] * warped_s - target_size[1])
                                 / tf.constant(2.0)), tf.int32)
            warped_translation = tf.stack([tf.constant(1), tf.constant(0), -pad_x,
                                           tf.constant(0), tf.constant(1), -pad_y,
                                           tf.constant(0),tf.constant(0), tf.constant(1)])
            warped_translation = tf.cast(tf.reshape(warped_translation, [3,3]), tf.float32)

            H = warped_translation @ down_scale @ H @ up_scale @ translation
            return H

        sample = self.samples[index]
        ch = self.config['input_channels']
        assert ch in {1, 3}
        rgb = ch == 3
        image_original = _read_image(sample['image'], rgb)
        warped_image_original = _read_image(sample['warped_image'], rgb)

        image = _preprocess(image_original)
        warped_image = _preprocess(warped_image_original)
        to_numpy = False
        if to_numpy:
            image, warped_image = np.array(image), np.array(warped_image)
        homographies = {'homography': sample['homography'],
                                            'shape': image_original.shape[:2],
                                            'warped_shape': warped_image_original.shape[:2]}

        if image_original.shape != warped_image_original.shape:
            print("stop!")

        homography = _adapt_homography_to_preprocessing(homographies)

        sample = {'image': np.array(image)/255., 'warped_image': np.array(warped_image)/255., 'homography': np.array(homography)}
        sample['image'] = sample['image'].transpose(2,0,1)
        sample['warped_image'] = sample['warped_image'].transpose(2,0,1)
        return sample

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        base_path = Path('datasets', 'hpatches')
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            num_images = 5
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + '.ppm')))
                warped_image_paths.append(str(Path(path, str(i) + '.ppm')))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies}
        return files
