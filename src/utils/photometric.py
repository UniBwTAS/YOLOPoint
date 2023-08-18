""" photometric augmentation
# used in dataloader
"""

from imgaug import augmenters as iaa
import numpy as np
import cv2
from random import randrange

class ImgAugTransform:
    def __init__(self, **config):

        if config['photometric']['enable']:
            params = config['photometric']['params']
            aug_all = []

            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Add((-change, change))
                aug_all.append(aug)

            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.LinearContrast((change[0], change[1]))
                aug_all.append(aug)

            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)

            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                aug = iaa.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)

            if change := params.get('add_elementwise', False):
                aug = iaa.AddElementwise(**change)
                aug_all.append(aug)

            if change := params.get('add', False):
                aug = iaa.Sometimes(0.5, iaa.Add(**change))
                aug_all.append(aug)

            if change := params.get('channel_shuffle', False):
                aug = iaa.ChannelShuffle(change)
                aug_all.append(aug)

            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                if change > 3:
                    change = randrange(3, change, step=2)
                aug = iaa.Sometimes(0.5, iaa.MotionBlur(change))
                aug_all.append(aug)

            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.GaussianBlur(sigma=change)
                aug_all.append(aug)

            if hsv := params.get('hsv'):
                (h, s, v) = hsv
                aug_hs = iaa.MultiplyHueAndSaturation(mul_hue=(1-h, 1+h), mul_saturation=(1-s, 1+s))
                aug_v = iaa.MultiplyBrightness((1-v, 1+v))
                aug_all += [aug_hs, aug_v]

            self.aug = iaa.Sequential(aug_all)

        else:
            self.aug = iaa.Sequential([iaa.Identity()])

    def __call__(self, img):
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = img.astype(np.float32) / 255
        return img


class CustomizedTransform:
    def __init__(self):
        pass

    @staticmethod
    def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                       kernel_size_range=[250, 350]):
        min_dim = min(image.shape[:2]) / 4
        mask = np.zeros(image.shape[:2], np.uint8)
        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, image.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)
        transparency = np.random.uniform(*transparency_range)
        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = image * (1 - transparency * mask[..., np.newaxis] / 255.)
        shaded = np.clip(shaded, 0, 255)

        return shaded

    def __call__(self, img, **config):
        if shade := config['photometric']['params'].get('additive_shade'):
            img = self.additive_shade(img * 255, **shade) / 255
        return img

def imgPhotometric(img, **config_aug):
    # Performs photometric augmentations
    augmentation = ImgAugTransform(**config_aug)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    img = augmentation(img)
    cusAug = CustomizedTransform()
    img = cusAug(img, **config_aug)
    return img