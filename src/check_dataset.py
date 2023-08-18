"""
Some functions to help do checks on data integrity
"""

from utils.loader import dataLoader
from utils.utils import dict_update
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from utils.utils import warp_image_batch, squeezeToNumpy, toNumpy, img_overlap
from utils.general_yolo import xywhn2xyxy
import os


def add_labels(im, l_2D):
    im = np.array(im*255.).astype(np.uint8)
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    l_2D = l_2D[:,:,np.newaxis]
    l_2D = l_2D.astype(np.uint8)
    blank = np.zeros_like(l_2D)
    l_2D = np.concatenate((blank, l_2D, blank), axis=2)*255
    im = cv2.add(l_2D, im)
    im = cv2.resize(im, (640, 480))
    return im

# def ch2front(rgb_img):
#     # HWC --> CWH
#     k = 1 if len(rgb_img.shape) == 4 else 0
#     return rgb_img.transpose(0+k,2+k).transpose(1+k,2+k)

def ch2back(rgb_img):
    # CWH --> HWC
    return rgb_img.transpose(1,2,0)

def visualize_images(loader, names):

    def plot_points(im, pts, name):
        img_new = im
        pts = np.round(pts[:]).astype(int)
        for p in pts:
            cv2.circle(img_new, p, 2, (255, 0, 0), -1)
        cv2.imshow(name, img_new)

    for i, sample in enumerate(loader):
        print(i)
        img = np.ascontiguousarray((np.squeeze(sample['image'].numpy())*255.).astype(np.uint8).transpose((1,2,0)))
        img_warp = np.ascontiguousarray((np.squeeze(sample['warped_image'].numpy())*255.).astype(np.uint8).transpose((1,2,0)))

        # cv2.imshow("image_raw", img)
        # cv2.imshow("image_raw_warp", img_warp)

        mask = (np.squeeze(sample['valid_mask'].numpy())*255.).astype(np.uint8)
        mask_warp = (np.squeeze(sample['warped_valid_mask'].numpy()) * 255.).astype(np.uint8)

        points = toNumpy(sample['points'][0])
        points_warp = toNumpy(np.squeeze(sample['warped_points'][0]))
        if (points < 0).any():
            print("negative!")

        bboxes = sample['box_labels']
        # print(bboxes)
        clss = bboxes[:, 1].numpy()
        bboxes = bboxes[:, 2:].numpy()

        # bboxes = np.rint(xywhn2xyxy(bboxes, *HW[:2][::-1], *HW[2:])).astype(int)
        # bboxes = np.rint(xywhn2xyxy(bboxes, mask.shape[1], mask.shape[0], 0,0)).astype(int)
        # for bbox, cls in zip(bboxes, clss):
        #     cls = names[int(cls)]
        #     orig = bbox[:2]+np.array((0, 10))
        #     img = cv2.putText(img, cls, orig, cv2.FONT_HERSHEY_SIMPLEX,
        #                         fontScale=0.4, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        #     img = cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 1)

        print(len(points))

        plot_points(img, points, 'image')
        plot_points(img_warp, points_warp, 'warp')
        # cv2.imshow("img_test", img)
        cv2.imshow('mask', mask)
        cv2.imshow('mask_warp', mask_warp)

        cv2.waitKey(0)


def distribution_heatmap(loader, dims=(240, 320), normalize=False, vis_at_runtime=False):
    canvas = np.zeros(dims)
    for i, sample in enumerate(tqdm(loader)):
        labels_2D = np.squeeze(sample["labels_2D"].detach().numpy(), axis=(0, 1))
        canvas += labels_2D
        if vis_at_runtime and i % 1 == 0:
            img = (canvas/np.max(canvas)*255.).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
            cv2.imshow(f"distribution", img)
            cv2.waitKey(1)
    canvas /= len(loader)
    if normalize:
        canvas /= max(canvas)
    plt.imshow(canvas, cmap='hot', interpolation='nearest')
    plt.title("Point Distribution")
    plt.show()

def distribution_numbers(loader):
    points_list = []
    for i, sample in enumerate(tqdm(loader)):
        labels_2D = np.squeeze(sample["labels_2D"].detach().numpy(), axis=(0, 1))
        points_num = len(np.where(labels_2D == 1)[0])
        pts = {"num": points_num, "name": sample["name"]}
        points_list.append(pts)
    num_list = np.array([x["num"] for x in points_list])
    bad_pts = [x["name"] for x in points_list if x["num"] <= 5 or 500 <= x["num"]]
    print(f"max: {np.max(num_list)}, min: {np.min(num_list)}\n"
          f"sigma: {np.std(num_list)}, mean: {np.mean(num_list)}")
    for bad_pt in bad_pts:
        print(bad_pt)
    fig, axs = plt.subplots(2)
    # plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    axs[0].hist(num_list, bins=50)
    axs[1].boxplot(num_list)
    # plt.gca().set(title='Point Number Distribution', ylabel='Number')
    # plt.boxplot(points_list)
    plt.show()

def check_homographies(loader):

    for i, sample in enumerate(tqdm(loader)):

        homography, inv_homography = sample['homographies'], sample['inv_homographies']
        unwarped_image = warp_image_batch(sample['warped_image'], inv_homography)
        # unwarped_image_np = squeezeToNumpy(unwarped_image)
        image = sample['image']
        # valid_mask = sample['valid_mask']
        warped_image = sample['warped_image']
        warped_valid_mask = sample['warped_valid_mask']

        unwarped_image_np = (squeezeToNumpy(unwarped_image) * 255.).astype(np.uint8).transpose((1, 2, 0))
        image_np = (squeezeToNumpy(image) * 255.).astype(np.uint8).transpose((1,2,0))
        warped_image_np = (squeezeToNumpy(warped_image) * 255.).astype(np.uint8).transpose((1,2,0))
        # valid_mask_np = (squeezeToNumpy(valid_mask) * 255.).astype(np.uint8)
        warped_labels_2D = cv2.cvtColor((squeezeToNumpy(sample['warped_labels']) * 255.).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        labels_2D = cv2.cvtColor((squeezeToNumpy(sample['labels_2D']) * 255.).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        result_overlap = cv2.add(warped_image_np, warped_labels_2D)
        result_overlap_2D = cv2.add(image_np, labels_2D)

        cv2.imshow("image", image_np)
        cv2.imshow("unwarped image", unwarped_image_np)
        cv2.imshow("mask", result_overlap_2D)
        cv2.imshow("labels", warped_labels_2D)
        cv2.imshow("original", warped_image_np)
        cv2.imshow('ovelrap', result_overlap)
        cv2.waitKey(0)

if __name__ == "__main__":
    # with open("configs/superpoint_coco_train_YOLO.yaml", 'r') as f:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_directory, '..'))
    with open("configs/coco.yaml", 'r') as f:
        config = yaml.safe_load(f)
    # dataset = config['data']['dataset']
    bs = {"training_params": {
        "train_batch_size": 1,
        "val_batch_size": 1,
        "workers_train": 0,
        "workers_val": 0}}
    config = dict_update(config, bs)
    print(config)
    loader = dataLoader(config, DEBUG=False, return_points=True, action='train')

    # device = "gpu"

    # distribution_numbers(train_loader)
    # distribution_heatmap(train_loader, vis_at_runtime=True)
    names = config['names']
    visualize_images(loader, names)
    # check_homographies(loader)
    # check_augmentation(val_loader)
