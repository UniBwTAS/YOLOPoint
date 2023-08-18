import torch
import cv2
import torch.nn.functional as F
import numpy as np
import collections
from importlib import import_module
quan = lambda x: x.round().long()
import math
from utils.general_yolo import LOGGER

def pop_class(box_targets, cls):
    """
    Pops object label class by class number
    :param box_targets: tensor [n, cls_num, x, y, h, w]
    :param cls: int class to pop
    :return: labels and labels of popped class
    """
    popped_cls = box_targets[box_targets[:, 1] == cls]
    box_targets = box_targets[box_targets[:, 1] != cls]
    return box_targets, popped_cls

def parse_str_slice(layers_slice):
    layers_slice = layers_slice.replace(' ', '').split(',')
    layers_slice = [x.split('-') for x in layers_slice]
    layers_slice = [[int(x[0])] if len(x) == 1 else list(range(*[int(x[0]), int(x[1])+1])) for x in layers_slice]
    layers_slice = [item for sublist in layers_slice for item in sublist]
    return layers_slice

def filter_invalid_pts(points, mask):
    """
    Filter points outside valid mask
    :param points: ndarray [3, N] ([2, N] is also ok); xy
    :param mask: ndarray [H, W]
    :return: filtered points [3, M] ; M <= N
    """
    coors = points[:2].astype(int).transpose()
    keep = mask[coors[:, 1], coors[:, 0]] == 1
    points = points.transpose()[keep].transpose()
    return points

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def load_model(meta_model=True, **kwargs):
    model_name = 'Model' if meta_model else kwargs.pop('model_name')
    return getattr(import_module('models'), model_name)(**kwargs)

def data_size(data_loader, config, tag):
    LOGGER.info('== %s split size %d in %d batches of size %d' % \
                 (tag, len(data_loader) * config['training_params'][f'{tag}_batch_size'], len(data_loader),
                  config['training_params'][f'{tag}_batch_size']))

def load_pretrained_yolo(yolo_statedict, yolopoint_statedict, verbose=False):
    # inserts pretrained yolo weights into the full model
    for keys in zip(yolo_statedict, yolopoint_statedict):
        layer_yolo = '.'.join(keys[0].split('.')[-2:])
        layer_yp = '.'.join(keys[1].split('.')[-2:])

        if layer_yolo == layer_yp and yolo_statedict[keys[0]].shape == yolopoint_statedict[keys[1]].shape:
            if verbose:
                LOGGER.info(keys[0], ' '*(45 - len(keys[0])), keys[1])
            yolopoint_statedict[keys[1]] = yolo_statedict[keys[0]]
        else:
            return yolopoint_statedict

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def getPtsFromSemi(semi, conf_thresh=0.015, nms_dist=4):
    # Wrapper for get PtsFromHeatmap
    # semi: torch tensor [ch, Hc, Wc] (raw model output; no batch)
    semi_flat_tensor = flattenDetection(semi).detach()
    semi_flat = toNumpy(semi_flat_tensor)
    semi_thd = np.squeeze(semi_flat)
    pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
    return pts_nms

def getMasks(mask_2D, device, cell_size=8):
    """
    # 2D mask is constructed into 3D (Hc, Wc) space for training
    :param mask_2D:
        tensor [batch, 1, H, W]
    :param cell_size:
        8 (default)
    :param device:
    :return:
        flattened 3D mask for training
    """
    mask_3D = labels2Dto3D(mask_2D.to(device), cell_size=cell_size, add_dustbin=False).float()
    mask_3D_flattened = torch.prod(mask_3D, 1)
    return mask_3D_flattened

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]

    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def labels2Dto3D(labels, cell_size=8, add_dustbin=True):
    """
    Change the shape of labels into 3D. Batch of labels.
    Interest point decoder first part
    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    """
    batch_size, channel, H, W = labels.shape
    pu = torch.nn.PixelUnshuffle(cell_size)
    labels = pu(labels)
    Hc, Wc = H // cell_size, W // cell_size

    if add_dustbin: # for "no interest point" cells
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.] = 0
        labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        ## normalize
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
    return labels

def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def add_labels(im, l_2D):
    # alternative to img_overlap
    im = np.array(im*255.).astype(np.uint8)
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    l_2D = l_2D[:,:,np.newaxis]
    l_2D = l_2D.astype(np.uint8)
    blank = np.zeros_like(l_2D)
    l_2D = np.concatenate((blank, l_2D, blank), axis=2)*255
    im = cv2.add(l_2D, im)
    return im

def flattenDetection(semi, cell_size=8):
    """
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        2D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, H, W)
    """
    batch = True if len(semi.shape) == 4 else False

    if batch:
        dense = torch.nn.functional.softmax(semi, dim=1) # [batch, 65, Hc, Wc]
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]
    else:
        dense = torch.nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)

    # Reshape to get full resolution heatmap.
    ps = torch.nn.PixelShuffle(cell_size)
    heatmap = ps(nodust)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap

def filter_points(points, shape, return_mask=False):
    # Filters out points that are outside of image boundaries; expects shape to be W, H
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points[mask]

def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.
    :param points: torch Tensor (N, 2) i.e. no batch
    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    warped_points = homographies@points.transpose(0,1)
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

def homography_scaling(homography, H, W, device='cpu'):
    trans = torch.tensor([[2./W, 0., -1.], [0., 2./H, -1.], [0., 0., 1.]], dtype=torch.float32, device=device)
    homography = (trans.inverse() @ homography @ trans)
    return homography

def compute_valid_mask(image_shape, homography, erosion_radius=0, pad=(0, 0, 0, 0), ret_corners=False):
    """
    Compute a boolean mask of the valid pixels resulting from a homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.
    pad: Tuple (top, bottom, left, right), letterbox areas
    image_shape: Tensor, shape with letterbox
    """
    if homography.dim() == 2:
        homography = homography.view(-1, 3, 3)
    batch_size = homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0] - pad[0] - pad[1], image_shape[1] - pad[2] - pad[3])
    mask = F.pad(mask, (pad[2], pad[3], pad[0], pad[1]), 'constant', 0)
    mask = warp_image_batch(mask, homography, mode='nearest')

    # pad inside
    b = 1
    mask[:, :, :b, :] = 0  # Top edge
    mask[:, :, -b:, :] = 0  # Bottom edge
    mask[:, :, :, :b] = 0  # Left edge
    mask[:, :, :, -b:] = 0  # Right edge

    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    if erosion_radius > 0:
        mask = mask.cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)
        mask = torch.tensor(mask)
    if ret_corners:
        # return the coordinates of corners for sparse loss function later
        (y_tl, y_br, x_tl, x_br) = pad
        corners = torch.tensor([[x_tl, y_tl], [x_br, y_br]])
        return mask, corners
    return mask

def warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear', padding_mode='zeros'):
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W] or
        tensor [C, H, W] if RGB
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    """
    # compute inverse warped points
    transposed = False
    if img.shape[-1] == 3 and len(img.shape) == 3:
        # RGB image
        img = img.unsqueeze(0) # img.shape = [1, H, W, C]
    elif transposed := (img.shape[-1] == 3 and len(img.shape) == 4):
        img = img.transpose(1,3).transpose(2,3)
    elif len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])

    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    batch, channel, H, W = img.shape

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij'), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_image = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True, padding_mode=padding_mode)
    if transposed:
        warped_image = warped_image.transpose(1,3).transpose(1,2)
    return warped_image

def warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
    """
    Inverse warp image

    :param img:
        tensor [H, W] or [H, W, 3]
    :param mat_homo_inv:
        homography matrix
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        warped image [H, W]
    """
    if rearrange_dims := (img.shape[-1] == 3 and len(img.shape) == 3):
        # RGB image
        img = img.transpose(2, 0).transpose(1, 2).unsqueeze(0) # img.shape = [1, C, H, W]
    warped_image = warp_image_batch(img, mat_homo_inv, device, mode)
    if rearrange_dims:
        warped_image = warped_image.transpose(1, 3).transpose(1, 2)
    return warped_image.squeeze()

def extrapolate_points(pnts):
    pnts_int = pnts.long().type(torch.FloatTensor)
    pnts_x, pnts_y = pnts_int[:, 0], pnts_int[:, 1]

    stack_1 = lambda x, y: torch.stack((x, y), dim=1)
    pnts_ext = torch.cat((pnts_int, stack_1(pnts_x, pnts_y + 1),
                          stack_1(pnts_x + 1, pnts_y), pnts_int + 1), dim=0)

    pnts_res = pnts - pnts_int  # (x, y)
    x_res, y_res = pnts_res[:, 0], pnts_res[:, 1]  # residuals
    res_ext = torch.cat(((1 - x_res) * (1 - y_res), (1 - x_res) * y_res,
                         x_res * (1 - y_res), x_res * y_res), dim=0)
    return pnts_ext, res_ext

def scatter_points(warped_pnts, H, W, res_ext=1):
    # put warped labels on empty canvas
    warped_labels = torch.zeros(H, W)
    warped_labels[quan(warped_pnts)[:, 1], quan(warped_pnts)[:, 0]] = res_ext
    warped_labels = warped_labels.view(-1, H, W)
    return warped_labels

def get_labels(pnts, H, W):
    # Returns points on 2D matrix (1 is point, 0 is no point)
    labels = torch.zeros(H, W)
    pnts_int = torch.min(pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long())
    labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
    labels = labels.unsqueeze(0)
    return labels

def warpLabels(pnts, H, W, inv_homography):
    # must use the inverse homography to warp points according to original homography
    if isinstance(pnts, torch.Tensor):
        pnts = pnts.long()
    else:
        pnts = torch.tensor(pnts).long()
    warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                              homography_scaling(inv_homography, H, W))  # check the (x, y)

    warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))
    warped_labels = scatter_points(warped_pnts, H, W, res_ext=1)

    return {'labels': warped_labels, 'warped_pnts': toNumpy(warped_pnts)}

def np_to_tensor(img, H, W):
    img = torch.tensor(img).type(torch.FloatTensor).view(-1, H, W)
    return img

def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

def tensorto4d(inp):
    if len(inp.shape) == 2:
        inp = inp.view(1, 1, inp.shape[0], inp.shape[1])
    elif len(inp.shape) == 3:
        inp = inp.view(1, inp.shape[0], inp.shape[1], inp.shape[2])
    return inp

def squeezeToNumpy(tensor_arr):
    return tensor_arr.detach().cpu().numpy().squeeze()

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts
