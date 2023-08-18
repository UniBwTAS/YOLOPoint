import torch
from torch import nn
from utils.utils import warp_points
# from utils.debug import save_to_numpy
import torch.nn.functional as F
import numpy as np
from utils.metrics_yolo import bbox_iou
from utils.torch_utils_yolo import de_parallel
from utils.utils import warp_image_batch, getMasks, homography_scaling
# from info_nce import InfoNCE

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeObjectLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, config, device, autobalance=False):
        # device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=config.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = config['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model.Detect    # de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, config, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        # bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach()
        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


def descriptor_loss_dense(descriptors, descriptors_warped, homographies, mask_valid=None,
                          cell_size=8, lambda_d=0.05, margin_pos=1, margin_neg=0.2, device='cpu'):
    """
    Compute descriptor loss from descriptors_warped and given homographies
    Possibly better loss function can be found here:
    https://github.com/rpautrat/SuperPoint/blob/61bc16ff0e096340a26feb4cf80bf6b4bacee267/superpoint/models/utils.py#L110
    explanation here:
    https://github.com/rpautrat/SuperPoint/issues/157

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    """

    # adjust lambda_d approx. according to image size
    homographies = homographies.to(device)
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]

    H, W = Hc * cell_size, Wc * cell_size

    with torch.no_grad():
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        ####### we only need to do all this once
        coord_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc), indexing='ij'), dim=2)
        coord_cells = coord_cells.type(torch.FloatTensor).to(device)
        coord_cells = coord_cells * cell_size + cell_size // 2
        # coord_cells is now a grid containing the coordinates of the Hc x Wc
        # center pixels of the 8x8 cells of the image

        coord_cells = coord_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order

        warped_coord_cells = coord_cells.view([-1, 2]) / shape * 2 - 1  # norm points
        warped_coord_cells = torch.stack((warped_coord_cells[:, 1], warped_coord_cells[:, 0]), dim=1)  # (y, x) to (x, y)

        ###### up to here we only need to do these things once --> make class with __call__

        warped_coord_cells = warp_points(warped_coord_cells, homographies, device)

        warped_coord_cells = torch.stack((warped_coord_cells[:, :, 1], warped_coord_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

        warped_coord_cells = (warped_coord_cells + 1) * shape / 2  # denorm points
        warped_coord_cells = warped_coord_cells.view([-1, Hc, Wc, 1, 1, 2])

        # compute the pairwise distance
        cell_distances = coord_cells - warped_coord_cells
        cell_distances = torch.norm(cell_distances, dim=-1)

        # cells that have travelled more than descriptor_dist are set to false
        mask = cell_distances.le(cell_size - 0.5) # 0.5 # trick (8 in the original paper, 7.5 in tf imp) # s_hwh'w'
        mask = mask.type(torch.FloatTensor).to(device)  # TODO: what happens when this part is left out? i.e. mixed procision?

    # compute the pairwise dot product between descriptors: d^t * d
    # descriptors should be
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)  # [batch, D, Hc, Wc] --> [batch, Hc, Wc, D]
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = (descriptors * descriptors_warped).sum(dim=-1)
    # dot_product_desc.shape = [batch_size, Hc, Wc, Hc, Wc]
    # dot_product_desc = torch.nn.functional.relu(dot_product_desc)   # do ablation if this actually improves results

    # ##### comment this out when training from scratch:
    # dot_product_desc = (torch.nn.functional.normalize(dot_product_desc.view([batch_size, Hc, Wc, Hc * Wc]), p=2, dim=3))\
    #     .view([batch_size, Hc, Wc, Hc, Wc])
    # # dot_product_desc.shape = [batch_size, Hc, Wc]
    # dot_product_desc = (torch.nn.functional.normalize(dot_product_desc.view([batch_size, Hc * Wc, Hc, Wc]), p=2, dim=1)) \
    #     .view([batch_size, Hc, Wc, Hc, Wc])

    # hinge loss
    positive_dist = torch.max(margin_pos - dot_product_desc, torch.tensor(0.).to(device))   # max isn't actually necessary here
    negative_dist = torch.max(dot_product_desc - margin_neg, torch.tensor(0.).to(device))
    # sum of the dimension

    if mask_valid is None:
        # dense method
        mask_valid = torch.ones(batch_size, 1, Hc * cell_size, Wc * cell_size)
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lambda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid

    # tf imp
    normalization = torch.sum(mask_valid) * Hc * Wc
    loss_desc = torch.sum(mask_valid * loss_desc) / normalization
    # pos_sum = (torch.sum(mask_valid * lambda_d * mask * positive_dist) / normalization).item()
    # neg_sum = (torch.sum(mask_valid * (1 - mask) * negative_dist) / normalization).item()

    return loss_desc #, pos_sum, neg_sum


def get_coor_cells(Hc, Wc, uv=False, device='cpu'):  # isn't this function already somewhere?
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc), indexing='ij'), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:, 1], coor_cells[:, 0]), dim=1)  # (y, x) to (x, y)

    return coor_cells.to(device)

def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts / shape * 2 - 1
    return pts

def descriptor_loss_sparse(descriptors, descriptors_warped, mask_valid_warp, inv_homographies, num_samples_per_image=1500,
                num_masked_non_matches_per_match=120, cell_size=8, device='cpu'):
    """
    Compute the batch-wise cross-image pixel-wise contrastive loss between descriptor and warped descriptor
    :param descriptors: torch Tensor [batch, D, Hc, Wc]
    :param descriptors_warped:  torch Tensor [batch, D, Hc, Wc]
    :param mask_valid_warp: warped valid mask [batch, 1, H, W]
    :param inv_homographies: [batch, 3, 3]
    :param num_masked_non_matches_per_match: number of pixel-wise non-matches per match
    """

    assert descriptors.shape[-1] * descriptors.shape[-2] >= num_samples_per_image, \
        "Number of samples per image must be greater than number of pixels in image"

    with torch.no_grad():

        batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]

        # matches
        uv_a = get_coor_cells(Hc, Wc, uv=True)

        inv_homographies = inv_homographies.to(device)

        # inverse warp mask valid for filtering out non-valid and out-of-frame points
        mask_valid_warp = warp_image_batch(mask_valid_warp, inv_homographies, mode='nearest', device=device)
        # downscale
        mask_valid_warp = getMasks(mask_valid_warp, device, cell_size)

        inv_homographies = homography_scaling(inv_homographies, Hc, Wc, device=device)

        uv_b_matches = warp_points(uv_a.to(device), inv_homographies, device).round_()

        # separate batches for variable length
        uv_b_matches = list(uv_b_matches.chunk(batch_size))
        uv_a = uv_a.to(device)
        uv_a = list(uv_a.repeat(batch_size, 1, 1).chunk(batch_size))

        mask_valid_warp = (mask_valid_warp == 1.).flatten(1, -1)  # gonna have to inverse warp the warped valid masks

        def mask_out_non_valid_matches(idx_list, batch_idx):    # mask out matches that are outside the valid mask
            idx_list[batch_idx] = (idx_list[batch_idx]).squeeze(0)
            idx_list[batch_idx] = idx_list[batch_idx][mask_valid_warp[batch_idx]]
            return idx_list

        for batch_idx in range(batch_size):
            uv_a = mask_out_non_valid_matches(uv_a, batch_idx)
            uv_b_matches = mask_out_non_valid_matches(uv_b_matches, batch_idx)

        match_pool_size = min([x.shape[0] for x in uv_a])
        match_pool_size = min(num_samples_per_image, match_pool_size)

        # crop and shuffle
        uv_a_positives = []
        uv_b_positives = []
        # truncate matches
        for batch_idx in range(batch_size):
            choice = torch.randperm((uv_a[batch_idx].shape[0]))
            uv_a_positives.append(((uv_a[batch_idx]).clone())[choice][:match_pool_size])
            uv_b_positives.append(((uv_b_matches[batch_idx]).clone())[choice][:match_pool_size])

        # list to tensor
        uv_a_positives = torch.stack(uv_a_positives).to(device) # indices of descriptor A
        uv_b_positives = torch.stack(uv_b_positives).to(device) # indices of descriptor B that match A

        # norm points to -1, 1 for grid_sample
        uv_a_positives_norm = normPts(uv_a_positives, torch.tensor([Wc, Hc]).float().to(device))
        uv_b_positives_norm = normPts(uv_b_positives, torch.tensor([Wc, Hc]).float().to(device))

    # get match similarity
    def sampleDescriptors(desc_pred, indices, mode='bilinear'):
        indices = indices.unsqueeze(1).to(device)
        sampled_descriptors = F.grid_sample(desc_pred, indices, mode=mode, align_corners=True)
        sampled_descriptors = sampled_descriptors.squeeze(2).transpose(1, 2)
        return sampled_descriptors
    # sample descriptors and matching descriptors from warped image
    descriptor_samples = sampleDescriptors(descriptors, uv_a_positives_norm)    # query [batch, num_samples_per_image, D]
    descriptor_warped_samples = sampleDescriptors(descriptors_warped, uv_b_positives_norm)  # positives [batch, num_samples_per_image, D]

    # similar is 1, dissimilar is -1
    # descriptor_samples.shape = [batch_size, samples_per_image, D]
    pos_match_prod = (descriptor_samples*descriptor_warped_samples).sum(-1).flatten()

    descriptor_samples = descriptor_samples.flatten(0, 1)   # [batch_size*samples_per_image, D]
    descriptor_warped_samples = descriptor_warped_samples.flatten(0, 1)

    num_matches = descriptor_samples.shape[0]

    # create matrix of randomly selected warped descriptors of dim [num_matches, num_non_matches_per_match, D]
    indices_shape = (num_masked_non_matches_per_match, num_matches)
    ordered_indices = np.arange(num_matches)
    ordered_indices = np.broadcast_to(ordered_indices, indices_shape)
    random_indices = np.random.randint(0, num_matches, size=indices_shape)

    # replace too close pixels (not implemented)
    # remove accidental matches
    same_mask = ordered_indices == random_indices
    if nonzero := np.count_nonzero(same_mask):
        while True:
            random_entries = np.random.randint(0, nonzero, nonzero)
            if (random_indices[same_mask] != random_entries).any():
                random_indices[same_mask] = random_entries
                break
    random_indices = torch.from_numpy(random_indices)

    descriptor_warped_samples_flat_mat = descriptor_warped_samples[random_indices]  # [num_non_matches_per_match, num_matches, D] negative
    descriptor_samples = descriptor_samples.unsqueeze(0).expand(num_masked_non_matches_per_match, -1, -1)   # query

    neg_match_loss = (descriptor_samples * descriptor_warped_samples_flat_mat).sum(-1)  # [num_non_matches_per_match, num_matches]
    # negative similarities within a margin of 0.1 of -1 are considered correct
    neg_match_loss = torch.clamp(neg_match_loss - 0.1, min=0).flatten()
    num_hard_negatives = torch.count_nonzero(neg_match_loss)
    neg_match_loss = neg_match_loss.sum()
    neg_match_loss /= (num_hard_negatives + 1)

    match_loss = torch.clamp(1 - pos_match_prod, min=0)
    match_loss = torch.mean(match_loss)

    loss = match_loss + neg_match_loss

    return loss



def infonce(descriptors, descriptors_warped, mask_valid_warp, inv_homographies, num_samples_per_image=1500,
            num_masked_non_matches_per_match=120, cell_size=8, device='cpu', tau=0.07):
    """
    Compute the batch-wise cross-image pixel-wise contrastive loss between descriptor and warped descriptor
    :param descriptors: torch Tensor [batch, D, Hc, Wc]
    :param descriptors_warped:  torch Tensor [batch, D, Hc, Wc]
    :param mask_valid_warp: warped valid mask [batch, 1, H, W]
    :param inv_homographies: [batch, 3, 3]
    :param num_masked_non_matches_per_match: number of pixel-wise non-matches per match
    """

    assert descriptors.shape[-1] * descriptors.shape[-2] >= num_samples_per_image, \
        "Number of samples per image must be greater than number of pixels in image"

    with torch.no_grad():

        batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]

        # matches
        uv_a = get_coor_cells(Hc, Wc, uv=True)

        inv_homographies = inv_homographies.to(device)

        # inverse warp mask valid for filtering out non-valid and out-of-frame points
        mask_valid_warp = warp_image_batch(mask_valid_warp, inv_homographies, mode='nearest', device=device)
        # downscale
        mask_valid_warp = getMasks(mask_valid_warp, device, cell_size)

        inv_homographies = homography_scaling(inv_homographies, Hc, Wc, device=device)

        uv_b_matches = warp_points(uv_a.to(device), inv_homographies, device).round_()

        # separate batches for variable length
        uv_b_matches = list(uv_b_matches.chunk(batch_size))
        uv_a = uv_a.to(device)
        uv_a = list(uv_a.repeat(batch_size, 1, 1).chunk(batch_size))

        mask_valid_warp = (mask_valid_warp == 1.).flatten(1, -1)  # gonna have to inverse warp the warped valid masks

        def mask_out_non_valid_matches(idx_list, batch_idx):
            idx_list[batch_idx] = (idx_list[batch_idx]).squeeze(0)
            idx_list[batch_idx] = idx_list[batch_idx][mask_valid_warp[batch_idx]]
            return idx_list

        for batch_idx in range(batch_size):
            uv_a = mask_out_non_valid_matches(uv_a, batch_idx)
            uv_b_matches = mask_out_non_valid_matches(uv_b_matches, batch_idx)

        match_pool_size = min([x.shape[0] for x in uv_a])
        match_pool_size = min(num_samples_per_image, match_pool_size)

        # crop and shuffle
        uv_a_positives = []
        uv_b_positives = []

        for batch_idx in range(batch_size):
            choice = torch.randperm((uv_a[batch_idx].shape[0]))
            uv_a_positives.append(((uv_a[batch_idx]).clone())[choice][:match_pool_size])
            uv_b_positives.append(((uv_b_matches[batch_idx]).clone())[choice][:match_pool_size])

        # list to tensor
        uv_a_positives = torch.stack(uv_a_positives).to(device)
        uv_b_positives = torch.stack(uv_b_positives).to(device)

        # norm points to -1, 1 for grid_sample
        uv_a_positives_norm = normPts(uv_a_positives, torch.tensor([Wc, Hc]).float().to(device))
        uv_b_positives_norm = normPts(uv_b_positives, torch.tensor([Wc, Hc]).float().to(device))

    # get match similarity
    def sampleDescriptors(desc_pred, indices, mode='bilinear'):
        indices = indices.unsqueeze(1).to(device)
        sampled_descriptors = F.grid_sample(desc_pred, indices, mode=mode, align_corners=True)
        sampled_descriptors = sampled_descriptors.squeeze(2).transpose(1, 2)
        return sampled_descriptors

    descriptor_samples = sampleDescriptors(descriptors, uv_a_positives_norm)
    descriptor_warped_samples = sampleDescriptors(descriptors_warped, uv_b_positives_norm)

    # similar is 1, dissimilar is -1
    # descriptor_samples.shape = [batch_size, samples_per_image, D)
    pos_match_prod = (descriptor_samples*descriptor_warped_samples).sum(-1).flatten()   # [batch_size*samples_per_image,]

    descriptor_samples = descriptor_samples.flatten(0, 1)
    descriptor_warped_samples = descriptor_warped_samples.flatten(0, 1)

    num_matches = descriptor_samples.shape[0]

    # create matrix of randomly selected warped descriptors of dim [num_matches, num_non_matches_per_match, D]
    indices_shape = (num_masked_non_matches_per_match, num_matches)
    ordered_indices = np.arange(num_matches)
    ordered_indices = np.broadcast_to(ordered_indices, indices_shape)
    random_indices = np.random.randint(0, num_matches, size=indices_shape)

    # replace too close pixels (not implemented)

    same_mask = ordered_indices == random_indices
    if nonzero := np.count_nonzero(same_mask):
        while True:
            random_entries = np.random.randint(0, nonzero, nonzero)
            if (random_indices[same_mask] != random_entries).any():
                random_indices[same_mask] = random_entries
                break
    random_indices = torch.from_numpy(random_indices)

    descriptor_warped_samples_flat_mat = descriptor_warped_samples[random_indices].transpose(0, 1)
    descriptor_samples = descriptor_samples.unsqueeze(0).expand(num_masked_non_matches_per_match, -1, -1).transpose(0, 1)
    pos_match_prod = pos_match_prod.unsqueeze(1)

    neg_match_prod = (descriptor_samples * descriptor_warped_samples_flat_mat).sum(-1)
    logits = torch.cat([pos_match_prod, neg_match_prod], dim=1) / tau

    loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
    # TODO: try implementation with hard mining!
    return loss


class ComputeDetectorLoss:
    """
    :param input: prediction
        tensor [batch_size, 65, Hc, Wc]
    :param target: constructed from labels
        tensor [batch_size, 65, Hc, Wc]
    :param mask: valid region in an image
        tensor [batch_size, 1, Hc, Wc]
    :return: normalized loss
        tensor
    """
    def __init__(self, device):
        self.sm = nn.Softmax(dim=1)
        self.loss_func_BCE = nn.BCELoss(reduction='none').to(device)

    def __call__(self, inp, target, mask):
        loss = self.loss_func_BCE(self.sm(inp), target)
        loss = (loss.sum(dim=1) * mask).sum()
        loss = loss / (mask.sum() + 1e-10)
        return loss


class PixelwiseContrastiveLoss(object):

    def __init__(self, image_shape, config):
        self.type = "pixelwise_contrastive"
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]
        self._config = config

        self._debug_data = dict()

        self._debug = False

    @property
    def debug(self):
        return self._debug

    @property
    def config(self):
        return self._config

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def debug_data(self):
        return self._debug_data

    def get_loss_matched_and_non_matched_with_l2(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a,
                                                 non_matches_b,
                                                 M_descriptor=None, M_pixel=None, non_match_loss_weight=1.0,
                                                 use_l2_pixel_loss=None):
        """
        Computes the loss function

        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension


        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2)^2

        loss = match_loss + non_match_loss

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_a
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        PCL = PixelwiseContrastiveLoss

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        if M_pixel is None:
            M_pixel = self._config["M_pixel"]

        if use_l2_pixel_loss is None:
            use_l2_pixel_loss = self._config['use_l2_pixel_loss_on_masked_non_matches']

        match_loss, _, _ = PCL.match_loss(image_a_pred, image_b_pred, matches_a, matches_b)

        if use_l2_pixel_loss:
            non_match_loss, num_hard_negatives = \
                self.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b,
                                                       non_matches_a, non_matches_b,
                                                       M_descriptor=M_descriptor,
                                                       M_pixel=M_pixel)
        else:
            # version with no l2 pixel term
            non_match_loss, num_hard_negatives = self.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                                     non_matches_a, non_matches_b,
                                                                                     M_descriptor=M_descriptor)

        return match_loss, non_match_loss, num_hard_negatives

    @staticmethod
    def get_triplet_loss(image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b, alpha):
        """
        Computes the loss function

        \sum_{triplets} ||D(I_a, u_a, I_b, u_{b,match})||_2^2 - ||D(I_a, u_a, I_b, u_{b,non-match)||_2^2 + alpha

        """
        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]
        multiplier = num_non_matches / num_matches

        ## non_matches_a is already replicated up to be the right size
        ## non_matches_b is also that side
        ## matches_a is just a smaller version of non_matches_a
        ## matches_b is the only thing that needs to be replicated up in size

        matches_b_long = torch.t(matches_b.repeat(multiplier, 1)).contiguous().view(-1)

        matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b_long)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)

        triplet_losses = (matches_a_descriptors - matches_b_descriptors).pow(2) - (
                    matches_a_descriptors - non_matches_b_descriptors).pow(2) + alpha
        triplet_loss = 1.0 / num_non_matches * torch.clamp(triplet_losses, min=0).sum()

        return triplet_loss

    @staticmethod
    def match_loss(image_a_pred, image_b_pred, matches_a, matches_b, M=1.0,
                   dist='euclidean', method='1d'):  # dist = 'cos'
        """
        Computes the match loss given by

        1/num_matches * \sum_{matches} ||D(I_a, u_a, I_b, u_b)||_2^2

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b

        :return: match_loss, matches_a_descriptors, matches_b_descriptors
        :rtype: torch.Variable(),

        matches_a_descriptors is torch.FloatTensor with shape torch.Shape([num_matches, descriptor_dimension])
        """
        if method == '2d':

            num_matches = matches_a.size()[0]
            mode = 'bilinear'

            def sampleDescriptors(image_a_pred, matches_a, mode, norm=False):
                image_a_pred = image_a_pred.unsqueeze(0)  # torch [1, D, H, W]
                matches_a.unsqueeze_(0).unsqueeze_(2)
                matches_a_descriptors = F.grid_sample(image_a_pred, matches_a, mode=mode, align_corners=True)
                matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0, 1)

                if norm:
                    dn = torch.norm(matches_a_descriptors, p=2, dim=1)  # Compute the norm of b_descriptors
                    matches_a_descriptors = matches_a_descriptors.div(
                        torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
                return matches_a_descriptors


            matches_a_descriptors = sampleDescriptors(image_a_pred, matches_a, mode, norm=False)
            matches_b_descriptors = sampleDescriptors(image_b_pred, matches_b, mode, norm=False)
        else:
            num_matches = matches_a.size()[0]
            matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
            matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(matches_a) == 1:
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)

        if dist == 'cos':
            match_loss = torch.clamp(M - (matches_a_descriptors * matches_b_descriptors).sum(dim=-1), min=0)
            match_loss = match_loss.sum() / num_matches
        else:
            match_loss = 1.0 / num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        return match_loss, matches_a_descriptors, matches_b_descriptors

    @staticmethod
    def non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, M=0.5, invert=False,
                                  dist='euclidear'):
        """
        Computes the max(0, M - D(I_a,I_b,u_a,u_b))^2 term

        This is effectively:       "a and b should be AT LEAST M away from each other"
        With invert=True, this is: "a and b should be AT MOST  M away from each other"

         :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :param M: the margin
        :type M: float
        :return: torch.FloatTensor with shape torch.Shape([num_non_matches])
        :rtype:
        """

        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a).squeeze()
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b).squeeze()

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(non_matches_a) == 1:
            non_matches_a_descriptors = non_matches_a_descriptors.unsqueeze(0)
            non_matches_b_descriptors = non_matches_b_descriptors.unsqueeze(0)

        norm_degree = 2
        if dist == 'cos':
            non_match_loss = (non_matches_a_descriptors * non_matches_b_descriptors).sum(dim=-1)
        else:
            non_match_loss = (non_matches_a_descriptors - non_matches_b_descriptors).norm(norm_degree, 1)
        if not invert:  # invert = True
            non_match_loss = torch.clamp(M - non_match_loss, min=0).pow(2)
        else:
            if dist == 'cos':
                non_match_loss = torch.clamp(non_match_loss - M, min=0)
            else:
                non_match_loss = torch.clamp(non_match_loss - M, min=0).pow(2)

        hard_negative_idxs = torch.nonzero(non_match_loss)
        num_hard_negatives = len(hard_negative_idxs)

        return non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors

    def non_match_loss_with_l2_pixel_norm(self, image_a_pred, image_b_pred, matches_b,
                                          non_matches_a, non_matches_b, M_descriptor=0.5,
                                          M_pixel=None):

        """

        Computes the total non_match_loss with an l2_pixel norm term

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_a
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a

        :param M_descriptor: margin for descriptor loss term
        :type M_descriptor: float
        :param M_pixel: margin for pixel loss term
        :type M_pixel: float
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        if M_pixel is None:
            M_pixel = self._config["M_pixel"]

        PCL = PixelwiseContrastiveLoss

        num_non_matches = non_matches_a.size()[0]

        non_match_descriptor_loss, num_hard_negatives, _, _ = PCL.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                                                            non_matches_a,
                                                                                            non_matches_b,
                                                                                            M=M_descriptor)

        non_match_pixel_l2_loss, _, _ = self.l2_pixel_loss(matches_b, non_matches_b, M_pixel=M_pixel)

        non_match_loss = (non_match_descriptor_loss * non_match_pixel_l2_loss).sum()

        if self.debug:
            self._debug_data['num_hard_negatives'] = num_hard_negatives
            self._debug_data['fraction_hard_negatives'] = num_hard_negatives * 1.0 / num_non_matches

        return non_match_loss, num_hard_negatives

    def non_match_loss_descriptor_only(self, image_a_pred, image_b_pred, non_matches_a, non_matches_b, M_descriptor=0.5,
                                       invert=False):
        """
        Computes the non-match loss, only using the desciptor norm
        :param image_a_pred:
        :type image_a_pred:
        :param image_b_pred:
        :type image_b_pred:
        :param non_matches_a:
        :type non_matches_a:
        :param non_matches_b:
        :type non_matches_b:
        :param M:
        :type M:
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """
        PCL = PixelwiseContrastiveLoss

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        non_match_loss_vec, num_hard_negatives, _, _ = PCL.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                                                     non_matches_a,
                                                                                     non_matches_b, M=M_descriptor,
                                                                                     invert=invert)

        num_non_matches = long(non_match_loss_vec.size()[0])

        non_match_loss = non_match_loss_vec.sum()

        if self._debug:
            self._debug_data['num_hard_negatives'] = num_hard_negatives
            self._debug_data['fraction_hard_negatives'] = num_hard_negatives * 1.0 / num_non_matches

        return non_match_loss, num_hard_negatives

    def l2_pixel_loss(self, matches_b, non_matches_b, M_pixel=None):
        """
        Apply l2 loss in pixel space.

        This weights non-matches more if they are "far away" in pixel space.

        :param matches_b: A torch.LongTensor with shape torch.Shape([num_matches])
        :param non_matches_b: A torch.LongTensor with shape torch.Shape([num_non_matches])
        :return l2 loss per sample: A torch.FloatTensorof with shape torch.Shape([num_matches])
        """

        if M_pixel is None:
            M_pixel = self._config['M_pixel']

        num_non_matches_per_match = len(non_matches_b) / len(matches_b)

        ground_truth_pixels_for_non_matches_b = torch.t(
            matches_b.repeat(num_non_matches_per_match, 1)).contiguous().view(-1, 1)

        ground_truth_u_v_b = self.flattened_pixel_locations_to_u_v(ground_truth_pixels_for_non_matches_b)
        sampled_u_v_b = self.flattened_pixel_locations_to_u_v(non_matches_b.unsqueeze(1))

        # each element is always within [0,1], you have 1 if you are at least M_pixel away in
        # L2 norm in pixel space
        norm_degree = 2
        squared_l2_pixel_loss = 1.0 / M_pixel * torch.clamp(
            (ground_truth_u_v_b - sampled_u_v_b).float().norm(norm_degree, 1), max=M_pixel)

        return squared_l2_pixel_loss, ground_truth_u_v_b, sampled_u_v_b

    def flattened_pixel_locations_to_u_v(self, flat_pixel_locations):
        """
        :param flat_pixel_locations: A torch.LongTensor of shape torch.Shape([n,1]) where each element
         is a flattened pixel index, i.e. some integer between 0 and 307,200 for a 640x480 image

        :type flat_pixel_locations: torch.LongTensor

        :return A torch.LongTensor of shape (n,2) where the first column is the u coordinates of
        the pixel and the second column is the v coordinate

        """
        u_v_pixel_locations = flat_pixel_locations.repeat(1, 2)
        u_v_pixel_locations[:, 0] = u_v_pixel_locations[:, 0] % self.image_width
        u_v_pixel_locations[:, 1] = u_v_pixel_locations[:, 1] / self.image_width
        return u_v_pixel_locations

    def get_l2_pixel_loss_original(self):
        pass

    def get_loss_original(self, image_a_pred, image_b_pred, matches_a,
                          matches_b, non_matches_a, non_matches_b,
                          M_margin=0.5, non_match_loss_weight=1.0):

        # this is pegged to it's implemenation at sha 87abdb63bb5b99d9632f5c4360b5f6f1cf54245f
        """
        Computes the loss function
        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension
        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )
        loss = match_loss + non_match_loss
        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]

        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

        match_loss = 1.0 / num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        # add loss via non_matches
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        non_match_loss = non_match_loss_weight * 1.0 / num_non_matches * torch.max(zeros_vec, pixel_wise_loss).sum()

        loss = match_loss + non_match_loss

        return loss, match_loss, non_match_loss
