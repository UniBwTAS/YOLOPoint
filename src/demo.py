import argparse
import numpy as np
import os
from time import time
import cv2
import torch
import torch.nn.functional as F
from utils.utils import load_model, nms_fast
import yaml
from glob import glob
from utils.plots_yolo import Annotator, colors
from utils.general_yolo import non_max_suppression


class YoloPointFrontend:
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, config, args, ros=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.config = config
        model_config = config['model']

        self.sp_config = model_config['superpoint']
        self.yolo_config = model_config['yolo']

        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        version = model_config['version']
        model_name = model_config['name']
        self.names = config['names']
        self.model = load_model(inp_ch=3, names=self.names, version=version, model_name=model_name).to(device)
        state_dict = torch.load(args.weights_path)['model_state_dict']
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.fuse()

        self.filter_pts = args.filter_pts

        # use only for demo visualization with surround front
        self.crop_resize = self.config.get('crop_resize')

        self.display_scale = args.display_scale

        self.visualize = args.visualize
        if self.visualize:
            self.tracker = PointTracker(max_length=config['model']['superpoint']['max_length'],
                                   nn_thresh=self.sp_config['nn_thresh'])

        if not ros:
            img_paths = sorted(glob(os.path.join(args.source, '*.png'))) + \
                        sorted(glob(os.path.join(args.source, '*.jpg')))
            assert len(img_paths) > 0, f"Directory {args.source} either does not " \
                                       f"exist or no images with extension png or jpg found"
            self.stream_imgs(img_paths)

        self.templates = {} # currently only used with ROS

    def preprocess(self, img, interpolation=cv2.INTER_LINEAR):
        shape0 = img.shape[:2]
        resize_fac = 1.
        cut_h0, cut_w0 = 0, 0

        if self.crop_resize:
            # crop and resize image
            cr = self.crop_resize
            w1 = cr[4]
            img = img[cr[0]:cr[1], cr[2]:cr[3]]
            h0, w0 = img.shape[:2]  # shape after crop
            resize_fac = w1 / w0
            h1 = round(h0 * resize_fac)
            img = cv2.resize(img, (w1, h1), interpolation=interpolation)

        if shape0[0] % 32 != 0 or shape0[1] % 32 != 0:
            # make dims divisable by 32 by cropping
            h0, w0 = img.shape[:2]  # orig hw
            cut_h = (h0 % 32) / 2
            cut_w = (w0 % 32) / 2
            cut_h0 = np.ceil(cut_h).astype(int)
            cut_h1 = np.floor(cut_h).astype(int)
            cut_w0 = np.ceil(cut_w).astype(int)
            cut_w1 = np.floor(cut_w).astype(int)
            img = img[cut_h0:h0 - cut_h1, cut_w0:w0 - cut_w1]

        return img, cut_h0, cut_w0, resize_fac

    @torch.no_grad()
    def process_img(self, img, rostpc=None):

        img, cth, ctw, resize_fac = self.preprocess(img)
        H, W, ch = img.shape
        img = img.transpose((2,0,1)).astype(np.float32)/255.
        inp = torch.clone(torch.tensor(img))
        inp = inp.unsqueeze(0).to(self.device)

        # Forward pass of network.
        outs = self.model(inp)
        semi, coarse_desc, (obj_preds, obj) = outs["semi"], outs["desc"], outs["objects"]
        # Convert pytorch -> numpy
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense /= (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.sp_config['detection_threshold'])  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = nms_fast(pts, H, W, dist_thresh=self.sp_config['nms'])  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        # TODO: parallelize object nms with point nms
        # Filter points with object bboxes
        obj_preds = non_max_suppression(obj_preds,
                                  self.yolo_config['conf_thres_box'],
                                  self.yolo_config['iou_thres_box'],
                                  labels=[],
                                  multi_label=True,
                                  agnostic=True,
                                  max_det=self.yolo_config['max_det'])

        if self.filter_pts:
            def filter_points(obj_preds, points, im_shape):
                obj_preds = obj_preds.cpu().numpy()
                mask = np.ones(im_shape)
                points = points.transpose()
                points2 = points[:, :2].astype(int)
                for *xyxy, _, _ in obj_preds:
                    x0, y0, x1, y1 = np.rint(xyxy).astype(int)
                    mask[y0:y1, x0:x1] = 0
                if rostpc:
                    # also filter points on ego vehicle for given template
                    try:
                        mask *= self.templates[rostpc]
                    except KeyError:
                        print(f"Template for {rostpc} not found.")
                mask1D = mask[points2[:, 1], points2[:, 0]] == 1
                points = points[mask1D]
                return points.transpose()

            img_shape = (H, W)
            pts = filter_points(obj_preds[0], pts, img_shape)

        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)
            desc = F.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        # adjust points to original image coords before crop
        pts = pts.transpose()

        pts[:, 0] = ((pts[:, 0] + ctw) / resize_fac)
        pts[:, 1] = ((pts[:, 1] + cth) / resize_fac)
        pts = pts.transpose()
        obj_preds[0][:, :4] = (obj_preds[0][:, :4] + torch.tensor([ctw, cth, ctw, cth], device=self.device)) / resize_fac

        if cr:= self.crop_resize:
            pts[:, 0] += cr[2]
            pts[:, 1] += cr[0]
            obj_preds[0][:, :4] += torch.tensor([cr[2], cr[0], cr[2], cr[0]], device=self.device)

        return pts, desc, obj_preds

    def stream_imgs(self, img_paths):
        # Usage: stream images from directory
        for p in img_paths:
            img = cv2.imread(p)
            pts, desc, obj_preds = self.process_img(img)
            if self.visualize:
                self.visualize_objects_and_tracks(img, pts, desc, obj_preds)

    def visualize_objects_and_tracks(self, img, pts, desc, obj):

        self.tracker.update(pts, desc)

        # Get tracks for points which were matched successfully across all frames.
        tracks = self.tracker.get_tracks(self.config['model']['superpoint']['min_length'])
        out = img.copy()

        tracks[:, 1] /= float(self.config['model']['superpoint']['nn_thresh'])  # Normalize track scores to [0,1].
        self.tracker.draw_tracks(out, tracks)
        object_annotator = Annotator(out, line_width=1, example=str(self.names))
        # annotate bboxes and draw tracks
        for *xyxy, conf, cls in reversed(obj[0]):
            # Add bbox to image
            c = int(cls)  # integer class
            label = f'{self.names[c]} {conf:.2f}'
            # Stream results
            object_annotator.box_label(xyxy, label, color=colors(c, True))
            out = object_annotator.result()
        if self.crop_resize:
            out = cv2.resize(out, (np.array(out.shape[:2][::-1])*2).astype(int))
        if self.display_scale != 1.:
            new_shape = (np.array(out.shape[:2][::-1]) * self.display_scale).astype(int)
            out = cv2.resize(out, new_shape)
        cv2.imshow("Visualization", out)
        cv2.waitKey(1)


class PointTracker:
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.
    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        # Jet colormap for visualization.
        self.myjet = np.array([[0., 0., 0.5],
                          [0., 0., 0.99910873],
                          [0., 0.37843137, 1.],
                          [0., 0.83333333, 1.],
                          [0.30044276, 1., 0.66729918],
                          [0.66729918, 1., 0.30044276],
                          [1., 0.90123457, 0.],
                          [1., 0.48002905, 0.],
                          [0.99910873, 0.07334786, 0.],
                          [0.5, 0., 0.]])

    @staticmethod
    def nn_match_two_way(desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.
        Inputs:
          desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.
        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.
        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.
        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """ Visualize tracks all overlayed on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = self.myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


if __name__ == '__main__':
    # Parse command line arguments.
    demo_parser = argparse.ArgumentParser(description='YOLOPoint Demo.')
    demo_parser.add_argument('config', type=str, default='configs/kitti_inference.yaml',
                             help='path/to/config (default: configs/kitti_inference.yaml)')
    demo_parser.add_argument('source', type=str, default='',
                             help='path/to/directory')
    demo_parser.add_argument('--weights_path', type=str, default='logs/YOLOPointS_kitti_nomo/checkpoints'
                                                            '/YOLOPointS_kitti_nomo_110_11111_last_ckpt.pth.tar',
                             help='Path to pretrained weights file'
                             '(default: logs/YOLOPointS_kitti_nomo/checkpoints'
                             '/YOLOPointS_kitti_nomo_110_11111_last_ckpt.pth.tar).')
    demo_parser.add_argument('--filter_pts', action='store_false',
                             help='Filter out dynamic points by default')
    demo_parser.add_argument('--visualize', action='store_true',
                             help='Opencv visualization of points and tracks.')
    demo_parser.add_argument('--display_scale', type=float, default=1.,
                             help='Factor to scale output visualization (default: 1, requires --visualize argument).')
    args = demo_parser.parse_args()

    # Ensure that all relative paths start from project root
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_directory, '..'))

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    fe = YoloPointFrontend(config=config, args=args)
