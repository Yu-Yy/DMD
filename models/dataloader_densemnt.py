# -*- encoding: utf-8 -*-
'''
@File    :   dataloader_densemnt.py
@Time    :   2023/11/07 20:56:01
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2023, Zhiyu Pan, Tsinghua University. All rights reserved
'''
import os.path as osp
import pickle

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from torch.utils.data import Dataset

cv2.setUseOptimized(True)

class MntDataset(Dataset):
    def __init__(
        self,
        prefix,
        pkl_path,
        img_ppi=500,
        tar_shape=(299, 299),
        middle_shape=(512, 512),
        dataname="NIST4",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.pkl_path = pkl_path
        self.img_ppi = img_ppi
        self.tar_shape = np.array(tar_shape)
        self.middle_shape = np.array(middle_shape)
        self.dataname = dataname

        self.scale = self.img_ppi * 1.0 / 500 * self.tar_shape[0] / self.middle_shape[0]

        with open(pkl_path, "rb") as fp:
            items = pickle.load(fp)
            self.items = items[dataname] # 以细节点为主导


    def load_img(self, img_path):
        img = np.asarray(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        return img

    def __len__(self):
        return len(self.items)

    def _processing_(self, img, minu, pose_2d):
        rot = 0
        shift = np.zeros(2)
        flow = np.zeros(198)
        center = self.tar_shape[::-1] / 2.0

        # TPS deformation
        matches = [cv2.DMatch(ii, ii, 0) for ii in range(len(flow) // 2)]
        tps_pts, minu = fast_tps_distortion(
            img.shape,
            self.tar_shape,
            flow,
            matches,
            minu=minu,
            p_center=pose_2d[:2],
            p_theta=np.deg2rad(pose_2d[2]), # in clockwise for positive
            t_scale=self.scale,
            t_shift=shift,
            t_rotation=-np.deg2rad(rot),
        )

        img = cv2.remap(img, tps_pts[..., 0], tps_pts[..., 1], cv2.INTER_LINEAR, borderValue=127.5)

        return img, minu, center, shift, rot

    def __getitem__(self, index):
        item = self.items[index]
        img = self.load_img(osp.join(self.prefix, item["img"]))
        minu = None
        pose_2d = item["pose_2d"] # (x, y, theta) # in clockwise
        path = "/".join(item["img"].split("/")[-3:]).split(".")[0]
        img_r, _, _, _, _ = self._processing_(img, minu, pose_2d)
        img_r = (img_r - 127.5) / 127.5

        return {
            "img_r": img_r[None].astype(np.float32),
            'minu_r': pose_2d,
            'index': index,
            "name": path,
        }


class VirtDataset(Dataset):
    def __init__(
        self,
        prefix,
        pkl_path,
        img_ppi=500,
        tar_shape=(299, 299),
        middle_shape=(512, 512),
        dataname="NIST4",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.pkl_path = pkl_path
        self.img_ppi = img_ppi
        self.tar_shape = np.array(tar_shape)
        self.middle_shape = np.array(middle_shape)
        self.scale = self.img_ppi * 1.0 / 500 * self.tar_shape[0] / self.middle_shape[0]

        with open(pkl_path, "rb") as fp:
            items = pickle.load(fp)
            self.items = items[dataname]['datalist'] # 以细节点为主导
            self.count_number = items[dataname]["count_list"]

    def load_img(self, img_path):
        img = np.asarray(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        return img

    def __len__(self):
        return self.count_number[-1]

    def _processing_(self, img, minu, pose_2d):
        rot = 0
        shift = np.zeros(2)
        flow = np.zeros(198)
        center = self.tar_shape[::-1] / 2.0

        # TPS deformation
        matches = [cv2.DMatch(ii, ii, 0) for ii in range(len(flow) // 2)]
        tps_pts, minu = fast_tps_distortion(
            img.shape,
            self.tar_shape,
            flow,
            matches,
            minu=minu,
            p_center=pose_2d[:2],
            p_theta=np.deg2rad(pose_2d[2]), # in clockwise for positive
            t_scale=self.scale,
            t_shift=shift,
            t_rotation=-np.deg2rad(rot),
        )

        img = cv2.remap(img, tps_pts[..., 0], tps_pts[..., 1], cv2.INTER_LINEAR, borderValue=127.5)

        return img, minu, center, shift, rot

    def __getitem__(self, index):
        # parse the image_idx and the anchor_idx
        img_idx = np.where(self.count_number > index)[0][0] - 1
        anchor_idx = index - self.count_number[img_idx]
        item = self.items[img_idx]
        img = self.load_img(osp.join(self.prefix, item["img"]))
        pose_2d = np.loadtxt(osp.join(self.prefix, item["pose_2d"]), dtype=np.float32)[anchor_idx,:] # (x, y, theta) # in clockwise
        minu = None
        path = "/".join(item["img"].split("/")[-3:]).split(".")[0]
        img_r, _, _, _, _ = self._processing_(img, minu, pose_2d)
        img_r = (img_r - 127.5) / 127.5
        return {
            "img_r": img_r[None].astype(np.float32),
            'minu_r': pose_2d,
            'index': index,
            "name": path,
        }


def affine_matrix(scale=1.0, theta=0.0, trans=np.zeros(2), trans_2=np.zeros(2)):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    t = np.dot(R, trans) + trans_2
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])


def fast_tps_distortion(
    cur_shape,
    tar_shape,
    flow,
    matches,
    minu=None,
    p_center=None,
    p_theta=0,
    t_scale=1,
    t_shift=np.zeros(2),
    t_rotation=0,
    padding=32,
    num_ctrl=16,
):
    cur_center = np.array([cur_shape[1], cur_shape[0]]) / 2
    tar_center = np.array([tar_shape[1], tar_shape[0]]) / 2
    if p_center is None:
        p_center = cur_center
    R_theta = np.array([[np.cos(p_theta), -np.sin(p_theta)], [np.sin(p_theta), np.cos(p_theta)]])
    R_rotation = np.array([[np.cos(t_rotation), -np.sin(t_rotation)], [np.sin(t_rotation), np.cos(t_rotation)]])

    src_x, src_y = np.meshgrid(np.linspace(-200, 200, 11), np.linspace(-160, 160, 9))
    src_x = src_x.T.reshape(-1)
    src_y = src_y.T.reshape(-1)
    src_cpts = np.stack((src_x, src_y), axis=-1)
    tar_cpts = src_cpts + flow.reshape(-1, 2)

    src_cpts = src_cpts.dot(R_theta.T) + p_center[None]
    tar_cpts = tar_cpts.dot(R_rotation.T) * t_scale + tar_center + t_shift

    if minu is not None:
        minu = minu.astype(np.float32)
        minu_tar = np.zeros_like(minu)

        tps_inv = cv2.createThinPlateSplineShapeTransformer()
        tps_inv.estimateTransformation(src_cpts[None].astype(np.float32), tar_cpts[None].astype(np.float32), matches)
        p_s = minu[:, :2] - 5 * np.stack([np.cos(np.deg2rad(minu[:, 2])), np.sin(np.deg2rad(minu[:, 2]))], axis=-1)
        p_e = minu[:, :2] + 5 * np.stack([np.cos(np.deg2rad(minu[:, 2])), np.sin(np.deg2rad(minu[:, 2]))], axis=-1)
        p_s = tps_inv.applyTransformation(p_s.reshape(1, -1, 2))[1].reshape(*p_s.shape)
        p_e = tps_inv.applyTransformation(p_e.reshape(1, -1, 2))[1].reshape(*p_e.shape)
        minu_tar[:, :2] = tps_inv.applyTransformation(minu[:, :2].reshape(1, -1, 2))[1].reshape(*p_e.shape)
        delta = (p_e - p_s) / 10
        minu_tar[:, 2] = np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0]))
    else:
        minu_tar = None

    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(tar_cpts[None].astype(np.float32), src_cpts[None].astype(np.float32), matches)
    grid_x = np.linspace(-padding / 2, tar_shape[1] + padding / 2, num_ctrl)
    grid_y = np.linspace(-padding / 2, tar_shape[0] + padding / 2, num_ctrl)
    tar_pts = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).astype(np.float32)
    src_pts = tps.applyTransformation(tar_pts.reshape(1, -1, 2))[1].reshape(*tar_pts.shape)

    bspline_x = RectBivariateSpline(grid_y, grid_x, src_pts[..., 0])
    bspline_y = RectBivariateSpline(grid_y, grid_x, src_pts[..., 1])
    tmp_x, tmp_y = np.meshgrid(np.arange(tar_shape[1]), np.arange(tar_shape[0]))
    tps_x = bspline_x.ev(tmp_y, tmp_x).astype(np.float32)
    tps_y = bspline_y.ev(tmp_y, tmp_x).astype(np.float32)
    tps_pts = np.stack((tps_x, tps_y), axis=-1)

    return tps_pts, minu_tar


def normlization_angle(delta):
    delta = np.abs(delta) % 360
    return np.deg2rad(np.minimum(delta, 360 - delta))

from itertools import product
import pickle
import os
from tqdm import tqdm
# dataloader for LSRA-matching
class MatchDataset(Dataset):
    def __init__(self, feat_folder) -> None:
        super().__init__()
        self.feat_folder = feat_folder
        self.search_folder = osp.join(feat_folder, "search")
        self.gallery_folder = osp.join(feat_folder, "gallery")
        # search and gallery list
        self.search_list = os.listdir(self.search_folder)
        self.search_list.sort()
        self.gallery_list = os.listdir(self.gallery_folder)
        self.gallery_list.sort()        
        # get the combination of search and gallery
        self.items = list(product(self.search_list, self.gallery_list))
        # get the max mnt number, for padding

        # self.max_mnt_num_search = 0
        # print("Calculating the max mnt number for search and gallery")
        # for s_file in tqdm(self.search_list):
        #     s_feat = pickle.load(open(osp.join(self.search_folder, s_file), "rb"))
        #     self.max_mnt_num_search = max(self.max_mnt_num_search, s_feat["mnt"].shape[0])
        # self.max_mnt_num_gallery = 0
        # for g_file in tqdm(self.gallery_list):
        #     g_feat = pickle.load(open(osp.join(self.gallery_folder, g_file), "rb"))
        #     self.max_mnt_num_gallery = max(self.max_mnt_num_gallery, g_feat["mnt"].shape[0])

    def load_img(self, img_path):
        img = np.asarray(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        return img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        current_item = self.items[index]
        search_feat = pickle.load(open(osp.join(self.search_folder, current_item[0]), "rb"))
        gallery_feat = pickle.load(open(osp.join(self.gallery_folder, current_item[1]), "rb"))
        # get the index of search and gallery of the original list
        index_search = self.search_list.index(current_item[0])
        index_gallery = self.gallery_list.index(current_item[1])
        index_pair = np.array([index_search, index_gallery])

        search_feat_mask = search_feat["mask"]
        gallery_feat_mask = gallery_feat["mask"]
        search_feat_mnt = search_feat["mnt"]
        gallery_feat_mnt = gallery_feat["mnt"]
        search_feat_desc = search_feat["feat"]
        gallery_feat_desc = gallery_feat["feat"]
        # # padding for search and gallery, padding the np.nan
        # search_feat_mnt = np.pad(search_feat_mnt, ((0, self.max_mnt_num_search - search_feat_mnt.shape[0]), (0, 0)), "constant", constant_values=np.nan)
        # gallery_feat_mnt = np.pad(gallery_feat_mnt, ((0, self.max_mnt_num_gallery - gallery_feat_mnt.shape[0]), (0, 0)), "constant", constant_values=np.nan)
        # search_feat_desc = np.pad(search_feat_desc, ((0, self.max_mnt_num_search - search_feat_desc.shape[0]), (0, 0)), "constant", constant_values=np.nan)
        # gallery_feat_desc = np.pad(gallery_feat_desc, ((0, self.max_mnt_num_gallery - gallery_feat_desc.shape[0]), (0, 0)), "constant", constant_values=np.nan)
        # search_feat_mask = np.pad(search_feat_mask, ((0, self.max_mnt_num_search - search_feat_mask.shape[0]), (0, 0)), "constant", constant_values=np.nan)
        # gallery_feat_mask = np.pad(gallery_feat_mask, ((0, self.max_mnt_num_gallery - gallery_feat_mask.shape[0]), (0, 0)), "constant", constant_values=np.nan)
        return {
            "search_mnt": search_feat_mnt,
            "gallery_mnt": gallery_feat_mnt,
            "search_desc": search_feat_desc,
            "gallery_desc": gallery_feat_desc,
            "search_mask": search_feat_mask,
            "gallery_mask": gallery_feat_mask,
            "index": index_pair,
        }

if __name__ == "__main__":
    folder = "/disk2/panzhiyu/fingerprint/NIST_SD27/DMD_6"
    dataset = MatchDataset(folder)
    print(len(dataset))
    output_dict = dataset[0]
    import pdb; pdb.set_trace()