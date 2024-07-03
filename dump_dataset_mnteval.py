"""
This file (dump_dataset.py) is designed for:
    dump dataset for DMD evaluating
Copyright (c) 2024, Zhiyu Pan. All rights reserved.
"""
import os
import os.path as osp
import pickle
import random
import numpy as np
import argparse

from fptools import fp_verifinger
area_thresh = 40000

def create_datalist(prefix, dataname, img_type='bmp'):
    # NIST4
    img_lst = []
    anchor_2d = []
    mnt_folder = osp.join(prefix, f'{dataname}/mnt')
    mnt_gallery_folder = osp.join(mnt_folder, 'gallery')
    mnt_query_folder = osp.join(mnt_folder, 'query')
    mnt_gallery_files = os.listdir(mnt_gallery_folder)
    mnt_query_files = os.listdir(mnt_query_folder)
    for mnt_f in mnt_gallery_files: 
        mnts = fp_verifinger.load_minutiae(osp.join(mnt_gallery_folder, mnt_f))[:, :3]
        for mnt_ in mnts: # one mnt per sample
            # search part
            img_lst.append(osp.join(f"{dataname}", "image", 'gallery', mnt_f.split('.')[0] + f".{img_type}")) # list 对齐
            anchor_2d.append(mnt_)

    for mnt_f in mnt_query_files: 
        mnts = fp_verifinger.load_minutiae(osp.join(mnt_query_folder, mnt_f))[:, :3]
        for mnt_ in mnts:
            # search part
            img_lst.append(osp.join(f"{dataname}", "image", 'query', mnt_f.split('.')[0] + f".{img_type}"))
            anchor_2d.append(mnt_)

    data_lst = {"img": img_lst,  "pose_2d": anchor_2d}
    print(f'{dataname} total {len(img_lst)} samples')
    return data_lst

if __name__ == "__main__":
    random.seed(1016)
    np.random.seed(1016)
    parser = argparse.ArgumentParser("Evaluation for DMD")
    parser.add_argument("--prefix", type=str, default="/disk2/panzhiyu/fingerprint/")
    args = parser.parse_args()
    save_file = './datasets/NISTmnt_eval.pkl'
    NIST_Seris_dict = {}
    datalist = create_datalist(args.prefix, "NIST27_sta")
    datalist = [dict(zip(datalist, v)) for v in zip(*datalist.values())]
    NIST_Seris_dict["NIST27"] = datalist
    # save the data
    with open(save_file, "wb") as fp:
        pickle.dump(NIST_Seris_dict, fp)


