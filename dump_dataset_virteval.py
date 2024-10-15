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
    mnt_folder = osp.join(prefix, f'{dataname}/virt_mnt')
    mnt_gallery_folder = osp.join(mnt_folder, 'gallery')
    mnt_query_folder = osp.join(mnt_folder, 'query')
    mnt_gallery_files = os.listdir(mnt_gallery_folder)
    mnt_query_files = os.listdir(mnt_query_folder)
    count_list = [0]
    counting_sum = 0
    for mnt_f in mnt_gallery_files: 
        img_lst.append(osp.join(f"{dataname}", "image", 'gallery', mnt_f.split('.')[0] + f".{img_type}")) 
        anchor_2d.append(osp.join(f"{dataname}", "virt_mnt", 'gallery', mnt_f))
        anchor_2d_num = np.loadtxt(osp.join(mnt_gallery_folder, mnt_f), dtype=np.float32).shape[0]
        counting_sum += anchor_2d_num
        count_list.append(counting_sum)

    for mnt_f in mnt_query_files: 
        img_lst.append(osp.join(f"{dataname}", "image", 'query', mnt_f.split('.')[0] + f".{img_type}"))
        anchor_2d.append(osp.join(f"{dataname}", "virt_mnt", 'query', mnt_f))
        anchor_2d_num = np.loadtxt(osp.join(mnt_query_folder, mnt_f), dtype=np.float32).shape[0]
        counting_sum += anchor_2d_num
        count_list.append(counting_sum)

    count_list = np.array(count_list)
    data_lst = {"img": img_lst,  "pose_2d": anchor_2d}
    print(f'{dataname} total {len(img_lst)} samples with {count_list[-1]} anchors')
    return data_lst, count_list

if __name__ == "__main__":
    random.seed(1016)
    np.random.seed(1016)
    parser = argparse.ArgumentParser("Evaluation for DMD")
    parser.add_argument("--prefix", type=str, default="/disk2/panzhiyu/fingerprint")
    args = parser.parse_args()
    # # the NIST series dataset
    save_file = './datasets/NISTvirt_eval.pkl'
    datasets = ['NIST_SD27']
    img_types = ['bmp']
    NIST_Seris_dict = {}
    for dataset, img_type in zip(datasets, img_types):
        datalist, count_list = create_datalist(args.prefix, dataset, img_type)
        datalist = [dict(zip(datalist, v)) for v in zip(*datalist.values())]
        NIST_Seris_dict[dataset] = {"datalist": datalist, "count_list": count_list}

    # save the data
    with open(save_file, "wb") as fp:
        pickle.dump(NIST_Seris_dict, fp)


