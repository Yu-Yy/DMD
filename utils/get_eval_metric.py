# -*- encoding: utf-8 -*-
'''
@File    :   get_eval_metric.py
@Time    :   2025/05/22 00:17:03
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2025, Zhiyu Pan, Tsinghua University. All rights reserved
'''
import numpy as np
import os
import pandas as pd
from sklearn import metrics

def rank1_general(score_mat, target, dataname):
    all_rank = np.argsort(score_mat, axis=1)
    rank_value = np.arange(1, 21)
    cmc = np.zeros(20)
    for i in range(20):
        rank = all_rank[:, -rank_value[i]:]
        rank_target = np.take_along_axis(target, rank,axis=1)
        correct = np.any(rank_target, axis=1)
        cmc[i] = np.sum(correct) / len(correct)
    print(f"{dataname}: rank1 value is {cmc[0]*100:.2f}%")
    # and the rank10
    print(f"{dataname}: rank10 value is {cmc[9]*100:.2f}%")
    return cmc

def TAR_flatten(score_mat, target, dataname=None):
    far, tar, thresholds = metrics.roc_curve(target.flatten(), score_mat.flatten())
    far_01 = np.where(far <= 0.001)[0][-1]
    tar_01 = tar[far_01]
    far_001 = np.where(far <= 0.0001)[0][-1]
    tar_001 = tar[far_001]
    print(f"{dataname}: tar@far=0.1% is {tar_01 *100: .2f} %, tar@far=0.01% is {tar_001*100:.2f}%")
    fmr, fnmr, _ = metrics.det_curve(target.flatten(), score_mat.flatten())
    return far, tar, thresholds, tar_01, fmr, fnmr