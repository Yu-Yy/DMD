import numpy as np
import os
import pandas as pd
from sklearn import metrics

# def rank1_value(score_mat, target, data_name="NIST14"):
#     rank1 = np.argmax(score_mat, axis=1)
#     target = np.arange(len(rank1))
#     # judge if the rank1 is the correct value
#     correct = rank1 == target
#     # get the rank1 value
#     rank1_value = np.sum(correct) / len(rank1)

#     # get the rank10 value
#     rank10 = np.argsort(score_mat, axis=1)[:,-10:]
#     # judge if the rank10 is the correct value
#     correct = np.any(rank10 == target.reshape(-1,1), axis=1)
#     # get the rank10 value
#     rank10_value = np.sum(correct) / len(rank10)

#     print(f"{data_name}: rank1 value is {rank1_value*100:.2f}%, rank10 value is {rank10_value*100:.2f}%")
#     return rank1_value, rank10_value

def rank1_general(score_mat, target, dataname):
    all_rank = np.argsort(score_mat, axis=1)
    rank_value = np.arange(1, 21)
    cmc = np.zeros(20)
    # target = query_id[:, None] == gallery_id[None]
    for i in range(20):
        rank = all_rank[:, -rank_value[i]:]
        # rank 从 target 找索引值
        rank_target = np.take_along_axis(target, rank,axis=1)
        correct = np.any(rank_target, axis=1)
        # import pdb;pdb.set_trace()
        # correct = np.any(rank==target, axis=1)
        cmc[i] = np.sum(correct) / len(correct)
    print(f"{dataname}: rank1 value is {cmc[0]*100:.2f}%")
    # and the rank10
    print(f"{dataname}: rank10 value is {cmc[9]*100:.2f}%")
    return cmc

def TAR_flatten(score_mat, target, dataname=None):
    # calculate the tar@far=0.1% 
    far, tar, thresholds = metrics.roc_curve(target.flatten(), score_mat.flatten())
    # find the far=0.1% and get the tar
    far_01 = np.where(far <= 0.001)[0][-1]
    tar_01 = tar[far_01]
    far_001 = np.where(far <= 0.0001)[0][-1]
    tar_001 = tar[far_001]
    print(f"{dataname}: tar@far=0.1% is {tar_01 *100: .2f} %, tar@far=0.01% is {tar_001*100:.2f}%")
    fmr, fnmr, _ = metrics.det_curve(target.flatten(), score_mat.flatten())
    return far, tar, thresholds, tar_01, fmr, fnmr
# def TAR(score_mat, th=0.0001, data_name="NIST14"):
#     # calculate the tar@far=0.1% 
#     target = np.eye(score_mat.shape[0])
#     # extend the zeros to the axis 1
#     target = np.concatenate([target, np.zeros((score_mat.shape[0], score_mat.shape[1] - score_mat.shape[0]))], axis=1)
#     # flatten the score_mat and target to 1d
#     far, tar, thresholds = metrics.roc_curve(target.flatten(), score_mat.flatten())
#     # find the far=0.1% and get the tar
#     far_01 = np.where(far <= th)[0][-1]
#     tar_01 = tar[far_01] # 
#     print(f"{data_name}: tar@far={th * 100}% is {tar_01 * 100:.2f}%")
#     return tar_01 #far, tar, thresholds

def draw_roc(*args):
    labels_lab = ["DensePrint", "DensePrint_mini"]
    import matplotlib.pyplot as plt
    input_num = len(args)
    for i in range(input_num):
        plt.plot(args[i][0], args[i][1], label=labels_lab[i])
    plt.legend()
    plt.savefig("roc.png") 
