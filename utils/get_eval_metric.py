import numpy as np
import os
import pandas as pd
from sklearn import metrics

def rank1_value(score_mat, data_name="NIST14"):
    # read the csv file and get the rank1 value
    # df = pd.read_csv(score_file)
    # # get the matrix value except the index and column title
    # score_mat = df.iloc[:,1:].values
    # get the rank1 value
    rank1 = np.argmax(score_mat, axis=1)
    target = np.arange(len(rank1))
    # judge if the rank1 is the correct value
    correct = rank1 == target
    # get the rank1 value
    rank1_value = np.sum(correct) / len(rank1)

    # get the rank10 value
    rank10 = np.argsort(score_mat, axis=1)[:,-10:]
    # judge if the rank10 is the correct value
    correct = np.any(rank10 == target.reshape(-1,1), axis=1)
    # get the rank10 value
    rank10_value = np.sum(correct) / len(rank10)

    print(f"{data_name}: rank1 value is {rank1_value*100:.2f}%, rank10 value is {rank10_value*100:.2f}%")
    return rank1_value, rank10_value

def TAR(score_mat, th=0.0001, data_name="NIST14"):
    # df = pd.read_csv(score_file)
    # score_mat = df.iloc[:,1:].values  
    # calculate the tar@far=0.1% 
    target = np.eye(score_mat.shape[0])
    # extend the zeros to the axis 1
    target = np.concatenate([target, np.zeros((score_mat.shape[0], score_mat.shape[1] - score_mat.shape[0]))], axis=1)
    # flatten the score_mat and target to 1d
    far, tar, thresholds = metrics.roc_curve(target.flatten(), score_mat.flatten())
    # find the far=0.1% and get the tar
    far_01 = np.where(far <= th)[0][-1]
    tar_01 = tar[far_01] # 
    print(f"{data_name}: tar@far={th * 100}% is {tar_01 * 100:.2f}%")
    return tar_01 #far, tar, thresholds

    # for i, scores in enumerate(score_mat):
    #     # label is i idx is 1 and other is 0
    #     label = np.zeros(len(scores))
    #     label[i] = 1
    #     far, tar, thresholds = metrics.roc_curve(label, scores)
    #     # find the far=0.1% and get the tar
    #     far_01 = np.where(far <= 0.001)[0][-1]
    #     tar_01 = tar[far_01]

def draw_roc(*args):
    labels_lab = ["DensePrint", "DensePrint_mini"]
    import matplotlib.pyplot as plt
    input_num = len(args)
    for i in range(input_num):
        plt.plot(args[i][0], args[i][1], label=labels_lab[i])
    plt.legend()
    plt.savefig("roc.png") 


if __name__ == '__main__':
    # score_file = "/disk2/panzhiyu/fingerprint/NIST14/fingernet/DensePrint_feat_score_matrix.csv"
    # # rank1_value(score_file)
    # wrap_1 = TAR(score_file)
    score_file = "/disk2/panzhiyu/fingerprint/NIST27/fingernet/DensePrint_womnt_NIST4_trip_feat_score_matrix.csv"
    rank1 = rank1_value(score_file)
    wrap_2 = TAR(score_file, th=0.001) # th is 0.0001 for plain and roll and 0.001 for latent
    # draw_roc(wrap_1, wrap_2)