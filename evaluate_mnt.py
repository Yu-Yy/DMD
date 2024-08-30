# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_mnt.py
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2023, Zhiyu Pan, Tsinghua University. All rights reserved
'''
import argparse
import logging
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import time
from utils.get_eval_metric import *
from fptools import  uni_io
from models.dataloader_densemnt import MntDataset  # TODO:
from models.model_zoo import *
from scipy.spatial import  distance
import pickle
import yaml
import queue as Queue
import threading
from torch.utils.data import DataLoader

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Manager

multiprocessing.set_start_method('spawn', force=True) 

import pandas
class BackgroundGenerator(threading.Thread):
    # def __init__(self, generator, local_rank, max_prefetch=4):
    def __init__(self, generator, max_prefetch=4):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        # self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        # torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
    
from torch.utils.data.dataloader import default_collate
def my_collate_fn(batch):
    batch = list(filter(lambda x: x['img_r'] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=self.batch_size)

def get_matched_parameters(model, keys):
    for name, module in model.named_children():
        if name in keys:
            for _, param in module.named_parameters(recurse=True):
                if param.requires_grad:
                    yield param


def get_non_matched_parameters(model, keys):
    for name, module in model.named_children():
        if name not in keys:
            for _, param in module.named_parameters(recurse=True):
                if param.requires_grad:
                    yield param

def if_find(k, keys=[]):
    for c in keys:
        if c in k:
            return True
    return False


def load_model(model, ckp_path, keys=[], by_name=False):
    def remove_module_string(k):
        items = k.split(".")
        idx = items.index("module")
        items = items[0:idx] + items[idx + 1 :]
        return ".".join(items)

    if isinstance(ckp_path, str):
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        try:
            ckp_model_dict = ckp["model"]
        except:
            ckp_model_dict = ckp
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {remove_module_string(k): v for k, v in ckp_model_dict.items()}

    if len(keys):
        ckp_model_dict = {k: v for k, v in ckp_model_dict.items() if if_find(k, keys)}

    if by_name:
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in ckp_model_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        ckp_model_dict = model_dict

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)

def process_checkpoint(ckp):
    new_ckp = {}
    for k, v in ckp.items():
        if "_branch" in k:
            new_k = k.replace("_branch", "")
            new_ckp[new_k] = v
        else:
            new_ckp[k] = v
    return new_ckp



class Evaluator:
    def __init__(self,cfg_path, gpus, is_load=False, is_relax=False, Normalize=False, Binary=False) -> None:
        params = self.load_config_file(cfg_path)
        self.update_attrs(params)
        self.gpus = gpus
        self.relax = is_relax
        self.binary = Binary
        print(yaml.dump(params, allow_unicode=True, default_flow_style=False))
        self.params = params
        self.Normalize = Normalize
        if self.Normalize:
            postfix = ''
        else:
            postfix = '_nonorm'
        logging.info(f"Current checkpoint: {self.eval_path}")
        if is_load:
            # model
            try:
                model = DensePrintB(
                    ndim_feat=self.ndim_feat, pos_embed=self.pos_embed, tar_shape=self.tar_shape,
                )
            except Exception as ex:
                raise ValueError(ex)
            self.model = model
            self.main_dev = torch.device(f"cuda:{self.gpus[0]}")
            # load the model params
            logging.info(f"Resuming existing trained model: {self.eval_path}")
            model_name = osp.join("./", self.eval_path, "best_model.pth.tar")
            ckp = torch.load(model_name)
            logging.info(f"Resuming existing trained model: {model_name}")
            if 'model' in ckp.keys():
                ckp = ckp['model']
            # check the model's state_dict
            self.model.load_state_dict(ckp)
            # using the gpus
            self.model = nn.DataParallel(self.model, device_ids=self.gpus) # set the gpus into the model
            cudnn.benchmark = True 
            self.model.to(self.main_dev)
            self.model.eval() # set the model into eval mode for evaluation
            self.test_dataset = MntDataset(
                self.prefix,
                f"./datasets/NISTmnt_eval.pkl", 
                img_ppi=self.img_ppi,
                tar_shape=self.tar_shape,
                middle_shape=self.middle_shape,
                dataname=self.eval_dataset,
            )
            logging.info(f"Dataset: {self.eval_dataset}")
            workers = min(16, self.batch_size // 2)
            self.evalloader = DataLoaderX(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )


        save_folder = osp.join(self.prefix, self.eval_dataset, f'DMD_{self.ndim_feat}') 

        self.save_folder = save_folder
        uni_io.mkdir(save_folder)
        self.gallery_folder = osp.join(save_folder, 'gallery')
        uni_io.mkdir(self.gallery_folder)
        self.search_folder = osp.join(save_folder, 'search')
        uni_io.mkdir(self.search_folder)

        if self.relax:
            if self.binary:
                self.score_file = os.path.join(self.search_folder, '..', f'score_matrix_relax_binary{postfix}.csv')
            else:
                self.score_file = os.path.join(self.search_folder, '..', f'score_matrix_relax{postfix}.csv')

        else:
            if self.binary:
                self.score_file = os.path.join(self.search_folder, '..', f'score_matrix_binary{postfix}.csv')
            else:
                self.score_file = os.path.join(self.search_folder, '..', f'score_matrix{postfix}.csv')

    def save_config_file(self):
        if not osp.exists(osp.join(self.ckp_dir, "configs.yaml")):
            with open(osp.join(self.ckp_dir, "configs.yaml"), "w") as fp:
                yaml.safe_dump(self.params, fp)

    def load_config_file(self, config_file):
        return yaml.safe_load(open(config_file, "r"))

    def update_attrs(self, kwargs):
        self.__dict__.update(kwargs)

    def extract_feat(self):
        with tqdm(
            total=len(self.test_dataset), desc=f"Eval", ncols=min(160, int(os.popen("stty size").read().split()[1]))
        ) as pbar:
            for iterx, item in enumerate(self.evalloader):
                img = item["img_r"].to(self.main_dev)
                mnts = item["minu_r"]
                name = item["name"]
                indices = item["index"]
                with torch.no_grad():
                    outputs = self.model.module.get_embedding(img)
                    features = outputs["feature"].cpu().numpy()
                    masks = outputs["mask"].cpu().numpy()
                    # save the feature and mask
                    for i in range(len(name)):
                        save_dict = {}
                        feat = features[i]
                        mask = masks[i]
                        name_i = name[i]
                        mnt = mnts[i]
                        index = indices[i]
                        save_dict['feat'] = feat
                        save_dict['mask'] = mask
                        save_dict['mnt'] = mnt.numpy()
                        _, name_s = name_i.split('/')[0],name_i.split('/')[-1]
                        if name_i.split('/')[1] == 'gallery': # gallery folder
                            uni_io.mkdir(osp.join(self.gallery_folder, name_s))
                            save_path = osp.join(self.gallery_folder, name_s, f'{index}.pkl')
                        else: # search folder
                            uni_io.mkdir(osp.join(self.search_folder, name_s))
                            save_path = osp.join(self.search_folder, name_s, f'{index}.pkl')
                        with open(save_path, 'wb') as f:
                            pickle.dump(save_dict, f)
                pbar.update(self.batch_size)
        logging.info("Done.")

    def concatenate_feat(self):
        '''
        let all the extracted features of one images into one files
        '''
        search_imgs = os.listdir(self.search_folder)
        search_imgs = [x for x in search_imgs if not x.endswith('.pkl')]
        search_imgs.sort()
        gallery_imgs = os.listdir(self.gallery_folder)
        gallery_imgs = [x for x in gallery_imgs if not x.endswith('.pkl')]
        gallery_imgs.sort()

        for img in tqdm(search_imgs):
            img_path = osp.join(self.search_folder, img)
            img_feats = os.listdir(img_path)
            img_feats.sort()
            img_org_fs = [osp.join(img_path, feat) for feat in img_feats]
            img_org_fs = [pickle.load(open(feat, 'rb')) for feat in img_org_fs]
            feats = [feat['feat'][None,...] for feat in img_org_fs]
            feats = np.concatenate(feats, axis=0)
            masks = [feat['mask'][None,...] for feat in img_org_fs]
            masks = np.concatenate(masks, axis=0)
            mnts = [feat['mnt'][None,...] for feat in img_org_fs]
            mnts = np.concatenate(mnts, axis=0)
            save_dict = {}
            save_dict['feat'] = feats
            save_dict['mask'] = masks
            save_dict['mnt'] = mnts
            save_path = osp.join(self.search_folder, img + '.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
            # delete the original folder and its files using os
            os.system(f'rm -rf {img_path}')

        for img in tqdm(gallery_imgs):
            img_path = osp.join(self.gallery_folder, img)
            img_feats = os.listdir(img_path)
            img_feats.sort()
            img_org_fs = [osp.join(img_path, feat) for feat in img_feats]
            img_org_fs = [pickle.load(open(feat, 'rb')) for feat in img_org_fs]
            feats = [feat['feat'][None,...] for feat in img_org_fs]
            feats = np.concatenate(feats, axis=0)
            masks = [feat['mask'][None,...] for feat in img_org_fs]
            masks = np.concatenate(masks, axis=0)
            mnts = [feat['mnt'][None,...] for feat in img_org_fs]
            mnts = np.concatenate(mnts, axis=0)
            save_dict = {}
            save_dict['feat'] = feats
            save_dict['mask'] = masks
            save_dict['mnt'] = mnts
            save_path = osp.join(self.gallery_folder, img + '.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
            # delete the original folder and its files using os
            os.system(f'rm -rf {img_path}')


    def calculate_scores(self, num_process=32):
        '''
        using multiple processes to calculate the score
        '''
        print(f'start calculating the scores on {self.search_folder}')
        manager = Manager()
        search_imgs = os.listdir(self.search_folder)
        search_imgs.sort()
        gallery_imgs = os.listdir(self.gallery_folder)
        gallery_imgs.sort()
        # load the corresponding features and masks
        search_ = []
        gallery_ = []
        # loading the features and masks
        for i in tqdm(range(len(search_imgs))):
            search_.append(pickle.load(open(osp.join(self.search_folder, search_imgs[i]), 'rb')))
        for i in tqdm(range(len(gallery_imgs))):
            gallery_.append(pickle.load(open(osp.join(self.gallery_folder, gallery_imgs[i]), 'rb')))
        
        tasks = [(i, search_[i], gallery_) for i in range(len(search_imgs))]
        pool = Pool(num_process)
        jobs = []
        score_matrix_pre = manager.list([0 for i in range(len(search_imgs))])
        # divide the tasks into several parts
        for worker_id in range(num_process):
            task_process = tasks[worker_id::num_process]
            job = pool.apply_async(self.task_func, (task_process, score_matrix_pre))
            jobs.append(job)
        
        with tqdm(total=len(tasks)) as pbar:
            while True:
                complete_tasks = sum([i is not 0 for i in score_matrix_pre])
                pbar.update(complete_tasks - pbar.n)
                if complete_tasks == len(tasks):
                    break
                time.sleep(1)
        pool.close()
        pool.join()
        score_matrix = np.concatenate(score_matrix_pre, axis=0)
        score_df = pandas.DataFrame(score_matrix)
        score_df.columns = gallery_imgs
        score_df.index = search_imgs
        score_df.to_csv(self.score_file) # using the .. to return the last folder

    def task_func(self, tasks, result_array):
        ndim_feat = self.ndim_feat * 2 # concatenate the feature
        for task in tasks:
            i, search_, gallery_imgs = task
            search_feat = search_['feat']
            search_mask = search_['mask']
            search_mnt = search_['mnt'] # using the mnt to add geometric constraint
            score_sub_matrix = np.zeros((1, len(gallery_imgs)))
            for j, gallery_ in enumerate(gallery_imgs):
                gallery_feat = gallery_['feat']
                gallery_mask = gallery_['mask']
                gallery_mnt = gallery_['mnt']
                score_matrix = calculate_score(search_feat, gallery_feat, search_mask, gallery_mask, 
                                               p2p=False, ndim_feat=ndim_feat, Normalize=self.Normalize,
                                               binary=self.binary, f2f_type=(2, 1))
                # locally fuse the score_matrix, do not need to do the seriously one to one 
                try:
                    if self.relax:
                        score,_ = lsar_score(score_matrix, search_mnt, gallery_mnt)
                    else:
                        score,_ = lsa_score(score_matrix)
                except:
                    score = 0
                score_sub_matrix[0, j] = score

            result_array[i] = score_sub_matrix

    def eval_metric(self, th=0.0001):
        '''
        calculate the rank1 and tar@far=0.1%
        '''
        score_file = os.path.join(self.score_file)
        score_pd = pandas.read_csv(score_file)
        score_matrix = score_pd.iloc[:, 1:].values
        rank1 = rank1_value(score_matrix, data_name=self.eval_dataset)
        wrap_1 = TAR(score_matrix, th=th, data_name=self.eval_dataset)
        return rank1, wrap_1
    

THRESHS = {0: 0.2, 1: 0.002, 2: 0.5} # 0 for plain, 1 for rolled, 2 for latent
def calculate_score(feat1, feat2, mask1, mask2, ndim_feat=6, p2p=True, N_mean=1327, Normalize=False, binary=False, f2f_type=(2, 1)):
    '''
    The function to calculate the score between two images or two set images
    '''
    feat1_dense = feat1
    feat1_mask = np.tile(mask1, (1, ndim_feat))

    feat2_dense = feat2
    feat2_mask = np.tile(mask2, (1, ndim_feat))

    if binary:
        if p2p:
            feat1_dense = feat1_dense > 0
            feat2_dense = feat2_dense > 0
            feat1_mask = feat1_mask > THRESHS[f2f_type[0]]
            feat2_mask = feat2_mask > THRESHS[f2f_type[1]]
            d12 = (feat1_mask & (feat1_dense ^ feat2_dense) & feat2_mask).sum(1)
            n12 = (feat1_mask & feat2_mask).sum(1)
            score = 1 - 2 * np.where(n12 > 0, d12 / n12.clip(1e-3, None), 0.5)
        else:
            feat1_dense = (feat1_dense > 0).astype(np.float32)
            feat2_dense = (feat2_dense > 0).astype(np.float32)
            feat1_mask = (feat1_mask > THRESHS[f2f_type[0]]).astype(np.float32)
            feat2_mask = (feat2_mask > THRESHS[f2f_type[1]]).astype(np.float32)
            n12 = np.matmul(feat1_mask, feat2_mask.T)
            d12 = (
                n12
                - np.matmul((feat1_mask * feat1_dense), (feat2_mask * feat2_dense).T)
                - np.matmul((feat1_mask * (1 - feat1_dense)), (feat2_mask * (1 - feat2_dense)).T)
            )
            score = 1 - 2 * np.where(n12 > 0, d12 / n12.clip(1e-3, None), 0.5)
    else:
        if p2p:
            f1 = feat1_mask * feat1_dense**2 * feat2_mask
            f2 = feat1_mask * feat2_dense**2 * feat2_mask
            f12 = feat1_mask * feat1_dense * feat2_dense * feat2_mask
            
            x1 = np.sqrt(f1.sum(1))
            x2 = np.sqrt(f2.sum(1))
            x12 = f12.sum(1)
        else:
            x1 = np.sqrt(np.matmul(feat1_mask * feat1_dense**2, feat2_mask.T))
            x2 = np.sqrt(np.matmul(feat1_mask, (feat2_dense**2 * feat2_mask).T))
            x12 = np.matmul(feat1_mask * feat1_dense, (feat2_mask * feat2_dense).T)

        score = x12 / (x1 * x2).clip(1e-3, None)
        if p2p:
            n12 = (feat1_mask * feat2_mask).sum(1)
        else:
            n12 = np.matmul(feat1_mask, feat2_mask.T)
    
    if Normalize:
        score = score * np.sqrt(n12 / N_mean)

    return score

def sigmoid(z, mu_p, tau_p):
    return 1 / (1 + np.exp(-tau_p * np.clip(z - mu_p, None, 100)))

def k_largest_index_argpartition(a, k):
    idx = np.argpartition(-a.ravel(), k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))

def distance_theta(theta, theta2=None):
    theta2 = theta if theta2 is None else theta2
    d = (theta[:, None] - theta2[None] + 180) % 360 - 180
    return d

def distance_R(mnts):
    d = np.rad2deg(np.arctan2(mnts[:, 1, None] - mnts[None, :, 1], mnts[None, :, 0] - mnts[:, 0, None]))
    d = (mnts[:, 2, None] + d + 180) % 360 - 180
    return d

def lsa_score(S, min_pair=4, max_pair=12, mu_p=20, tau_p=0.4):
    '''
    S: the similarity matrix
    '''
    n1 = S.shape[0]
    n2 = S.shape[1]
    n_pair = min_pair + np.rint(
        sigmoid(min(n1, n2), mu_p, tau_p) * (max_pair - min_pair)
    ).astype(int)
    max_n = max(n1, n2)
    new_S = np.pad(1 - S, ((0, max_n - n1), (0, max_n - n2)), constant_values=2)
    new_S = np.pad(new_S, (0, max_n - n_pair), constant_values=2)
    new_S[:max_n, n_pair - max_n :] = -1
    pairs = np.column_stack(linear_sum_assignment(new_S))
    pairs = pairs[(pairs[:, 0] < n1) & (pairs[:, 1] < n2)]
    p_scores = S[pairs[:, 0], pairs[:, 1]]
    score = np.mean(p_scores)
    return score, pairs

def lsar_score(S, mnt1, mnt2, min_pair=4, max_pair=12, mu_p=20, tau_p=0.4, with_pose=False):
    n1 = S.shape[0]
    n2 = S.shape[1]
    assert n1 == mnt1.shape[0] and n2 == mnt2.shape[0]
    if with_pose:
        S2 = S * compute_pose_constaints(mnt1, mnt2)[0]
    else:
        S2 = S
    n_pair = min(n1, n2)
    n_pair = min(n1, n2)
    max_n = max(n1, n2)
    new_S = np.pad(1 - S2, ((0, max_n - n1), (0, max_n - n2)), constant_values=2)
    pairs = np.column_stack(linear_sum_assignment(new_S))
    pairs = pairs[(pairs[:, 0] < n1) & (pairs[:, 1] < n2)]
    scores = S[pairs[:, 0], pairs[:, 1]]
    n_pair = min_pair + np.rint(
        sigmoid(min(n1, n2), mu_p, tau_p) * (max_pair - min_pair)
    ).astype(int)
    score, pairs, p_scores = relax_labeling(mnt1, mnt2, scores, pairs, n_pair, with_pose)
    return score, pairs

def compute_pose_constaints(mnts1, mnts2):
    mu_loc = 60
    tau_loc = -0.1
    mu_ang = np.pi / 9
    tau_ang = mu_loc * tau_loc / mu_ang
    D1 = distance.cdist(mnts1[:, :2], mnts2[:, :2])
    D2 = np.deg2rad(distance_theta(mnts1[:, 2], mnts2[:, 2]))
    dist = sigmoid(D1, mu_loc, tau_loc) * sigmoid(D2, mu_ang, tau_ang)
    th = sigmoid(mu_loc, mu_loc, tau_loc) * sigmoid(mu_ang, mu_ang, tau_ang)
    mask = np.where(dist >= th, 1, 0)
    return mask, dist

def relax_labeling(mnts1, mnts2, scores, pairs, n_pair, with_pose=False): #
    
    mu_1 = 5
    mu_2 = np.pi / 12
    mu_3 = np.pi / 12
    tau_1 = -8.0 / 5
    tau_2 = -30
    tau_3 = -30
    w_R = 1.0 / 2
    n_rel = 5

    mnts1 = mnts1[pairs[:, 0]]
    mnts2 = mnts2[pairs[:, 1]]
    D1 = np.abs(distance.squareform(distance.pdist(mnts1[:, :2])) - distance.squareform(distance.pdist(mnts2[:, :2])))
    D2 = np.deg2rad(np.abs((distance_theta(mnts1[:, 2]) - distance_theta(mnts2[:, 2]) + 180) % 360 - 180))  
    D3 = np.deg2rad(np.abs((distance_R(mnts1[:, :3]) - distance_R(mnts2[:, :3]) + 180) % 360 - 180))
    lambda_t = scores
    rp = (
        sigmoid(D1, mu_1, tau_1)
        * sigmoid(D2, mu_2, tau_2)
        * sigmoid(D3, mu_3, tau_3)
    )
    if with_pose:
        dist = compute_pose_constaints(mnts1, mnts2)[1]
        rp *= dist
    np.fill_diagonal(rp, 0)
    n_R = len(pairs)
    for _ in range(n_rel): # iteration for relexing
        lambda_t = w_R * lambda_t + (1 - w_R) * np.sum(rp * lambda_t[None], axis=1) / (n_R - 1)
    efficiency = lambda_t / scores.clip(1e-6, None)
    selected = k_largest_index_argpartition(efficiency, n_pair)[:, 0]
    pairs = pairs[selected]
    score = np.mean(lambda_t[selected])
    return score, pairs, lambda_t[selected]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    term = 'latent'
    processes = {
        'rolled': 24,
        'latent': 6,
    }
    ths = {
        'rolled': 0.0001,
        'latent': 0.001,
    }
    parser = argparse.ArgumentParser("Evaluation for DMD")
    parser.add_argument("--gpus", "-g", default=[0], type=int, nargs="+")
    args = parser.parse_args()

    yaml_path = f'DMD.yaml' 
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logging.info(f"loading training profile from {yaml_path}")

    t = Evaluator(yaml_path, args.gpus, is_load=True, is_relax=True, Normalize=True, Binary=False) # pose is not necessary
    logging.info(f"Start to evaluate the dataset: {t.eval_dataset}")
    t.extract_feat() # extract the features for each patch and save them into the corresponding folder
    t.concatenate_feat() # concatenate the features of the same image into one file
    t.calculate_scores(num_process=processes[term]) # calculate the scores between the gallery and search images
    t.eval_metric(th=ths[term]) # simple evaluation function, need to be modified for the specific dataset
