# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_mnt.py
@Time    :   2025/05/22 00:14:50
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2025, Zhiyu Pan, Tsinghua University. All rights reserved
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
from fptools import uni_io
from models.dataloader_densemnt import MntDataset, MatchDataset
from models.model_zoo import *
import pickle
import yaml
import queue as Queue
import threading
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import os.path as osp
import pandas
IDX = 0
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

def pad_collate_fn(batch):
    def pad_to_max_N(tensor_list, target_N):
        padded_list = []
        for tensor in tensor_list:
            current_N = tensor.shape[0]
            if current_N < target_N:
                padding_size = (0, 0, 0, target_N - current_N)  
                padded_tensor = torch.nn.functional.pad(tensor, padding_size, value=float('nan'))
            else:
                padded_tensor = tensor
            padded_list.append(padded_tensor)
        return torch.stack(padded_list, dim=0)  #
    N_list_search = [x['search_mnt'].shape[0] for x in batch]
    max_N_search = max(N_list_search)
    N_list_gallery = [x['gallery_mnt'].shape[0] for x in batch]
    max_N_gallery = max(N_list_gallery)
    search_mnt = [torch.tensor(item["search_mnt"], dtype=torch.float32) for item in batch]  
    gallery_mnt = [torch.tensor(item["gallery_mnt"], dtype=torch.float32) for item in batch]
    search_desc = [torch.tensor(item["search_desc"], dtype=torch.float32) for item in batch]
    gallery_desc = [torch.tensor(item["gallery_desc"], dtype=torch.float32) for item in batch]
    search_mask = [torch.tensor(item["search_mask"], dtype=torch.float32) for item in batch]
    gallery_mask = [torch.tensor(item["gallery_mask"], dtype=torch.float32) for item in batch]
    index = [torch.tensor(item["index"]) for item in batch]

    search_mnt = pad_to_max_N(search_mnt, max_N_search)
    gallery_mnt = pad_to_max_N(gallery_mnt, max_N_gallery)
    search_desc = pad_to_max_N(search_desc, max_N_search)
    gallery_desc = pad_to_max_N(gallery_desc, max_N_gallery)
    search_mask = pad_to_max_N(search_mask, max_N_search)
    gallery_mask = pad_to_max_N(gallery_mask, max_N_gallery)

    batch_dict = {
        "search_mnt": search_mnt,
        "gallery_mnt": gallery_mnt,
        "search_desc": search_desc,
        "gallery_desc": gallery_desc,
        "search_mask": search_mask,
        "gallery_mask": gallery_mask,
        "index": torch.stack(index)  #
    }

    return batch_dict 

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


from torch_linear_assignment import batch_linear_assignment
THRESHS = {0: 0.2, 1: 0.002, 2: 0.5} # 0 for plain, 1 for rolled, 2 for latent
def calculate_score_torchB(feat1, feat2, mask1, mask2, ndim_feat=6, N_mean=1327, Normalize=False, binary=False, f2f_type=(2, 1)):
    '''
    The function to calculate the score between two images or two set images
    '''
    feat1_dense = feat1
    feat1_mask = mask1.repeat(1, 1, ndim_feat)

    feat2_dense = feat2
    feat2_mask = mask2.repeat(1, 1, ndim_feat) 

    if binary:
        feat1_dense = (feat1_dense > 0).float()
        feat2_dense = (feat2_dense > 0).float()
        feat1_mask = (feat1_mask > THRESHS[f2f_type[0]]).float()
        feat2_mask = (feat2_mask > THRESHS[f2f_type[1]]).float()
        n12 = torch.bmm(feat1_mask, feat2_mask.transpose(1, 2))
        d12 = (
            n12
            - torch.bmm((feat1_mask * feat1_dense), (feat2_mask * feat2_dense).transpose(1, 2))
            - torch.bmm((feat1_mask * (1 - feat1_dense)), (feat2_mask * (1 - feat2_dense)).transpose(1, 2))
        )
        score = 1 - 2 * torch.where(n12 > 0, d12 / n12.clamp(min=1e-3), torch.tensor(0.5, dtype=torch.float32))
    else:
        x1 = torch.sqrt(torch.bmm(feat1_mask * feat1_dense**2, feat2_mask.transpose(1, 2)))
        x2 = torch.sqrt(torch.bmm(feat1_mask, (feat2_dense**2 * feat2_mask).transpose(1, 2)))
        x12 = torch.bmm(feat1_mask * feat1_dense, (feat2_mask * feat2_dense).transpose(1, 2))

        score = x12 / (x1 * x2).clamp(min=1e-3)

        n12 = torch.bmm(feat1_mask, feat2_mask.transpose(1, 2))
    
    if Normalize:
        score = score * torch.sqrt(n12 / N_mean)

    return score

def lsa_score_torchB(S, min_pair=4, max_pair=12, mu_p=20, tau_p=0.4):
    def sigmoid(z, mu_p, tau_p):
        return 1 / (1 + torch.exp(-tau_p * torch.clamp(z - mu_p, min=-1e10, max=100)))
    n1 = S.shape[1] # for the batch, it has been consistent
    n2 = S.shape[2] 
    B = S.shape[0]
    S2 = S
    max_n = max(n1, n2)
    new_S = torch.nn.functional.pad(1 - S2, (0, max_n - n2, 0, max_n - n1, 0 , 0), value=2)
    # replace all the torch.nan element with 2
    new_S = torch.where(torch.isnan(new_S), torch.tensor(2.0).to(new_S.device), new_S)
    batch_set_pairs = batch_linear_assignment(new_S)
    org_pair = torch.arange(new_S.shape[1])[None,...].repeat(B,1).to(new_S.device)
    pairs = torch.stack((org_pair,batch_set_pairs),dim=-1)

    pairs = pairs[:,:n1, :] # B, n1, 2
    # select the [B, n1] scores according to the pairs indexing
    scores = torch.gather(S, 2, pairs[:,:,1].unsqueeze(-1).repeat(1,1,1)).squeeze(-1)
    scores = torch.where(torch.isnan(scores), torch.tensor(0.0).to(scores.device), scores)
    scores = torch.sort(scores, dim=-1, descending=True)[0]
    n1_batch = torch.sum(~torch.isnan(S[:,:,0]), dim=-1)
    n2_batch = torch.sum(~torch.isnan(S[:,0,:]), dim=-1)
    min_number = torch.min(n1_batch, n2_batch)
    n_pair = min_pair + torch.round(sigmoid(min_number, mu_p, tau_p) * (max_pair - min_pair)).int()
    k_indices = n_pair.unsqueeze(1)
    C = scores.shape[-1]
    mask = torch.arange(C).to(k_indices.device).expand(B, C) < k_indices
    score_select = scores * mask
    score = torch.sum(score_select, dim=-1) / n_pair

    return score

def lsar_score_torchB(S, mnt1, mnt2, min_pair=4, max_pair=12, mu_p=20, tau_p=0.4):
    # S in shape (B, N1, N2), mnt1 in shape (B, N1, 3), mnt2 in shape (B, N2, 3), and not all the score or mnts are valid,
    # it has the placeholder 0 for ensuring the same size for parallel computing
    def sigmoid(z, mu_p, tau_p):
        return 1 / (1 + torch.exp(-tau_p * torch.clamp(z - mu_p, min=-1e10, max=100)))
    
    def distance_theta(theta, theta2=None):
        theta2 = theta if theta2 is None else theta2
        d = (theta[:, :, None] - theta2[:, None] + 180) % 360 - 180
        return d
    
    def distance_R(mnts):
        d = torch.rad2deg(torch.atan2(mnts[:, :, 1, None] - mnts[:, None, :, 1], mnts[:,None, :, 0] - mnts[:, :, 0, None]))
        d = (mnts[:, :, 2, None] + d + 180) % 360 - 180
        return d
    
    def distance_mnts(mnts):
        d = torch.sqrt((mnts[:, :, 0, None] - mnts[:, None, :, 0])**2 + (mnts[:, :, 1, None] - mnts[:, None, :, 1])**2)
        return d
    
    def relax_labeling(mnts1, mnts2, scores, min_number, n_pair): # min_number is the valid number of the mnts for each batch
        mu_1 = 5
        mu_2 = torch.pi / 12
        mu_3 = torch.pi / 12
        tau_1 = -8.0 / 5
        tau_2 = -30
        tau_3 = -30
        w_R = 1.0 / 2
        n_rel = 5

        D1 = torch.abs(distance_mnts(mnts1) - distance_mnts(mnts2))
        D2 = torch.deg2rad(torch.abs((distance_theta(mnts1[:, :, 2]) - distance_theta(mnts2[:,:, 2])+180) % 360 - 180))
        D3 = torch.deg2rad(torch.abs((distance_R(mnts1[:, :, :3]) - distance_R(mnts2[:, :, :3]) + 180) % 360 - 180))
        lambda_t = scores
        rp = (
            sigmoid(D1, mu_1, tau_1)
            * sigmoid(D2, mu_2, tau_2)
            * sigmoid(D3, mu_3, tau_3)
        )
        B, N, _ = rp.shape
        indices = torch.arange(N)
        rp[:, indices, indices] = 0
        rp = torch.where(torch.isnan(rp), torch.tensor(0.0).to(rp.device), rp)
        lambda_t = torch.where(torch.isnan(lambda_t), torch.tensor(0.0).to(lambda_t.device), lambda_t)
        for _ in range(n_rel): 
            lambda_t = w_R * lambda_t + (1 - w_R) * torch.sum(rp * lambda_t[:,None,:], axis=-1) / (min_number[:,None] - 1)
        efficiency = lambda_t / torch.clamp(scores, min=1e-6)
        C = efficiency.shape[1]
        efficiency = torch.where(torch.isnan(efficiency), torch.tensor(-torch.inf).to(efficiency.device), efficiency)
        _, sorted_indices = torch.sort(efficiency, dim=1, descending=True)
        lambda_t_sorted = torch.gather(lambda_t, 1, sorted_indices)
        k_indices = n_pair.unsqueeze(1) 
        mask = torch.arange(C).to(k_indices.device).expand(B, C) < k_indices
        lambda_t_sorted = lambda_t_sorted * mask
        score = torch.sum(lambda_t_sorted, dim=-1) / n_pair
        return score
    
    n1 = S.shape[1] 
    n2 = S.shape[2] 
    B = S.shape[0]
    assert n1 == mnt1.shape[1] and n2 == mnt2.shape[1]
    S2 = S
    max_n = max(n1, n2)
    new_S = torch.nn.functional.pad(1 - S2, (0, max_n - n2, 0, max_n - n1, 0 , 0), value=2)
    new_S = torch.where(torch.isnan(new_S), torch.tensor(2.0).to(new_S.device), new_S)

    if n1 < n2:
        batch_set_pairs = batch_linear_assignment(new_S)
        org_pair = torch.arange(new_S.shape[1])[None,...].repeat(B,1).to(new_S.device)
        pairs = torch.stack((org_pair,batch_set_pairs),dim=-1)
        pairs = pairs[:,:n1, :]
        scores = torch.gather(S, 2, pairs[:,:,1].unsqueeze(-1).repeat(1,1,1)).squeeze(-1)
    else:
        batch_set_pairs = batch_linear_assignment(new_S.transpose(1,2)) 
        org_pair = torch.arange(new_S.shape[2])[None,...].repeat(B,1).to(new_S.device) 
        pairs = torch.stack((batch_set_pairs, org_pair), dim=-1) 
        pairs = pairs[:,:n2, :]
        scores = torch.gather(S, 1, pairs[:,:,0].unsqueeze(-2).repeat(1,1,1)).squeeze(-2)

    n1_batch = torch.sum(~torch.isnan(S[:,:,0]), dim=-1)
    n2_batch = torch.sum(~torch.isnan(S[:,0,:]), dim=-1)
    min_number = torch.min(n1_batch, n2_batch)
    n_pair = min_pair + torch.round(sigmoid(min_number, mu_p, tau_p) * (max_pair - min_pair)).int()
    mnt1_order = torch.gather(mnt1, 1, pairs[:,:,0].unsqueeze(-1).repeat(1,1,3))
    mnt2_order = torch.gather(mnt2, 1, pairs[:,:,1].unsqueeze(-1).repeat(1,1,3)) 
    score = relax_labeling(mnt1_order, mnt2_order, scores, min_number, n_pair) 
    return score
class Evaluator:
    def __init__(self, params, gpus, is_load=False, is_relax=False, Normalize=False, Binary=False) -> None:
        # params = self.load_config_file(cfg_path)
        self.update_attrs(params)
        self.gpus = gpus
        self.relax = is_relax
        self.binary = Binary
        self.extract_time = 0
        self.matching_time = 0
        print(yaml.dump(params, allow_unicode=True, default_flow_style=False))
        self.params = params
        self.Normalize = Normalize
        if self.Normalize:
            postfix = ''
        else:
            postfix = '_nonorm'
        logging.info(f"Current checkpoint: {self.eval_path}")
        self.main_dev = torch.device(f"cuda:{self.gpus[0]}")
        if is_load:
            # model
            try:
                model = DMD(
                    ndim_feat=self.ndim_feat, pos_embed=self.pos_embed, tar_shape=self.tar_shape, input_norm = self.input_norm,
                )
            except Exception as ex:
                raise ValueError(ex)
            self.model = model
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
                os.path.join("./datasets",self.eval_dataset+'.pkl'), 
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
                    start = time.time()
                    features = outputs["feature"].cpu().numpy()
                    self.extract_time += time.time() - start
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
        # report the extract speed
        samples_num = len(search_imgs) + len(gallery_imgs)
        logging.info(f"totally {samples_num} images, extract speed: {(self.extract_time/samples_num):.2e}s/sample")

    def calculate_scores(self):
        '''
        Using the torch to calculate the scores between the gallery and search images
        '''
        global IDX
        print(f'start calculating the scores on {self.search_folder}')
        search_imgs = os.listdir(self.search_folder)
        search_imgs.sort()
        gallery_imgs = os.listdir(self.gallery_folder)
        gallery_imgs.sort()
        # calculate the scores
        score_matrix = np.zeros((len(search_imgs), len(gallery_imgs)))
        # create the dataset for calculating the scores
        match_dataset = MatchDataset(self.save_folder)
        workers = min(16, self.batch_size // 2)
        matchloader = DataLoaderX(
            dataset=match_dataset,
            batch_size=256,
            collate_fn=pad_collate_fn,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        with torch.no_grad():
            for item in tqdm(matchloader): #
                search_feat = item["search_desc"].to(self.main_dev)
                gallery_feat = item["gallery_desc"].to(self.main_dev)
                search_mask = item["search_mask"].to(self.main_dev)
                gallery_mask = item["gallery_mask"].to(self.main_dev)
                search_mnt = item["search_mnt"].to(self.main_dev)
                gallery_mnt = item["gallery_mnt"].to(self.main_dev)
                index_pair = item["index"]
                start = time.time()
                scores = calculate_score_torchB(search_feat, gallery_feat, search_mask, gallery_mask, ndim_feat=self.ndim_feat*2,  Normalize=self.Normalize, N_mean=5, binary=self.binary, f2f_type=(2,1))
                if self.relax:
                    score = lsar_score_torchB(scores, search_mnt, gallery_mnt)
                else:
                    score = scores
                self.matching_time += time.time() - start
                # assign the score into the score_matrix
                # index_pair is in [B,2]
                score_matrix[index_pair[:,0], index_pair[:,1]] = score.cpu().numpy()
        # save the score_matrix
        df = pandas.DataFrame(score_matrix)
        df.columns = gallery_imgs
        df.index = search_imgs
        df.to_csv(self.score_file)
        # report the matching speed
        logging.info(f"Matching time: {(self.matching_time/score_matrix.size):.2e}s")

    def eval_metric(self):
        # generate the score matrix
        df = pd.read_csv(self.score_file)
        score_mat = df.iloc[:,1:].values
        score_mat = np.where(np.isnan(score_mat), 0, score_mat) # invalid value process
        # read the genuine pairs
        genuine_pairs_file = osp.join(self.prefix, self.eval_dataset, 'genuine_pairs.txt')
        # load the name from the file
        with open(genuine_pairs_file, 'r') as f:
            genuine_pairs = f.readlines()
        genuine_pairs = [x.strip().split(',') for x in genuine_pairs]
        genuine_pairs = np.array(genuine_pairs)
        # load the whole name
        search_imgs = os.listdir(os.path.join(self.prefix, self.eval_dataset, 'image','query'))
        search_imgs.sort()
        search_imgs = [x.split('.')[0] for x in search_imgs]
        gallery_imgs = os.listdir(os.path.join(self.prefix, self.eval_dataset, 'image','gallery'))
        gallery_imgs.sort()
        gallery_imgs = [x.split('.')[0] for x in gallery_imgs]
        target_matrix = np.zeros((len(search_imgs), len(gallery_imgs)))
        score_matrix = np.zeros((len(search_imgs), len(gallery_imgs))) 
        gallery_name = list(df.columns[1:])
        gallery_name = [x.split('.')[0] for x in gallery_name]
        search_name = list(df.iloc[:,0].values)
        search_name = [x.split('.')[0] for x in search_name]

        # fill the score_matrix from the score_mat
        search_idx= [search_imgs.index(x) for x in search_name]
        gallery_idx = [gallery_imgs.index(x) for x in gallery_name]
        ixgrid = np.ix_(search_idx, gallery_idx)
        score_matrix[ixgrid] = score_mat
        # fill the target_matrix
        search_idx = [search_imgs.index(x) for x in genuine_pairs[:,0]]
        gallery_idx = [gallery_imgs.index(x) for x in genuine_pairs[:,1]]
        target_matrix[search_idx, gallery_idx] = 1

        rank1_general(score_matrix, target_matrix, dataname=self.eval_dataset)
        TAR_flatten(score_matrix, target_matrix, dataname=self.eval_dataset)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation for DMD")
    parser.add_argument("--eval_dataset", "-d", type=str, required=True, default="NIST_SD27", help="The dataset for evaluation")
    parser.add_argument("--gpus", "-g", default=[0], type=int, nargs="+")
    parser.add_argument("--extract", "-e", action="store_true")
    parser.add_argument("--binary", "-b", action="store_true")
    parser.add_argument("--method", "-m", type=str, required=True, default='DMD',  help="The DMD version for evaluation")
    args = parser.parse_args()
    yaml_path = f'{args.method}.yaml' 
    params = edict(yaml.safe_load(open(yaml_path, "r")))
    params.update(vars(args))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logging.info(f"loading training profile from {yaml_path}")
    t = Evaluator(params, args.gpus, is_load=args.extract, is_relax=True, Normalize=args.score_norm, Binary=args.binary) 
    logging.info(f"Start to evaluate the dataset: {t.eval_dataset}")
    if args.extract:
        t.extract_feat() # extract the features for each patch and save them into the corresponding folder
        t.concatenate_feat() # concatenate the features of the same image into one file
    # calculate the scores
    t.calculate_scores() # calculate the scores between the gallery and search images
    t.eval_metric() # simple evaluation function, need to be modified for the specific dataset