from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip
import torch
import torch.nn as nn
from torch.nn import Module

from clip_debiasing import Dotdict
from torch.utils.data import DataLoader

import torch.nn.functional as F

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np
from clip_debiasing.datasets import IATDataset, FairFace, UTKface

from collections import defaultdict
from tqdm import tqdm

def Mixed_KSG(x,y,k=5): # source: https://github.com/wgao9/mixed_KSG/tree/master
    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])   	
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
    return ans

class CLIP_clipped(nn.Module):

    def __init__(self, arch_str, device, hidden_dim, m, attribute='gender', trained_set='fairface', **_kwargs,):
        super().__init__()

        self.dtype = torch.float32

        self.clip, self.preprocess = clip.load(arch_str, device=device)
        self.keep_ind = None
        
        if self.keep_ind == None:
            if trained_set == 'fairface':
                ds = FairFace(iat_type=attribute, mode="train", transforms=self.preprocess)

            else:
                ds = UTKface(iat_type=attribute, mode='train', transforms=self.preprocess)
            dl = DataLoader(ds, batch_size=10, num_workers=6)
            x = defaultdict(lambda : [])
            y = []
            for batch in tqdm(dl):
                with torch.no_grad():
                    image_embeddings = self.clip.encode_image(batch['img'].to('cuda'))
                for dim in range(hidden_dim):
                    x[dim] += image_embeddings[:, dim].tolist()
                y += batch["iat_label"].tolist()

            scores = []
            for dim in range(hidden_dim): 
                scores.append(Mixed_KSG(np.array(x[dim]), np.array(y)))
            scores = torch.FloatTensor(scores)
            remaining_inds = torch.topk(scores, m, largest=False).indices
            remaining_inds = remaining_inds.sort().values
            self.keep_ind = remaining_inds


    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text) 
        return text_embeddings[:, self.keep_ind]

    def encode_image(self, image):
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return image_embeddings[:, self.keep_ind]