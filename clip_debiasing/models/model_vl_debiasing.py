import clip
import torch
import torch.nn as nn

import torch.nn.functional as F

import random

class MLP(nn.Module):
    def __init__(self, D_in, D_out, n_layer=1, hidden_size=512):
        super().__init__()
        if n_layer == 1:
            self.network = nn.Linear(D_in, D_out)
        elif n_layer == 2:
            self.network = nn.Sequential(
                nn.Linear(D_in, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, D_out)
            )
        
    def forward(self,x):
        x = self.network(x)
        x = F.relu(x)
        return x


class DebiasedCLIP(nn.Module):

    def __init__(self, arch_str, device, mlp1_hidden_size=None, mlp2_hidden_size=None, alpha=None, debiasing_modules=True, **_kwargs,):
        super().__init__()

        self.dtype = torch.float32

        self.clip, self.preprocess = clip.load(arch_str, device=device)
        
        for param in self.parameters():
            param.requires_grad = False # freeze all parameters
        
        if debiasing_modules:
            if arch_str == 'ViT-L/14':
                input_dim = 768
            else:
                input_dim = 512

            self.mlp1 = MLP(input_dim, input_dim, 2, hidden_size=mlp1_hidden_size).to(device)
            self.mlp2 = MLP(input_dim, input_dim, 2, hidden_size=mlp2_hidden_size).to(device) 

            for param in list(self.mlp1.parameters()) + list(self.mlp2.parameters()):
                param.requires_grad = True 

            self.queue_size = 65536
            self.momentum = 0.995
            self.ratio = 0.4

            self.register_buffer("image_queue", torch.randn(input_dim, self.queue_size))
            self.register_buffer("text_queue", torch.randn(input_dim, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
            self.temp = nn.Parameter(torch.ones([]) * 0.07)

            self.mlp1_m = MLP(input_dim, input_dim, 2, hidden_size=mlp1_hidden_size).to(device)
            self.mlp2_m = MLP(input_dim, input_dim, 2, hidden_size=mlp2_hidden_size).to(device)

            self.model_pairs = [[self.mlp1,self.mlp1_m],
                                [self.mlp2,self.mlp2_m],
                            ]
            self.copy_params()
            self.alpha = alpha
        

    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text)
        return self.mlp2(text_embeddings.float()) + text_embeddings

    def encode_image(self, image):
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return self.mlp2(self.mlp1(image_embeddings.float()) + image_embeddings) + image_embeddings
        
    def forward(self, image, text1, text2, epoch):        
        with torch.no_grad():
            self._momentum_update()
            original_image_features = F.normalize(self.clip.encode_image(image), dim=-1)
            original_text_features = F.normalize(self.clip.encode_text(text1), dim=-1)

            image_embeddings = self.clip.encode_image(image)
            debiased_image_features_m = F.normalize(self.mlp2_m(self.mlp1_m(image_embeddings.float()) + image_embeddings) + image_embeddings, dim=-1)

            text_embeddings = self.clip.encode_text(text1)
            debiased_text_features_m = F.normalize(self.mlp2_m(text_embeddings.float()) + text_embeddings, dim=-1)

            original_image_features_all = torch.cat([original_image_features.t(),self.image_queue.clone().detach()],dim=1)
            original_text_features_all = torch.cat([original_text_features.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_targets = F.softmax(original_image_features.float() @ original_text_features_all.float() / self.temp, dim=1)
            sim_t2i_targets = F.softmax(original_text_features.float() @ original_image_features_all.float() / self.temp, dim=1)

            sim_i2t_targets_m = F.softmax(debiased_image_features_m.float() @ original_text_features_all.float() / self.temp, dim=1)
            sim_t2i_targets_m = F.softmax(debiased_text_features_m.float() @ original_image_features_all.float() / self.temp, dim=1)

            if epoch >= 5: # learn a basic debiasing module first
                sim_i2t_targets = self.ratio * sim_i2t_targets_m + (1 - self.ratio) * sim_i2t_targets
                sim_t2i_targets = self.ratio * sim_t2i_targets_m + (1 - self.ratio) * sim_t2i_targets

        fair_image_features = F.normalize(self.encode_image(image), dim=-1)

        fair_text1_features = F.normalize(self.encode_text(text1), dim=-1)
        fair_text2_features = F.normalize(self.encode_text(text2), dim=-1)

        biased_text1_features = original_text_features - fair_text1_features
        biased_image_features = original_image_features - fair_image_features

        bias_t2t = biased_text1_features.float() @ original_text_features_all.float() / self.temp
        bias_i2i = biased_image_features.float() @ original_image_features_all.float() / self.temp

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss_ba = kl_loss(F.log_softmax(bias_i2i, dim=1), F.log_softmax(bias_t2t, dim=1))

        fair_text_random = fair_text1_features if random.random() < 0.5 else fair_text2_features

        fair_i2t = fair_image_features.float() @ original_text_features_all.float() / self.temp
        fair_t2i = fair_text_random.float() @ original_image_features_all.float() / self.temp

        loss_i2t = -torch.sum(F.log_softmax(fair_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(fair_t2i, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_cd = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(original_image_features, original_text_features)
        return loss_cd * self.alpha + loss_ba * (1 - self.alpha)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            self.image_queue[:, ptr:] = image_feats[:self.queue_size - ptr].T
            self.text_queue[:, ptr:] = text_feats[:self.queue_size - ptr].T
            self.image_queue[:, :ptr + batch_size - self.queue_size] = image_feats[self.queue_size - ptr:].T
            self.text_queue[:, :ptr + batch_size - self.queue_size] = text_feats[self.queue_size - ptr:].T
        else:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data) 
                param_m.requires_grad = False  
