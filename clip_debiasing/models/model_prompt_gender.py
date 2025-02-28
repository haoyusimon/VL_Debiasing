import clip
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import TruncatedSVD


class CLIP_prompt_gender(nn.Module):

    def __init__(self, arch_str, device, **_kwargs,):

        def get_embeddings(text, clip_model, normalize=True):
            text_tokens = clip.tokenize(text)

            clip_model.to(device)
            clip_model.eval()
            with torch.no_grad():
                text_tokens = text_tokens.to(device)
                text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
                if normalize:
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            return text_embeddings
        
        # Helper functions for debiasing 
        def get_proj_matrix(embeddings):
            tSVD = TruncatedSVD(n_components=len(embeddings))
            embeddings_ = tSVD.fit_transform(embeddings)
            basis = tSVD.components_.T

            # orthogonal projection
            proj = np.linalg.inv(np.matmul(basis.T, basis))
            proj = np.matmul(basis, proj)
            proj = np.matmul(proj, basis.T)
            proj = np.eye(proj.shape[0]) - proj
            return proj

        def get_A(z_i, z_j):
            z_i = z_i[:, None]
            z_j = z_j[:, None]
            return np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T)

        def get_M(embeddings, S):
            d = embeddings.shape[1]
            M = np.zeros((d, d))
            for s in S:
                M  += get_A(embeddings[s[0]], embeddings[s[1]])
            return M / len(S)
        
        super().__init__()

        self.dtype = torch.float32
        self.device = device

        self.lam = 1000

        self.clip, self.preprocess = clip.load(arch_str, device=device)

        spurious_prompt = ['A photo of a male.',
                           'A photo of a female.']

        candidate_prompt = []
        S = []
        concepts = ["good", "evil"]
        # concepts = ["good", "kind", "successful", "noncriminal", "nonviolent", "moral", "rich", "honest", "smart", "evil", "unkind", "unsuccessful", "criminal", "violent", "immoral", "poor", "dishonest", "dumb", "clever", "stupid", "hardworking", "lazy", "nasty", "trustworthy", "caring", "dangerous", "heroic", "villainous"]
        for idx, word in enumerate(concepts):
            candidate_prompt += [f'A photo of a male {word} person.', f'A photo of a female {word} person.']
            article = 'an' if word in ["evil", "immoral", "unkind", "unsuccessful", "honest", "untrustworthy"] else 'a'
            candidate_prompt += [f'A photo of {article} {word} man.', f'A photo of {article} {word} woman.']
            S += [[2*idx, 2*idx + 1], [2*idx + 2, 2*idx + 3]]

        spurious_embeddings = get_embeddings(spurious_prompt,
                                             self.clip,
                                             normalize=True)
        
        spurious_embeddings = spurious_embeddings.numpy()
        P0 = get_proj_matrix(spurious_embeddings)

        # Calculate Embedding of Positive Pairs
        candidate_embeddings = get_embeddings(candidate_prompt,
                                             self.clip,
                                             normalize=True)
        candidate_embeddings = candidate_embeddings.numpy()

        # Closed Form Optimum
        print('Solve Closed Form Optimum')
        self.M = get_M(candidate_embeddings, S)
        self.G = self.lam * self.M + np.eye(self.M.shape[0])
        self.P = np.matmul(P0, np.linalg.inv(self.G))

    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip.encode_text(text)

        text_embeddings = np.matmul(text_embeddings.cpu(), self.P.T)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embeddings = torch.tensor(text_embeddings).float().to(self.device)
        return text_embeddings

    def encode_image(self, image):
        with torch.no_grad():
            image_embeddings = self.clip.encode_image(image)
    
        return image_embeddings
    