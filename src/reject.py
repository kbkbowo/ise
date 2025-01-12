import torch
from einops import rearrange
from src.feat_encoder import FeatEncoder
from src.utils import method2dim
import numpy as np

class RejectionSampler:
    def __init__(
            self, 
            dist_metric="l2", 
            agg_metric="min", 
            cache_plans=False,
            enc_method='dinov2'
        ):
        self.negatives = []
        self.cached_plans = []
        self.dist_metric = dist_metric
        # self.dist_metric = 'enc'
        self.agg_metric = agg_metric
        self.cache_plans = cache_plans
        self.feat_encoder = FeatEncoder(enc_method)
        self.feat_dim = method2dim[enc_method]
    
    def reset(self):
        self.negatives = []
        self.cached_plans = []

    def update_negatives(self, negatives):
        self.negatives += negatives

    def update_plans(self, plans):
        if self.cache_plans:
            self.cached_plans += plans
        else:
            self.cached_plans = plans

    def encode_video(self, video):
        return torch.from_numpy(self.feat_encoder(video)).cuda()

    def calc_dist(self, a_s, b_s):
        if self.dist_metric == "l2":
            # pairwise squared distance
            
            a_s = torch.from_numpy(rearrange(a_s, 'n f c h w -> 1 n (f c h w)')).float()
            b_s = torch.from_numpy(rearrange(b_s, 'm f c h w -> 1 m (f c h w)')).float()
            dist = torch.cdist(a_s, b_s, p=2)

            return dist
        elif self.dist_metric == 'enc':
            
            a_s = rearrange(a_s, 'n f c h w -> n f c h w')
            b_s = rearrange(b_s, 'm f c h w -> m f c h w')
            embedding_a = torch.stack([self.encode_video(a) for a in a_s]).detach().cpu()
            embedding_b = torch.stack([self.encode_video(b) for b in b_s]).detach().cpu()
            embedding_a_norm = embedding_a / torch.norm(embedding_a, dim=1, keepdim=True)
            embedding_b_norm = embedding_b / torch.norm(embedding_b, dim=1, keepdim=True)
            similarity = torch.matmul(embedding_a_norm, embedding_b_norm.transpose(0, 1)).unsqueeze(0)
            # dist = [torch.nn.functional.cosine_similarity(embedding_a, emb_b, dim=0) for emb_b in embedding_b]
            # dist = torch.stack(dist).unsqueeze(0) * (-1)
            # print(dist.shape)
            dist = 1 - similarity
            return dist
        
        else:
            raise NotImplementedError

    def aggregate(self, dists):
        if self.agg_metric == "mean":
            return dists.mean(dim=1)
        elif self.agg_metric == "min":
            return dists.min(dim=1)
        else:
            raise NotImplementedError
    
    def select_best(self):
        if len(self.negatives) == 0:
            # print("No negatives provided, returning a random plan...")
            return self.cached_plans[0]
        # print("Selecting best plan...")
        dists = self.calc_dist(self.negatives, self.cached_plans)
        agg_dists = self.aggregate(dists)
        return self.cached_plans[np.argmax(agg_dists.values).item()]