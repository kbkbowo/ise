import torch
from src.utils import init_env, upsample, get_env, sample_n_frames, method2dim, ALP_LIST
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import json
from PIL import Image
from tqdm import tqdm
import os
import imageio
from argparse import ArgumentParser
from flowdiffusion.inference_utils import get_video_model_diffae_feat_suc
from torchvision import transforms as T
from upsample_model import UpsampleModel
# pca
from sklearn.decomposition import PCA
import random
from extract_features import FeatEncoder
from torch import nn



class ExplicitRetrievalModule():
    def __init__(self, task_name, training_seeds=10, max_training_offset=36, on=True, prob_based=False, temperature=40, pca=False, enc_method="clip", guidance=0.0):
        self.prob_based = prob_based
        self.temperature = temperature
        self.pca = pca
        self.enc_method = enc_method
        self.guidance = guidance
        # import clip
        self.first_frame_transform = T.Compose([
            T.CenterCrop(128),
            T.Resize((32, 32)),
            T.ToTensor()
        ])
        # self.clip_pre_transform = T.Compose([
        #     T.CenterCrop(128),
        #     T.Resize((224, 224))
        # ])
        # self.model, self.preprocess = clip.load("ViT-B/32", device='cuda')
        self.feat_encoder = FeatEncoder(enc_method)
        self.feat_dim = method2dim[enc_method]

        if on:
            # preproces features
            with open(f"./down_dataset/{task_name}/{enc_method}/all_feats.npy", "rb") as f:
                all_dict = np.load(f, allow_pickle=True).item()

            filtered_dict = []
            filtered_feats = []
            filtered_out_feats = []
            for key, value in all_dict.items():
                try: 
                    (seed, cm, push) = key.split("_")
                except: 
                    (seed_0, seed_1, cm, push) = key.split("_")
                    seed = f"{seed_0}_{seed_1}"
                # print(seed, push, cm)
                # if (int(seed) < training_seeds) and (abs(int(cm) - int(push)) <= max_training_offset):
                filtered_dict.append(key)
                filtered_feats.append(value)
                out_feat = all_dict[f"{seed}_{cm}_{cm}"]
                filtered_out_feats.append(out_feat)
            self.filtered_feats = torch.from_numpy(np.array(filtered_feats)).float().cuda()
            self.filtered_dict = filtered_dict
            self.filtered_out_feats = torch.from_numpy(np.array(filtered_out_feats)).float().cuda()

            self.filtered_feats_mean = self.filtered_out_feats.mean(dim=0)
            self.filtered_feats_std = self.filtered_out_feats.std(dim=0)
        # optionally operate the retrieval on PCA space
        if self.pca:
            self.pca = PCA(n_components=32)
            self.pca.fit(self.filtered_feats.cpu().numpy())
            projected = self.pca.transform(self.filtered_feats.cpu().numpy())
            # normalize the features
            norm = np.linalg.norm(projected, axis=1)
            projected = projected / norm[:, None]
            self.filtered_feats = torch.from_numpy(projected).float().cuda()

        try: 
            global trainer
            self.diffusion = trainer.ema.ema_model # trainer->diffusion->unet
            trainer.model.cpu()
            self.diffusion.eval()
            self.text_feat = trainer.encode_batch_text([task_name.replace("_", " ")]).detach()
        except Exception as e:
            print(e)
            trainer = get_video_model_diffae_feat_suc(ckpt_dir=f"./pretrained/ckpts/{enc_method}", milestone=30, timestep=25, video_enc_dim=self.feat_dim)
            self.diffusion = trainer.ema.ema_model # trainer->diffusion->unet
            trainer.model.cpu()
            self.diffusion.eval()
            self.text_feat = trainer.encode_batch_text([task_name.replace("_", " ")]).detach()


        self.video = None


    def reset(self):
        self.video = None

    def encode_video(self, video):
        return torch.from_numpy(self.feat_encoder(video)).cuda()

    def refine_feat(self, interaction_video, feat, refine_steps=40):
        if refine_steps == 0:
            return feat

        # require grad
        self.diffusion.requires_grad_(True)
        feat = nn.Parameter(feat.unsqueeze(0).unsqueeze(0)).cuda()
        opt = torch.optim.Adam([feat], lr=1e-3)
        first_frame = torch.stack([self.first_frame_transform(Image.fromarray(interaction_video[0]))])
        first_frame = rearrange(first_frame, 'f c h w -> 1 (f c) h w')
        future_frames = torch.stack([self.first_frame_transform(Image.fromarray(frame)) for frame in interaction_video[1:]])
        future_frames = rearrange(future_frames, 'f c h w -> 1 (f c) h w')
        # print("refining feature...")
        for i in tqdm(range(refine_steps)):
            loss = self.diffusion(
                future_frames.cuda(),
                first_frame.cuda(),
                self.text_feat.expand(1, -1, -1).cuda(),
                feat, 
                torch.tensor([0]).cuda().unsqueeze(0),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.diffusion.requires_grad_(False)
        return feat.detach().squeeze().squeeze()


    def retrieve_nearest_n(self, feat, n=1):
        # find nearest
        # assert n == 1 # only support n=1 for now
        if self.pca:
            projected = self.pca.transform(feat[None].cpu().numpy())
            # normalize the features
            norm = np.linalg.norm(projected, axis=1)
            projected = projected / norm[:, None]
            feat = torch.from_numpy(projected[0]).float().cuda()

        if not self.prob_based:
            dist = torch.cdist(feat.unsqueeze(0), self.filtered_feats, p=2).squeeze()
            nearest_idx = torch.argsort(dist)[:n]
        else:
            # calculate cosine similarity similarity    
            if self.enc_method in ["clip", "custom", "dinov2"]:
                sim = torch.nn.functional.cosine_similarity(feat.unsqueeze(0), self.filtered_feats, dim=1)
            else: # euclidean distance
                sim = -torch.cdist(feat.unsqueeze(0), self.filtered_feats, p=2).squeeze()
            rank = torch.argsort(sim, descending=True).cpu().numpy()
            prob = torch.softmax(sim * self.temperature, dim=0)
            nearest_idx = torch.multinomial(prob, n, replacement=True).cpu().numpy()

        return [self.filtered_out_feats[i] for i in nearest_idx]

    def generate_n_sample(self, seed=0, resolution=(640, 480), camera='corner', task='push-test-v2-goal-observable', identity=None, n=1, f=8, f_o=8, refine_steps=0, refine_from_scratch=False):
        if self.video is None:
            # print("No video queried, generating random plan...")
            feats = [[None] for _ in range(n)]
        else:
            # print("Retrieving nearest plans...")
            video = sample_n_frames(self.video, f)
            feat = self.encode_video(video).float() # [1 1 5120]
            

            if refine_from_scratch:
                feat = torch.randn_like(feat) * self.filtered_feats_std + self.filtered_feats_mean

            feat = self.refine_feat(video, feat, refine_steps)
            feats = [f.cuda().unsqueeze(0).unsqueeze(0) for f in self.retrieve_nearest_n(feat, n)]
        # feat_dicts = self.retrieve_nearest_n(feat, n)
        samples = []
        env = get_env(task, seed, identity)
        obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)

        first_frame = self.first_frame_transform(Image.fromarray(image))

        for feat in feats:
            # generation = self.diffusion.p_sample_loop(
            generation = self.diffusion.sample(
                first_frame.unsqueeze(0).cuda(),
                self.text_feat.expand(1, -1, -1).cuda(),
                feat,
                torch.tensor([1]).cuda().unsqueeze(0),
                batch_size=1,
                guidance_weight=self.guidance
            )
            pred_frames = rearrange(generation, '1 (f c) h w -> c f h w', c=3).cpu()
            pred_video = torch.cat([first_frame.unsqueeze(1), pred_frames], dim=1)

            generation = upsample(pred_video.unsqueeze(0)).squeeze(0).numpy()

            # pad the generated video to the original resolution
            pad_h = (resolution[1] - generation.shape[2]) // 2
            pad_w = (resolution[0] - generation.shape[3]) // 2
            generation = np.pad(generation, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
            generation = ((generation * 255).astype(np.uint8))
            generation = rearrange(generation, 'c f h w -> f h w c')

            samples.append(generation)
        return samples

    def update_query(self, video):
        self.video = video
