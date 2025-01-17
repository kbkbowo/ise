import warnings
warnings.filterwarnings("ignore")
from unimatch.unimatch import UniMatch
from argparse import ArgumentParser
import torch
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from utils import get_seg, get_cmat, collect_video, sample_n_frames
import numpy as np
import torch.nn.functional as F
from myutils import get_transforms, get_transformation_matrix
from einops import rearrange
import json
from PIL import Image
from tqdm import tqdm
import json
import os
import imageio
from argparse import ArgumentParser
import pickle
from flowdiffusion.inference_utils import get_video_model_diffae_feat_suc
from torchvision import transforms as T
# from torchvision.video import transform as VT
from upsample_model import UpsampleModel
# pca
from sklearn.decomposition import PCA
import random
from extract_features import FeatEncoder
from torch import nn
from torchvideotransforms import video_transforms, volume_transforms
# global trainer
# global upsample_model

method2dim = {
    "clip": 4096,
    "dinov2": 6144,
    "r3m": 16384,
    "vc1": 6144,
    "custom": 512
}
ALP_LIST = list("LBKRSOEZ")

def max_trials(task_name):
    if task_name in ["push_alphabet"]:
        return 8
    else:
        return 16

def fix_seed(seed):
    # fix torch, np, and rendom seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_policy(env_name, relative_offset=None):
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    if relative_offset is None:
        policy = getattr(policies, policy_name)()
    elif env_name == 'push-alphabet-v2-goal-observable':
        policy = getattr(policies, policy_name)(offset=-relative_offset)
    else:
        policy = getattr(policies, policy_name)(relative_offset=relative_offset)
    return policy

def get_flow_model():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    parser.add_argument('--feature_channels', type=int, default=128)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--upsample_factor', type=int, default=4)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--ffn_dim_expansion', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--reg_refine', type=bool, default=True)
    parser.add_argument('--task', type=str, default='flow')
    args = parser.parse_args(args=[])
    DEVICE = 'cuda:0'

    model = UniMatch(feature_channels=args.feature_channels,
                        num_scales=args.num_scales,
                        upsample_factor=args.upsample_factor,
                        num_head=args.num_head,
                        ffn_dim_expansion=args.ffn_dim_expansion,
                        num_transformer_layers=args.num_transformer_layers,
                        reg_refine=args.reg_refine,
                        task=args.task).to(DEVICE)

    checkpoint = torch.load(args.model, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model'])

    model.to(DEVICE)
    model.eval()
    model._requires_grad = False
    return model

### predict per frame flow   
def pred_flow_frame(model, frames, stride=1, device='cuda:0'):
    DEVICE = device 
    model = model.to(DEVICE)
    frames = torch.from_numpy(frames).float()
    images1 = frames[:-1]
    images2 = frames[1:]
    flows = []
    flows_b = []
    # print("starting prediction")
    # t0 = time.time()
    for image1, image2 in zip(images1, images2):
        image1, image2 = image1.unsqueeze(0).to(DEVICE), image2.unsqueeze(0).to(DEVICE)
    
        # nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
        #                     int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        ### dumb upsampling to (480, 640)
        nearest_size = [480, 640]
        inference_size = nearest_size
        ori_size = image1.shape[-2:]
        
        # print("inference_size", inference_size)
        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
        with torch.no_grad():
            results_dict = model(image1, image2,
                attn_type='swin',
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task='flow',
                pred_bidir_flow=True,
            )
        
        flow_pr = results_dict['flow_preds'][-1]
        
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
                
        flows += [flow_pr[0:1].permute(0, 2, 3, 1).cpu()]
        flows_b += [flow_pr[1:2].permute(0, 2, 3, 1).cpu()]
        
    flows = torch.cat(flows, dim=0)
    flows_b = torch.cat(flows_b, dim=0)
    
    flows = flows.numpy()
    flows_b = flows_b.numpy()
    
    return images1, images2, flows, flows_b

def get_subgoals(seg, depth, cmat, video, flow_model, task_name=None):
    if task_name in ['open_box', 'turn_faucet']:
        video = sample_n_frames(video, 8).transpose(0, 3, 1, 2) # N, C, H, W
    elif task_name in ['push_bar', 'push_alphabet']:
        video = sample_n_frames(video, 8).transpose(0, 3, 1, 2)[:2]
    elif task_name in ['pick_bar']:
        video = sample_n_frames(video, 8).transpose(0, 3, 1, 2)[:6]
    else:
        raise NotImplementedError
    images1, images2, flows, flows_b = pred_flow_frame(flow_model, video, stride=1, device='cuda:0')

    grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flows)
    transform_mats = [get_transformation_matrix(*transform) for transform in transforms]

    subgoals = [grasp[0]]
    for i, transforms in enumerate(transform_mats):
        grasp_ext = np.concatenate([subgoals[-1], [1]])
        next_subgoal = (transforms @ grasp_ext)[:3]
        subgoals.append(next_subgoal)

    return subgoals

def pred_identity(task_name, subgoals):
    if task_name in ['push_bar', 'pick_bar', 'push_alphabet']:
        p_identity = (subgoals[0] - subgoals[-1])[0]
    elif task_name in ['open_box']:
        trans = subgoals[-1] - subgoals[0]
        trans_y = abs(trans[1])
        trans_z = abs(trans[2])
        if trans_y < trans_z:
            p_identity = "box-open-v2-goal-observable"
        else:
            p_identity = "box-slide-v2-goal-observable"
    elif task_name in ['turn_faucet']:
        trans = subgoals[-1] - subgoals[0]
        trans_x = trans[0]
        print(trans_x)
        if trans_x < 0:
            p_identity = "faucet-open-v2-goal-observable"
        else:
            p_identity = "faucet-close-v2-goal-observable"
    return p_identity


def make_policy(task_name, task, pred_identity):
    if task_name in ['push_bar', 'pick_bar', 'push_alphabet']:
        policy = get_policy(task, pred_identity)
        # return get_policy(task, pred_identity)
    elif task_name in ['open_box', 'turn_faucet']:
        policy = get_policy(pred_identity)
        # return get_policy(pred_identity)
    else:
        raise NotImplementedError
    return policy


# def get_push_loc(seg, depth, cmat, video, flow_model):
#     video = sample_n_frames(video, 8).transpose(0, 3, 1, 2)[:2] # N, C, H, W
#     images1, images2, flows, flows_b = pred_flow_frame(flow_model, video, stride=1, device='cuda:0')

#     grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flows)
#     transform_mats = [get_transformation_matrix(*transform) for transform in transforms]

#     subgoals = [grasp[0]]
#     for i, transforms in enumerate(transform_mats):
#         grasp_ext = np.concatenate([subgoals[-1], [1]])
#         next_subgoal = (transforms @ grasp_ext)[:3]
#         subgoals.append(next_subgoal)

#     x_offset = (subgoals[0] - subgoals[1])[0]
#     return x_offset

def init_env(env, resolution, camera, task):
    obs = env.reset()
    with open("name2maskid.json", "r") as f:
        name2maskid = json.load(f)
    seg_ids = name2maskid[task]
    image, depth = env.render(depth=True, resolution=resolution, camera_name=camera)
    seg = get_seg(env, camera, resolution, seg_ids)
    cmat = get_cmat(env, camera, resolution)
    return obs, image, depth, seg, cmat

# def interact(env, obs, relative_offset, resolution, camera):
#     policy = get_policy('push-test-v2-goal-observable', relative_offset)
#     images, depths, episode_return = collect_video(obs, env, policy, camera_name=camera)
#     video = np.array(images) # N, H, W, C
#     return video

def interact(env, obs, policy, resolution, camera):
    images, depths, episode_return = collect_video(obs, env, policy, camera_name=camera)
    video = np.array(images) # N, H, W, C
    return video

def get_demonstration_gt(env, obs, push_offset, cm_offset, resolution, camera):
    relative_offset = push_offset - cm_offset
    return interact(env, obs, relative_offset, resolution, camera)

def generate_sample_gt(seed, resolution, camera, task='push-test-v2-goal-observable'):
    cm_upper =  0.18
    cm_lower = -0.18
    cm_offset = np.random.uniform(cm_lower, cm_upper)
    env = env_dict[task](seed=seed, cm_offset=cm_offset, cm_visible=False)
    obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)
    video = interact(env, obs, 0, resolution, camera)
    return video

def batch_sample_plans_gt(seed, resolution, camera, task='push-test-v2-goal-observable', n=10, f=8):
    samples = []
    for i in range(n):
        sample = generate_sample_gt(seed+i, resolution, camera, task)
        sample = sample_n_frames(sample, f)
        samples.append(sample)
    return samples

def upsample(video_32):
    # video_32: N, C, F, H, W tensor
    try: 
        global upsample_model
        with torch.no_grad():
            upsampled = upsample_model(video_32.cuda()).cpu()
    except: 
        print("upsample model not found, loading...")
        # load model and make it global
        upsample_model_path = "./pretrained/model_15.pth"
        upsample_model = UpsampleModel().cuda()
        try:
            upsample_model.load_state_dict(torch.load(upsample_model_path, weights_only=True))
        except: # need to unwrap the model
            class dummy_model(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.module = model
                def forward(self, x):
                    return self.module(x)
            upsample_model = dummy_model(upsample_model)
            upsample_model.load_state_dict(torch.load(upsample_model_path, weights_only=True))
            upsample_model = upsample_model.module
        upsample_model.eval()
        with torch.no_grad():
            upsampled = upsample_model(video_32.cuda()).cpu()

    # clamp values
    upsampled = torch.clamp(upsampled, 0, 1)
    return upsampled

def get_env(task, seed, identity):
    print(task)
    if task in ['push-test-v2-goal-observable', 'pick-bar-v2-goal-observable']:
        env = env_dict[task](seed=seed, cm_offset=identity, cm_visible=False)
    elif task in ['box-open-v2-goal-observable', 'box-slide-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable']:
        env = env_dict[task](seed=seed)
    elif task in ['push-alphabet-v2-goal-observable']:
        alphabet = ALP_LIST[seed % len(ALP_LIST)]
        env = env_dict[task](seed=seed, init_seed=seed, cm_sample_seed=seed, letter=alphabet, cm_visible=False, font_size=10, font_path="/tmp2/seanfu/temp/mw_temp/alphabet_dataset_gen/basic.ttf")
    else:
        raise NotImplementedError
    return env

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
        # print("ENCODE_VIDEO_REJECTION", video.shape)
        return torch.from_numpy(self.feat_encoder(video)).cuda()

    def calc_dist(self, a_s, b_s):
        if self.dist_metric == "l2":
            # pairwise squared distance
            
            a_s = torch.from_numpy(rearrange(a_s, 'n f c h w -> 1 n (f c h w)')).float()
            b_s = torch.from_numpy(rearrange(b_s, 'm f c h w -> 1 m (f c h w)')).float()
            dist = torch.cdist(a_s, b_s, p=2)
            return dist

        elif self.dist_metric == 'dinov2':
            
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
            print("DIST_REJECTION", dist.shape)
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
            return self.cached_plans[0]
        dists = self.calc_dist(self.negatives, self.cached_plans)
        agg_dists = self.aggregate(dists).values
        # print("AGG_DIST_REJECTION_ORIGIN", agg_dists)
        agg_dists_normalized = torch.nn.functional.normalize(agg_dists, p=2, dim=1)
        # print("AGG_DIST_REJECTION_NORMALIZED", agg_dists_normalized)
        TEMPERATURE = 3
        agg_dists_soft = torch.softmax(agg_dists_normalized * TEMPERATURE, dim=1)
        # print("AGG_DIST_REJECTION_SOFTMAX", agg_dists_soft)
        SOFT = False
        if SOFT:
            idx = torch.distributions.Categorical(agg_dists_soft).sample().item()
        else:
            idx = torch.argmax(agg_dists).item()
        return self.cached_plans[idx]

class ExplicitRetrievalModule():
    def __init__(self, task_name, training_seeds=10, max_training_offset=36, on=True, prob_based=False, temperature=40, pca=False, enc_method="clip", guidance=0.0, random_generation=0.0):
        self.prob_based = prob_based
        self.temperature = temperature
        self.pca = pca
        self.enc_method = enc_method
        self.guidance = guidance
        self.random_generation = random_generation
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
        # video: N, H, W, C np array
        # convert to PIL images
        # images = [self.clip_pre_transform(Image.fromarray(frame)) for frame in video]
        # # preprocess
        # images = [self.preprocess(image) for image in images]
        # images = torch.stack(images).to('cuda')
        # # encode
        # with torch.no_grad():
        #     image_features = self.model.encode_image(images)
        # return image_features.flatten()
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
            feats = [[None] for _ in range(n)]
        elif refine_from_scratch:
            video = sample_n_frames(self.video, f)
            feat = self.encode_video(video).float()
            origin_feat = torch.randn_like(feat) * self.filtered_feats_std + self.filtered_feats_mean
            feat = self.refine_feat(video, origin_feat, refine_steps)
            
            num_conditioned = round(n * (1 - self.random_generation))
            num_unconditioned = n - num_conditioned
            
            feat_list = []
            for _ in range(num_conditioned):
                feat_list.append(feat.cuda().unsqueeze(0).unsqueeze(0))
            for _ in range(num_unconditioned):
                feat_list.append([None])
                
            # feats = [f.cuda().unsqueeze(0).unsqueeze(0) for f in feat_list]
            feats = feat_list
        else:
            video = sample_n_frames(self.video, f)
            feat = self.encode_video(video).float() # [1 1 5120]
            
            feat = self.refine_feat(video, feat, refine_steps)

            feats = [f.cuda().unsqueeze(0).unsqueeze(0) for f in self.retrieve_nearest_n(feat, n)]
        # feat_dicts = self.retrieve_nearest_n(feat, n)
        samples = []
        env = get_env(task, seed, identity)
        obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)

        first_frame = self.first_frame_transform(Image.fromarray(image))

        for feat in feats:
            # if feat is not None:
            #     print("feat shape", feat.shape)
            #     feat = feat.squeeze().squeeze()
            #     print("feat shape", feat.shape)
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


# retrieval_module = CLIPRetrievalModule()
# video = np.zeros((10, 480, 640, 3)).astype(np.uint8)
# image_features = retrieval_module.generate_n_sample(video)
# print(image_features[0].shape)
# print(len(image_features))
def main(seed, n, retrieve_on, retrieval_module, out_file, out_dir, agg_metric, temperature, pca, task_name, encode_method, save_video, refine_steps, refine_from_scratch, dist_metric, guidance, random_generation):
    # MAX_TRIALS = 16
    MAX_TRIALS = max_trials(task_name)
    resolution = (640, 480)
    camera = "corner"
    fix_seed(seed)

    try:
        with open(out_file, "r") as f:
            results = json.load(f)
    except:
        results = []

    last_done = (results[-1]["pred_identity"] is not None) if len(results) > 0 else True
    if not last_done:
        results = results[:-1]

    # check the last result
    last_seed = results[-1]["seed"] if len(results) > 0 else 0
    if last_seed >= seed:
        # skip the seed
        return
    # task = 'push-test-v2-goal-observable'

    # assert task_name in ['push_bar', 'pick_bar', 'open_box', 'open_faucet']
    if task_name == 'push_bar':
        identity = np.random.uniform(0, 0.36) - 0.18
        task = 'push-test-v2-goal-observable'
    elif task_name == 'pick_bar':
        identity = np.random.uniform(0, 0.36) - 0.18
        task = 'pick-bar-v2-goal-observable'
    elif task_name == 'open_box':
        identity = np.random.choice(["lift", "slide"])
        if identity == "lift":
            task = 'box-open-v2-goal-observable'
        else:
            task = 'box-slide-v2-goal-observable'
    elif task_name == 'turn_faucet':
        identity = np.random.choice(["open", "close"])
        if identity == "open":
            task = 'faucet-open-v2-goal-observable'
        else:
            task = 'faucet-close-v2-goal-observable'
    elif task_name == 'push_alphabet':
        task = 'push-alphabet-v2-goal-observable'
        alphabet = ALP_LIST[seed % len(ALP_LIST)]
        env = env_dict[task](seed=seed, axis_seed=seed, alphabet=alphabet, cm_visible=False, font_size=10, font_path="/tmp2/seanfu/temp/mw_temp/alphabet_dataset_gen/basic.ttf")
        identity = (alphabet, env.x_offset())
    else:
        raise NotImplementedError


    flow_model = get_flow_model()
    if retrieval_module == "clip":
        retrieval_module = CLIPRetrievalModule()
    elif retrieval_module == "implicit":
        retrieval_module = ImplicitRetrievalModule()
    elif retrieval_module == "explicit":
        retrieval_module = ExplicitRetrievalModule(task_name=task_name, on=retrieve_on, pca=pca, enc_method=encode_method, guidance=guidance, random_generation=random_generation)
    elif retrieval_module == "explicit_prob":
        retrieval_module = ExplicitRetrievalModule(task_name=task_name, on=retrieve_on, prob_based=True, temperature=temperature, pca=pca, enc_method=encode_method, guidance=guidance, random_generation=random_generation)
    else: 
        raise NotImplementedError
    rejection_sampler = RejectionSampler(agg_metric=agg_metric, dist_metric=dist_metric)

    results.append({
        "task": task_name,
        "seed": seed,
        "identity": identity,
        "success": False,
        "trials": MAX_TRIALS,
        "pred_identity": None,
        "int_length": None
    })

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    retrieval_module.reset()
    rejection_sampler.reset()
    success = False
    p_identities = []
    for tt in tqdm(range(MAX_TRIALS)):
        trail_dir = f"{out_dir}/{seed}/{tt}"

        env = get_env(task, seed, identity)
        obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)

        # gt_plan = get_demonstration_gt(env, obs, push_offset, cm_offset, resolution, camera)
        plans = retrieval_module.generate_n_sample(seed=seed, resolution=resolution, camera=camera, task=task, identity=identity, n=n if tt > 0 else 1, refine_steps=refine_steps, refine_from_scratch=refine_from_scratch)
        os.makedirs(f"{trail_dir}/plans", exist_ok=True)

        for i, plan in enumerate(plans):
            imageio.mimsave(f"{trail_dir}/plans/{i}.mp4", plan, fps=8)

        rejection_sampler.update_plans(plans)
        plan = rejection_sampler.select_best()

        imageio.mimsave(f"{trail_dir}/best_plan.mp4", plan, fps=8)

        # x_offset = get_push_loc(seg, depth, cmat, plan, flow_model)
        # x_offset = np.clip(x_offset, -0.18, 0.18)
        # push_attempts.append(x_offset)
        subgoals = get_subgoals(seg, depth, cmat, plan, flow_model, task_name)
        p_identity = pred_identity(task_name, subgoals)
        if task_name in ['push_bar', 'pick_bar', 'push_alphabet']:
            relative_offset = p_identity + obs[4]
            p_identity = relative_offset
        p_identities.append(p_identity)
        policy = make_policy(task_name, task, p_identity)

        # env = env_dict[task](seed=seed, cm_offset=cm_offset, cm_visible=False)
        env = get_env(task, seed, identity)
        obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)
        interaction = interact(env, obs, policy, resolution, camera)

        frameskip = 16
        imageio.mimsave(f"{trail_dir}/interaction.mp4", interaction[::frameskip], fps=16)

        if len(interaction) < 500:
            success = True
            break

        subsampled_interaction = sample_n_frames(interaction, 8)
        if retrieve_on:
            retrieval_module.update_query(subsampled_interaction)
        rejection_sampler.update_negatives([plan])

    results[-1] = {
        "task": task_name,
        "seed": seed,
        "identity": identity,
        "success": success,
        "trials": tt,
        "pred_identity": p_identities,
        "int_length": len(interaction)
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":

    # module = ExplicitRetrievalModule(prob_based=True)
    # video = np.zeros((8, 480, 640, 3)).astype(np.uint8)

    # # update query
    # module.update_query(video)
    # # generate n samples
    # samples = module.generate_n_sample(seed=0, resolution=(640, 480), camera='corner', task='push-test-v2-goal-observable', n=1, f=8, f_o=8)
    # raise


    parser = ArgumentParser()
    
    parser.add_argument('-r', '--reduced_output', action='store_true')
    
    parser.add_argument('-s', '--seeds', type=int, default=400)
    # parser.add_argument('--cm_offset', type=int, default=0)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--retrieve_on', action='store_true')
    parser.add_argument('--retrieval_module', type=str, default='explicit_prob')
    parser.add_argument('--out_file', type=str, default='results_0915_test.json')
    parser.add_argument('--out_dir', type=str, default='results_0915_test')
    parser.add_argument('--agg_metric', type=str, default='min')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--pca', action='store_false')
    parser.add_argument('--task_name', type=str, default='turn_faucet')
    parser.add_argument('--enc_method', type=str, default='dinov2')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--refine_steps', type=int, default=100)
    parser.add_argument('--refine_from_scratch', action='store_true')
    parser.add_argument('--dist_metric', type=str, default='l2')
    parser.add_argument('-g', '--guidance', type=float, default=0.0)
    parser.add_argument('-rg', '--random_generation', type=float, default=0.0)
    args = parser.parse_args()
    for seed in tqdm(range(1000, 1000 + args.seeds)):
        if not args.retrieve_on:
            args.pca = False
        main(seed, args.n, args.retrieve_on, args.retrieval_module, args.out_file, args.out_dir, args.agg_metric, args.temperature, args.pca, args.task_name, args.enc_method, args.save_video, args.refine_steps, args.refine_from_scratch,
             args.dist_metric, args.guidance, args.random_generation)