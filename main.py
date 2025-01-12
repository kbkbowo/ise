import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import torch
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies

import numpy as np
import torch.nn.functional as F
from myutils import get_transforms, get_transformation_matrix
import json
from tqdm import tqdm
import os
import imageio
from argparse import ArgumentParser
from upsample_model import UpsampleModel
# pca
import random

from src.utils import collect_video, sample_n_frames, init_env, upsample, get_env, get_flow_model
from src.retrival import ExplicitRetrievalModule
from src.reject import RejectionSampler

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


def main(seed, n, retrieve_on, retrieval_module, out_file, out_dir, agg_metric, temperature, pca, task_name, encode_method, save_video, refine_steps, refine_from_scratch, dist_metric, guidance):
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
        print(results[-1])
        print(f"Detected unfinished seed {seed}, pruning...")
        results = results[:-1]

    # check the last result
    last_seed = results[-1]["seed"] if len(results) > 0 else 0
    if last_seed >= seed:
        print(f"Seed {seed} already done, skipping...")
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
    if retrieval_module == "explicit":
        retrieval_module = ExplicitRetrievalModule(task_name=task_name, on=retrieve_on, pca=pca, enc_method=encode_method, guidance=guidance)
    elif retrieval_module == "explicit_prob":
        retrieval_module = ExplicitRetrievalModule(task_name=task_name, on=retrieve_on, prob_based=True, temperature=temperature, pca=pca, enc_method=encode_method, guidance=guidance)
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
        print("Calculating policy parameters...")
        subgoals = get_subgoals(seg, depth, cmat, plan, flow_model, task_name)
        print(subgoals)
        p_identity = pred_identity(task_name, subgoals)
        if task_name in ['push_bar', 'pick_bar', 'push_alphabet']:
            print(p_identity)
            print(obs[4])
            relative_offset = p_identity + obs[4]
            p_identity = relative_offset
        p_identities.append(p_identity)
        policy = make_policy(task_name, task, p_identity)

        print("Making interaction...")
        # env = env_dict[task](seed=seed, cm_offset=cm_offset, cm_visible=False)
        env = get_env(task, seed, identity)
        obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)
        interaction = interact(env, obs, policy, resolution, camera)

        frameskip = 16
        imageio.mimsave(f"{trail_dir}/interaction.mp4", interaction[::frameskip], fps=16)

        print("Interaction length:", len(interaction))
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

    parser = ArgumentParser()
    
    parser.add_argument('-r', '--reduced_output', action='store_true')
    
    parser.add_argument('--seeds', type=int, default=400)
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
    parser.add_argument('--refine_steps', type=int, default=40)
    parser.add_argument('--refine_from_scratch', action='store_true')
    parser.add_argument('--dist_metric', type=str, default='l2')
    parser.add_argument('-g', '--guidance', type=float, default=0.0)
    args = parser.parse_args()
    for seed in tqdm(range(1000, 1000 + args.seeds)):
        if not args.retrieve_on:
            args.pca = False
        main(seed, args.n, args.retrieve_on, args.retrieval_module, args.out_file, args.out_dir, args.agg_metric, args.temperature, args.pca, args.task_name, args.enc_method, args.save_video, args.refine_steps, args.refine_from_scratch,
             args.dist_metric, args.guidance)