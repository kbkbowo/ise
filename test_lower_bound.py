import warnings
warnings.filterwarnings("ignore")
import torch
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from utils import get_seg, get_cmat, collect_video
import numpy as np
import json
import json
import imageio
import random
import os
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from multiprocessing import Pool
def fix_seed(seed):
    # fix torch, np, and rendom seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_policy(env_name, relative_offset=None):
    print(env_name)
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"

    if relative_offset is None:
        policy = getattr(policies, policy_name)()
    else:
        policy = getattr(policies, policy_name)(relative_offset=relative_offset)
        
    return policy


def make_policy(task_name, task, pred_identity):

    conti_list = [
        'push-test-v2-goal-observable',
        'pick-bar-v2-goal-observable',
    ]
    
    discr_list = [
        'faucet-open-v2-goal-observable',
        'faucet-close-v2-goal-observable',
        'box-open-v2-goal-observable',
        'box-slide-v2-goal-observable'
    ]

    if task in conti_list:
        policy = get_policy(task, pred_identity)
    elif task in discr_list:
        policy = get_policy(pred_identity)
    return policy

def init_env(env, resolution, camera, task):
    obs = env.reset()
    with open("name2maskid.json", "r") as f:
        name2maskid = json.load(f)
    seg_ids = name2maskid[task]
    image, depth = env.render(depth=True, resolution=resolution, camera_name=camera)
    seg = get_seg(env, camera, resolution, seg_ids)
    cmat = get_cmat(env, camera, resolution)
    return obs, image, depth, seg, cmat

def interact(env, obs, policy, resolution, camera):
    images, depths, episode_return = collect_video(obs, env, policy, camera_name=camera)
    
    video = np.array(images) # N, H, W, C
    # print(video.shape)
    return video

def get_env(task, seed, identity):
    # print(task)
    
    conti_list = [
        'push-test-v2-goal-observable',
        'pick-bar-v2-goal-observable',
    ]
    
    discr_list = [
        'faucet-open-v2-goal-observable',
        'faucet-close-v2-goal-observable',
        'box-open-v2-goal-observable',
        'box-slide-v2-goal-observable'
    ]
    
    assert task in conti_list or task in discr_list
    
    if task in conti_list:
        env = env_dict[task](seed=seed, cm_offset=identity, cm_visible=False)
    elif task in discr_list:
        env = env_dict[task](seed=seed)
    
    
    return env

def test_env(friction, pred_identity, seed):
    
    task = 'push-test-v2-goal-observable'
    task_name = 'push_bar'
   
    # task = 'pick-bar-v2-goal-observable'
    # task_name = 'pick_bar' 

    task_name = 'open_box'
    
    if task_name == 'push_bar':
        task = 'push-test-v2-goal-observable'
    elif task_name == 'pick_bar':
        task = 'pick_bar-v2-goal-observable'
    elif task_name == 'open_box':
        identity = np.random.choice(['lift', 'slide'])
        if identity == 'lift':
            task = 'box-open-v2-goal-observable'
        else:
            task = 'box-slide-v2-goal-observable'
    elif task_name == 'turn_faucet':
        identity = np.random.choice(['open', 'close'])
        if identity == 'open':
            task = 'faucet-open-v2-goal-observable'
        else:
            task = 'faucet-close-v2-goal-observable'
    else:
        raise NotImplementedError
    
    resolution = (640, 480)
    camera = 'corner'
    seed = seed
    np.random.seed(seed)
    highest = None
    policy = make_policy(task_name, task, pred_identity)
    
    env = get_env(task, seed, friction)
    obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)
    interaction = interact(env, obs, policy, resolution, camera)
    
    frameskip = 16
    # imageio.mimsave(f"video.mp4", interaction[::frameskip], fps=16)
    print(len(interaction) )
    
    return int(len(interaction) < 500)
    
def worker(args):
    identity, p_identity, seed = args
    return test_env(identity, p_identity, seed)
    
def main():
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seeds = np.arange(0, 100)
    print(seeds)
    
    cm_list = np.linspace(-0.18, 0.18, 10)
    cm_list = [
        'box-open-v2-goal-observable',
        'box-slide-v2-goal-observable'
    ]
    
    # friction_list = [
    #     0.24, 0.25, 
    # ]
    
    results = []
                
    # with Pool() as pool:
    #     for result in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
    #         results.append(result)
    
    # print(results)
    
    pbar = tqdm(total=len(cm_list)*len(cm_list)*len(seeds))
    results = []
    for seed in seeds:
        for identity in cm_list:
            for p_identity in cm_list:
                # os.makedirs(f"down_dataset/slide_brick/{seed}/{friction}/{p_friction}", exist_ok=True)
                result = test_env(identity, p_identity, int(seed))
                results.append(result)
                pbar.set_description(f'{sum(results)/len(results)}')
                pbar.update(1)
            
    print(sum(results)/len(results))
    
if __name__ == "__main__":

    main()