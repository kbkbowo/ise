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
    elif env_name == 'push-alphabet-v2-goal-observable':
        policy = getattr(policies, policy_name)(offset=-relative_offset)
    else:
        print(policy_name, relative_offset)
        # policy = getattr(policies, policy_name)(relative_offset=relative_offset)
        policy = getattr(policies, policy_name)()
        
    return policy

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
    elif task_name in ['brick_slide']:
        policy = get_policy(task, pred_identity)
        # return get_policy(pred_identity)
    else:
        raise NotImplementedError
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
    print(video.shape)
    return video

def get_env(task, seed, identity):
    print(task)
    if task in ['push-test-v2-goal-observable', 'pick-bar-v2-goal-observable']:
        env = env_dict[task](seed=seed, cm_offset=identity, cm_visible=False)
    elif task in ['brick-slide-v2-goal-observable']:
        env = env_dict[task](seed=seed)
    else:
        raise NotImplementedError
    return env

def main():
    
    task_name = 'brick_slide'
    
    task = 'brick-slide-v2-goal-observable'
    
    resolution = (640, 480)
    camera = 'corner'
    identity = 0.0
    p_identity = 0.05
    seed = 0
    policy = make_policy(task_name, task, p_identity)

    env = get_env(task, seed, identity)
    obs, image, depth, seg, cmat = init_env(env, resolution, camera, task)
    interaction = interact(env, obs, policy, resolution, camera)
    
    print(len(interaction))
    
    frameskip = 4
    imageio.mimsave(f"test/interaction.mp4", interaction[::frameskip], fps=16)
    
    
if __name__ == "__main__":

    main()