import argparse
import json
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.feat_encoder import FeatEncoder
from src.utils import method2dim, sample_n_frames

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return np.array(frames)

def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')
    parser.add_argument('-t', '--task', type=str, help='Path to the file containing the task')
    parser.add_argument('-e', '--encoder', type=str, help='Type of the embedding')
    parser.add_argument('-s', '--seed', type=int, help='Calculate score for all seeds')
    
    args = parser.parse_args()
    
    json_file = os.path.join('results_no_early_stop', f'{args.task}.json')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    if args.seed == -1:
        seeds = glob.glob(os.path.join('results_no_early_stop', args.task, '*'))
    else:
        seeds = [args.seed]
    
    best_plan_video_embedding_list = []
    interaction_video_embedding_list = []
    
    if args.encoder == 'dinov2':
        feat_encoder = FeatEncoder(args.encoder, device=device)
    else:
        def feat_encoder(video):
            return video
    # feat_dim = method2dim[args.encoder]
    
    if args.seed != -1:
        result = results[args.seed - 1000]
        success_list = np.array(result['success'])
    else:
        success_list = np.array([result['success'] for result in results]).flatten()
        
    for i in tqdm(range(len(seeds))):
        for j in range(16):
            best_plan_video = read_video(
                os.path.join('results_no_early_stop', args.task, str(1000+i), str(j), 'best_plan.mp4')
            )
            interaction_video = read_video(
                os.path.join('results_no_early_stop', args.task, str(1000+i), str(j), 'interaction.mp4')
            )

            best_plan_video_embedding = feat_encoder(best_plan_video)
            interaction_video_embedding = feat_encoder(sample_n_frames(interaction_video, 8))
            
            best_plan_video_embedding_list.append(torch.tensor(best_plan_video_embedding))
            interaction_video_embedding_list.append(torch.tensor(interaction_video_embedding))
        # print(best_plan_video_embedding.shape)
        # print(interaction_video_embedding.shape)
        # print("=====================================")
    
    best_plan_video_embedding_list = torch.stack(best_plan_video_embedding_list).view(len(seeds)*16, -1).to(torch.float32)
    interaction_video_embedding_list = torch.stack(interaction_video_embedding_list).view(len(seeds)*16, -1).to(torch.float32)
    
    print(best_plan_video_embedding_list.shape)
    print(interaction_video_embedding_list.shape)
        
    if args.encoder == 'l2':
        dist = torch.functional.norm(
            best_plan_video_embedding_list -
            interaction_video_embedding_list, 2, dim=1
        )
    else:
        dist = torch.nn.functional.cosine_similarity(
            best_plan_video_embedding_list,
            interaction_video_embedding_list
        )
    
    
    dist = dist.numpy()
    
    # print(dist)
    
    success_dist = dist[success_list]
    failure_dist = dist[~success_list]
    
    print("cos_sim for success: ", np.average(success_dist))
    print("cos_sim for failure: ", np.average(failure_dist))
    
    print("std for success: ", np.std(success_dist))
    print("std for failure: ", np.std(failure_dist))
    
    print("se for success: ", np.std(success_dist) / np.sqrt(len(success_dist)))
    print("se for failure: ", np.std(failure_dist) / np.sqrt(len(failure_dist)))
    
    success_dist_sorted = np.sort(success_dist)
    failure_dist_sorted = np.sort(failure_dist)
    
    success_rate_list = np.zeros(len(success_dist_sorted))
    success_se_list = np.zeros(len(success_dist_sorted))
    
    failure_rate_list = np.zeros(len(failure_dist_sorted))
    failure_se_list = np.zeros(len(failure_dist_sorted))
    
    for i, success_sim in enumerate(success_dist_sorted):
        success_rate = i / len(success_dist_sorted)
        array = np.concatenate((
            np.array([1] * i),
            np.array([0] * (len(success_dist_sorted) - i))
        ))
        success_se = np.std(array) / np.sqrt(len(array))
        success_rate_list[i] = success_rate
        success_se_list[i] = success_se
    
    for i, failure_sim in enumerate(failure_dist_sorted):
        failure_rate = i / len(failure_dist_sorted)
        array = np.concatenate((
            np.array([1] * i),
            np.array([0] * (len(success_dist_sorted) - i))
        ))
        failure_se = np.std(array) / np.sqrt(len(array))
        failure_rate_list[i] = failure_rate
        failure_se_list[i] = failure_se
    
    np.save(f'data/distance_{args.task}.npy', dist)
    
    
    plt.plot(success_dist_sorted, success_rate_list, label='Success')
    plt.fill_between(success_dist_sorted, success_rate_list - success_se_list, success_rate_list + success_se_list, alpha=0.3)
    
    plt.plot(failure_dist_sorted, failure_rate_list, label='Failure')
    plt.fill_between(failure_dist_sorted, failure_rate_list - failure_se_list, failure_rate_list + failure_se_list, alpha=0.3)
    
    plt.title(f'CDF of Cosine Similarity for {args.task}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('CDF')
    
    plt.legend()
    plt.savefig(f'figures/side/similarity_cdf_{args.task}.png')
    
if __name__ == '__main__':
    main()