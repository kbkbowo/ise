import argparse
import json
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    if args.seed:
        best_plan_video_files = glob.glob(os.path.join('results_no_early_stop', args.task, str(args.seed), '*/best_plan.mp4'))
        interaction_video_files = glob.glob(os.path.join('results_no_early_stop', args.task, str(args.seed), '*/interaction.mp4'))
    else:
        best_plan_video_files = glob.glob(os.path.join('results_no_early_stop', args.task, str(args.seed), '*/best_plan.mp4'))
        interaction_video_files = glob.glob(os.path.join('results_no_early_stop', args.task, str(args.seed), '*/interaction.mp4'))
    

    print(best_plan_video_files, interaction_video_files)
    
    best_plan_video_embedding_list = []
    interaction_video_embedding_list = []
    
    feat_encoder = FeatEncoder(args.encoder)
    feat_dim = method2dim[args.encoder]
    
    result = results[args.seed - 1000]
    
    success_list = np.array(result['success'])
        
    for i in tqdm(range(len(best_plan_video_files))):
        best_plan_video = read_video(os.path.join('results_no_early_stop', args.task, str(args.seed), str(i), 'best_plan.mp4'))
        interaction_video = read_video(os.path.join('results_no_early_stop', args.task, str(args.seed), str(i), 'interaction.mp4'))

        best_plan_video_embedding = feat_encoder(best_plan_video)
        interaction_video_embedding = feat_encoder(sample_n_frames(interaction_video, 8))
        
        best_plan_video_embedding_list.append(best_plan_video_embedding)
        interaction_video_embedding_list.append(interaction_video_embedding)
        
        # print(best_plan_video_embedding.shape)
        # print(interaction_video_embedding.shape)
        # print("=====================================")
        
    best_plan_video_embedding_list = np.array(best_plan_video_embedding_list)
    interaction_video_embedding_list = np.array(interaction_video_embedding_list)
    
    all_videos = np.concatenate((best_plan_video_embedding_list, interaction_video_embedding_list), axis=0)
    
    print(all_videos.shape)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    
    all_videos_tsne = tsne.fit_transform(all_videos)    
    
    labels = [
        'Interaction Video Success',
        'Interaction Video Failure',
        'Plan Video Success',
        'Plan Video Failure'
    ]
    
    print(all_videos_tsne.shape, type(all_videos_tsne))
    print(all_videos_tsne)
    
    for i in range(4):
        if i == 0: # interaction video success
            mask = np.concatenate((np.zeros(len(best_plan_video_embedding_list)), success_list), axis=0)
            mask[-len(interaction_video_embedding_list):] = success_list
        elif i == 1: # interaction video failure
            mask = np.concatenate((np.zeros(len(best_plan_video_embedding_list)), ~success_list), axis=0)
        elif i == 2: # plan video success
            mask = np.concatenate((success_list, np.zeros(len(interaction_video_embedding_list))), axis=0)
        elif i == 3: # plan video failure
            mask = np.concatenate((~success_list, np.zeros(len(interaction_video_embedding_list))), axis=0)
        else:
            raise ValueError('Invalid mask')
        
        mask = np.array(mask == 1)
        print(mask, all_videos_tsne[mask])
        plt.scatter(all_videos_tsne[mask, 0], all_videos_tsne[mask, 1], s=20, label=labels[i])
        
    # plt.scatter(all_videos_tsne[:, 0], all_videos_tsne[:, 1], s=50, label=labels[0])
        
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig('figures/tsne.png')
        
    print(len(best_plan_video_embedding_list))

    
    

if __name__ == '__main__':
    main()