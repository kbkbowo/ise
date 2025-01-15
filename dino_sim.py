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
    
    parser.add_argument('-v1', '--video1', type=str, help='Path to the first video')
    parser.add_argument('-v2', '--video2', type=str, help='Path to the second video')
    
    args = parser.parse_args()
    
    feat_encoder = FeatEncoder('dinov2')
    
    video1 = read_video(args.video1)
    video2 = read_video(args.video2)
    
    video1_embedding = feat_encoder(sample_n_frames(video1, 8))
    video2_embedding = feat_encoder(sample_n_frames(video2, 8))
    
    video1_embedding, video2_embedding = torch.tensor(video1_embedding).unsqueeze(0), torch.tensor(video2_embedding).unsqueeze(0)
    
    print(video1_embedding.shape, video2_embedding.shape)
    
    similarity = torch.nn.functional.cosine_similarity(video1_embedding, video2_embedding)
    
    print(similarity)
    

if __name__ == '__main__':
    main()