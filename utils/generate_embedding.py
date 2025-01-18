import argparse
import json
import os


import glob
import imageio
from PIL import Image
from torchvision import transforms as T
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from torch import nn
from torchvision import models
from einops import rearrange

n_frames = 8
transform = T.Compose([
    T.CenterCrop(128),
    T.Resize((224, 224)),
])

def pre_transform(frames):
    transform = T.Compose([
        T.CenterCrop(128),
        T.Resize((224, 224)),
    ])
    return [transform(frame) for frame in frames]


def get_video(data_path):
    video = imageio.get_reader(data_path)
    # get len
    vidlen = video.count_frames()
    samples = []
    for i in range(n_frames-1):
        samples.append(int(i*(vidlen-1)/(n_frames-1)))
    samples.append(vidlen-1)
    frames = [video.get_data(s) for s in samples]
    frames = [transform(Image.fromarray(frame)) for frame in frames]
    return frames

def get_dinov2_feature(frames, processor=None, model=None, device='cpu'):
    model = AutoModel.from_pretrained('facebook/dinov2-base') if model is None else model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base') if processor is None else processor
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():

        output = model(**inputs)['pooler_output'].flatten(0).cpu().numpy()
    return output

class FeatEncoder:
    def __init__(self, method='clip', device='cpu'):
        self.method = method

        if method == 'dinov2':
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        
        self.device = device
        
    def __call__(self, frames):
        frames = pre_transform(frames)
        if self.method == 'dinov2':
            output = get_dinov2_feature(frames, self.processor, self.model, self.device)
        return output
    
    

def main():
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = 'cpu'
    parser = argparse.ArgumentParser(description='Generate dataset json')
    parser.add_argument('-r', '--root', type=str, required=True)
    
    args = parser.parse_args()
    
    json_file = os.path.join(args.root, 'dataset.json')
    
    with open(json_file, 'r') as f:
        files = json.load(f)['files']
        
    encoder = FeatEncoder(method='dinov2', device=device)
    
    feature_list = []
    
    # files = files[:10]
    
    for file in tqdm(files):
        frames = get_video(file)
        feature = encoder(frames)
        # print(feature.shape)
        feature_list.append(feature)

    feature_list = np.array(feature_list)
    print(feature_list.shape)
    
    np.save(os.path.join(args.root, 'features.npy'), feature_list)

if __name__ == '__main__':
    main()