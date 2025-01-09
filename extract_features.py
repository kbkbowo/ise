
import glob
import imageio
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms as T
import torch
import numpy as np
import os
from tqdm import tqdm
import json
# from s3d.s3dg import S3D
# from jepa.models.vision_transformer import VisionTransformer as JEPA
from argparse import ArgumentParser
from transformers import AutoImageProcessor, AutoModel
from r3m import load_r3m
import vc_models
from vc_models.models.vit import model_utils

from torch import nn
from torchvision import models
from einops import rearrange

data_paths = glob.glob("/tmp2/pochenko/temp/mw_temp/images0822/*/*/*/video.mp4")
n_frames = 8
transform = T.Compose([
    T.CenterCrop(128),
    T.Resize((224, 224)),
])
out_root = "./extrated_features"

def pre_transform(frames):
    transform = T.Compose([
        T.CenterCrop(128),
        T.Resize((224, 224)),
    ])
    return [transform(Image.fromarray(frame)) for frame in frames]


def get_video(data_path):
    video = imageio.get_reader(data_path)
    # get len
    vidlen = video.count_frames()
    samples = []
    for i in range(n_frames-1):
        samples.append(int(i*(vidlen-1)/(n_frames-1)))
    samples.append(vidlen-1)
    frames = [video.get_data(s) for s in samples]
    # transformations
    frames = [transform(Image.fromarray(frame)) for frame in frames]
    return frames

class CustomVideoEncoder(nn.Module): # stasks resnet18 features -> (video, )
    def __init__(self, in_res=128, in_frames=8, in_channels=3, out_dim=512):
        super(CustomVideoEncoder, self).__init__()
        self.in_frames = in_frames
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()
        self.out_proj = nn.Linear(512 * in_frames, out_dim)

        self.learned_temperature = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        # in: B C T H W
        # out: B C
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resnet(x)
        x = rearrange(x, '(b t) c -> b (c t)', t=self.in_frames)
        x = self.out_proj(x)
        return x / torch.norm(x, dim=-1, keepdim=True)

    def compute_loss(self, x, y):
        # in: B C T H W
        # out: B C
        x_feat = self(x)
        y_feat = self(y)

        x_feat = x_feat 
        y_feat = y_feat 

        # cosine similarity
        sim = x_feat @ y_feat.T
        
        logits = sim * torch.exp(self.learned_temperature)
        labels = torch.arange(x.size(0)).to(logits.device)
        loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)) / 2
        return loss

def get_custom_model():
    model = CustomVideoEncoder()
    model.load_state_dict(torch.load("pretrained/best_model.pth"))
    return model

def get_custom_feature(frames, model=None, transform=None):
    model = get_custom_model() if model is None else model
    transform = T.Compose([
        T.ToTensor()
    ]) if transform is None else transform
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames, dim=1).unsqueeze(0)
    with torch.no_grad():
        output = model(frames).flatten(0).cpu().numpy()
    return output

def get_clip_feature(frames, processor=None, model=None):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") if processor is None else processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") if model is None else model
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.get_image_features(**inputs).flatten(0).cpu().numpy()
    return output

def get_s3d_feature(frames, model=None, transform=None):
    model = S3D() if model is None else model
    transform = T.Compose([
        T.ToTensor()
    ]) if transform is None else transform
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames, dim=1)
    with torch.no_grad():
        output = model(frames.unsqueeze(0))['mixed_5c'].flatten(0).cpu().numpy()
    return output

def get_jepa_feature(frames, model=None, transform=None):
    model = JEPA.from_pretrained("nielsr/vit-large-patch16-v-jepa") if model is None else model
    transform = T.Compose([
        T.ToTensor()
    ]) if transform is None else transform
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames, dim=1)
    with torch.no_grad():
        output = model(frames.unsqueeze(0)).flatten(0).cpu().numpy()
    return output

def get_dinov2_feature(frames, processor=None, model=None):
    model = AutoModel.from_pretrained('facebook/dinov2-base') if model is None else model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base') if processor is None else processor
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    # model.config.return_dict = False
    with torch.no_grad():
        # traced_model = torch.jit.trace(model, [inputs.pixel_values])
        # output = traced_model(inputs.pixel_values)[1].flatten(0).cpu().numpy()

        output = model(**inputs)['pooler_output'].flatten(0).cpu().numpy()
    return output

def get_r3m_feature(frames, model=None, transform=None):
    model = load_r3m("resnet50") if model is None else model
    transform = T.Compose([
        T.ToTensor()
    ]) if transform is None else transform
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames, dim=0)
    with torch.no_grad():
        output = model(frames).flatten(0).cpu().numpy()
    return output

def get_vc1_feature(frames, model=None, transform=None):
    model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME) if model is None else model, None, None, None
    transform = T.Compose([
        T.ToTensor()
    ]) if transform is None else transform
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames, dim=0)
    with torch.no_grad():
        output = model(frames).flatten(0).cpu().numpy()
    return output

# def get_vc1_feature(frames, model=None, transform=None):

class FeatEncoder:
    def __init__(self, method='clip'):
        self.method = method

        if method == 'clip':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif method == 'dinov2':
            self.model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        elif method == 'r3m':
            self.model = load_r3m("resnet50").cuda()
            self.transform = T.Compose([
                T.ToTensor()
            ])
        elif method == 'vc1':
            self.model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
            self.transform = T.Compose([
                T.ToTensor()
            ])
        elif method == 'custom':
            self.model = CustomVideoEncoder()
            self.transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor()
            ])
        else:
            raise NotImplementedError
    def __call__(self, frames):
        frames = pre_transform(frames)
        if self.method == 'clip':
            output = get_clip_feature(frames, self.processor, self.model)
        elif self.method == 'dinov2':
            output = get_dinov2_feature(frames, self.processor, self.model)
        elif self.method == 'r3m':
            output = get_r3m_feature(frames, self.model, self.transform)
        elif self.method == 'vc1':
            output = get_vc1_feature(frames, self.model, self.transform)
        elif self.method == 'custom':
            output = get_custom_feature(frames, self.model, self.transform)
        else:
            raise NotImplementedError
        return output


def encode_features(method='clip', out_root='./extrated_features', data_paths=None):
    print(data_paths)
    print(method)
    out_root = f"./{data_paths}"# if exist: skip
    if os.path.exists(f"{out_root}/{method}/all_feats.npy"):
        print(f"Skipping {method}")
        return
    data_paths = glob.glob(f"./{data_paths}/*/*/*/video.mp4")

    print(out_root)


    if method == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        out_dir = f"{out_root}/clip"
    elif method == 's3d':
        model = S3D('s3d/s3d_dict.npy', 512)
        model.load_state_dict(torch.load('s3d/s3d_howto100m.pth'))
        transform = T.Compose([
            transform, 
            T.ToTensor()
        ])
        out_dir = f"{out_root}/s3d"
    elif method == 'jepa':
        model = JEPA.from_pretrained("nielsr/vit-large-patch16-v-jepa")
        out_dir = f"{out_root}/jepa"
        transform = T.Compose([
            transform,
            T.Resize((128, 128)),
            T.ToTensor()
        ])
    elif method == 'dinov2':
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        out_dir = f"{out_root}/dinov2"
    elif method == 'r3m':
        model = load_r3m("resnet50").cuda()
        out_dir = f"{out_root}/r3m"
        transform = T.Compose([
            transform,
            T.ToTensor()
        ])
    elif method == 'vc1':
        model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        out_dir = f"{out_root}/vc1"
        transform = T.Compose([
            transform,
            T.ToTensor()
        ])
    elif method == 'custom':
        model = CustomVideoEncoder()
        out_dir = f"{out_root}/custom"
        transform = T.Compose([
            transform,
            T.Resize((128, 128)),
            T.ToTensor()
        ])
    else:
        raise NotImplementedError

    all_feats = {}

    for data_path in tqdm(data_paths):
        frames = get_video(data_path)
        # with torch.amp.autocast("cuda"):
        if method == 'clip':
            output = get_clip_feature(frames, processor, model)
        elif method == 's3d':
            output = get_s3d_feature(frames, model, transform)
        elif method == 'jepa':
            output = get_jepa_feature(frames, model, transform)
        elif method == 'dinov2':
            output = get_dinov2_feature(frames, processor, model)
        elif method == 'r3m':
            output = get_r3m_feature(frames, model, transform)
        elif method == 'vc1':
            output = get_vc1_feature(frames, model, transform)
        elif method == 'custom':
            output = get_custom_feature(frames, model, transform)
        else:
            raise NotImplementedError

        # save output
        seed = data_path.split('/')[-4]
        p = data_path.split('/')[-3]
        cm = data_path.split('/')[-2]

        all_feats[f"{seed}_{p}_{cm}"] = output.tolist()

    # save all_feats
    os.makedirs(out_dir, exist_ok=True)
    # with open(f"{out_dir}/all_feats.json", "w") as f:
    #     json.dump(all_feats, f, indent=2)
    np.save(f"{out_dir}/all_feats.npy", all_feats)
    
    

        
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default='clip')
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--data_paths', type=str)
    args = parser.parse_args()
    encode_features(args.method, args.out_root, args.data_paths)