from .goal_diffusion import GoalGaussianDiffusion, Trainer
from .goal_diffusion_v1 import GoalGaussianDiffusion as GoalGaussianDiffusion_v1, Trainer as Trainer_v1
from .goal_diffusion_policy import GoalGaussianDiffusion as GoalGaussianDiffusionPolicy, Trainer as TrainerPolicy
from .goal_diffusion_supp_dupsam import GoalGaussianDiffusion as GoalGaussianDiffusionSupp, Trainer as TrainerSupp
from .goal_diffusion_supp import GoalGaussianDiffusion as GoalGaussianDiffusionSuppOld, Trainer as TrainerSuppOld
from .goal_diffusion_sup_suc import GoalGaussianDiffusion as GoalGaussianDiffusionSuc, Trainer as TrainerSuc
from .diffusion_policy_baseline.unet import Unet1D, TransformerNet
from .unet import UnetMW as Unet
from .unet import UnetMWFlow as Unet_flow
from .unet import UnetThor as Unet_thor
from .unet import UnetBridge as Unet_bridge
from .unet import UnetMWSuppDINO as Unet_supp
from .unet import DiffusionAutoEncoderIndex as Unet_diffae_index
from .unet import DiffusionAutoEncoderIndexSuc, DiffusionAutoEncoderFeatSuc32
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms as T
from einops import rearrange
import torch
from PIL import Image
from torch import nn
import numpy as np

def get_video_model_diffae_feat_suc(ckpt_dir='../ckpts/0827_implicit_ret', milestone=18, timestep=100, video_enc_dim=4096):
    unet = DiffusionAutoEncoderFeatSuc32(video_enc_dim=video_enc_dim)
    # unet = DiffusionAutoEncoderIndexSuc(embed_dim=64)
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    sample_per_seq = 8
    target_size = (32, 32)

    diffusion = GoalGaussianDiffusionSuc(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    trainer = TrainerSuc( 
        diffusion_model=diffusion, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        train_set=[1], 
        valid_set=[1], 
        results_folder=ckpt_dir,
        fp16 =True, 
        amp=True, 
    )
    trainer.load(milestone)
    return trainer

def get_video_model_diffae_index_suc(ckpt_dir='../ckpts/diffae_0819_most_suc', milestone=20, timestep=100):
    unet = DiffusionAutoEncoderIndexSuc()
    # unet = DiffusionAutoEncoderIndexSuc(embed_dim=64)
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    sample_per_seq = 8
    target_size = (128, 128)

    diffusion = GoalGaussianDiffusionSuc(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    trainer = TrainerSuc( 
        diffusion_model=diffusion, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        train_set=[1], 
        valid_set=[1], 
        results_folder=ckpt_dir,
        fp16 =True, 
        amp=True, 
    )
    trainer.load(milestone)
    return trainer

def get_video_model_diffae(ckpt_dir='../ckpts/diffae_0709', milestone=34, timestep=100):
    unet = Unet_diffae_index()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    sample_per_seq = 8
    target_size = (128, 128)

    diffusion = GoalGaussianDiffusionSuppOld(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    trainer = TrainerSuppOld( 
        diffusion_model=diffusion, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        train_set=[1], 
        valid_set=[1], 
        results_folder=ckpt_dir,
        fp16 =True, 
        amp=True, 
    )
    trainer.load(milestone)
    return trainer


def get_video_model_supp(ckpt_dir='../ckpts/mw_supp_0418', milestone=22, timestep=100):
    unet = Unet_supp()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    sample_per_seq = 8
    target_size = (128, 128)

    diffusion = GoalGaussianDiffusionSupp(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    trainer = TrainerSupp( 
        diffusion_model=diffusion, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        train_set=[1], 
        valid_set=[1], 
        results_folder=ckpt_dir,
        fp16 =True, 
        amp=True, 
    )
    trainer.load(milestone)
    return trainer



def get_diffusion_policy_T(ckpt_dir='../ckpts/diffusion_policy_T', milestone=1, sampling_timesteps=10):
    unet = TransformerNet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
        model=unet,
        image_size=10,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[0],
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =2500,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )

    trainer.load(milestone)
    return trainer

class DiffusionPolicy_T():
    def __init__(self, milestone=10, amp=True, sampling_timesteps=10):
        self.policy = get_diffusion_policy_T(milestone=milestone, sampling_timesteps=sampling_timesteps)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])

    def __call__(self,
                obs: np.array,
                task: str,
            ):
        device = self.policy.device
        obs = torch.stack([self.transform(Image.fromarray(o)) for o in obs], dim=0).float().to(device).unsqueeze(0)
        with torch.no_grad():
            return self.policy.sample(obs, [task]).cpu().squeeze(0).numpy()

def get_diffusion_policy(ckpt_dir='../ckpts/diffusion_policy', milestone=1, sampling_timesteps=10):
    unet = Unet1D()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
        model=unet,
        image_size=16,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[0],
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =2500,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )

    trainer.load(milestone)
    return trainer

class DiffusionPolicy():
    def __init__(self, milestone=10, amp=True, sampling_timesteps=10):
        self.policy = get_diffusion_policy(milestone=milestone, sampling_timesteps=sampling_timesteps)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])

    def __call__(self,
                obs: np.array,
                task: str,
            ):
        device = self.policy.device
        obs = torch.stack([self.transform(Image.fromarray(o)) for o in obs], dim=0).float().to(device).unsqueeze(0)
        with torch.no_grad():
            return self.policy.sample(obs, [task]).cpu().squeeze(0).numpy()


def get_video_model(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, timestep=100):
    unet = Unet_flow() if flow else Unet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (128, 128)
    channels = 3 if not flow else 2

    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer
        
def get_video_model_thor(ckpts_dir='../ckpts/ithor', milestone=30):
    unet = Unet_thor()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (64, 64)
    channels = 3

    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=100,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer

def get_video_model_bridge(ckpts_dir='../ckpts/bridge', milestone=42):
    unet = Unet_bridge()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (48, 64)
    channels = 3

    diffusion = GoalGaussianDiffusion_v1(
        model=unet,
        image_size=target_size,
        channels=channels*(sample_per_seq-1),
        timesteps=100,
        sampling_timesteps=100,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer_v1(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer

def pred_supp_embedding(model, supp=None):
    # supp: (1, n, c, f ,h, w)
    device = "cuda"
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    # process supp
    if supp is not None:
        n, c, f, h, w = supp.shape
        supp = rearrange(supp, 'n c f h w -> (n f) h w c')
        supp_processed = []
        for i in range(n*f):
            supp_processed.append(transform(Image.fromarray(supp[i])).unsqueeze(0))
        supp = torch.cat(supp_processed, dim=0)
        supp = rearrange(supp, '(n f) c h w -> n c f h w', n=n, f=f).unsqueeze(0)
        supp.to(device)
    else:
        supp = [None]
    supp_model = model.model.model.video_encoder
    supp_model.to(device)
    # print patrameters device
    supp = supp.to(device)
    supp_model.eval()
    with torch.no_grad():
        return supp_model(supp)
    # pad the image back to original shape (both sides)

def pred_video_supp(model, frame_0, task, supp=None, guidance=0.0, pg_attract=None, pg_repulse=None):
    # supp: (1, n, c, f ,h, w)
    device = model.device
    original_shape = frame_0.shape
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64

    channels = 3 
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    # process supp
    def process_vid(vid): # (1, n, c, f, h, w)
        n, c, f, h, w = vid.shape
        vid = rearrange(vid, 'n c f h w -> (n f) h w c')
        vid_processed = []
        for i in range(n*f):
            vid_processed.append(transform(Image.fromarray(vid[i])).unsqueeze(0))
        vid = torch.cat(vid_processed, dim=0)
        vid = rearrange(vid, '(n f) c h w -> n c f h w', n=n, f=f)
        vid = vid.to(device)
        return vid

    if supp is not None:
        supp = process_vid(supp).unsqueeze(0)
    else:
        supp = [None]
    if pg_attract is not None:
        pg_attract = process_vid(pg_attract)
        # save video
        import imageio
        imageio.mimsave("pg_attract.mp4", (pg_attract.squeeze(0).numpy().transpose(1, 2, 3, 0)*255).astype('uint8'))
    if pg_repulse is not None:
        pg_repulse = process_vid(pg_repulse)

    text = [task]
    preds = rearrange(model.sample(image.to(device), text, supp, guidance_weight=guidance, attraction=pg_attract, repulsion=pg_repulse).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    # pad the image back to original shape (both sides)
    images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    return (images.numpy()*255).astype('uint8')

def pred_video_ckm(model, frame_0, task, supp=None, guidance=0.0, attraction=None, repulsion=None, try_steps=1, try_samples=10, cont_noise=0.8, sim_kernel=None):
    # supp: (1, n, c, f ,h, w)
    device = model.device
    original_shape = frame_0.shape
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64

    channels = 3 
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    # process supp
    def process_vid(vid): # (1, n, c, f, h, w)
        c, f, h, w = vid.shape
        vid = rearrange(vid, 'c f h w -> f h w c')
        vid_processed = []
        for i in range(f):
            vid_processed.append(transform(Image.fromarray(vid[i])).unsqueeze(0))
        vid = torch.cat(vid_processed, dim=0)
        vid = rearrange(vid, 'f c h w ->c f h w', f=f)
        vid = vid.to(device)
        return vid

    if supp is not None:
        supp = process_vid(supp).unsqueeze(0)
    else:
        supp = [None]
    if attraction is not None:
        attraction = process_vid(attraction)
        # save video
        # import imageio
        # imageio.mimsave("pg_attract.mp4", (pg_attract.squeeze(0).numpy().transpose(1, 2, 3, 0)*255).astype('uint8'))
    if repulsion is not None:
        repulsion = process_vid(repulsion)

    text = [task]
    preds, samples, best_idx = model.ckm_sample(image.to(device), text, supp, guidance_weight=guidance, attraction=attraction, repulsion=repulsion, num_samples=try_samples, num_steps=try_steps, cont_noise=cont_noise, sim_kernel=sim_kernel)
    preds = rearrange(preds.cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    samples = rearrange(torch.tensor(samples), "s (f c) w h -> s f c w h", c=channels)
    # duplicate the first frame
    image = image.repeat(samples.shape[0], samples.shape[1], 1, 1, 1)
    samples = torch.cat([image, samples], dim=1)
    print(samples.shape)
    # pad the image back to original shape (both sides)
    padded_images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    return (padded_images.numpy()*255).astype('uint8'), (preds.numpy()*255).astype('uint8'), (samples.numpy()*255).astype('uint8'), best_idx




def pred_video(model, frame_0, task, flow=False):
    device = model.device
    original_shape = frame_0.shape
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64

    channels = 3 if not flow else 2
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image.to(device), text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    if not flow:
        preds = torch.cat([image, preds], dim=0)
    # pad the image back to original shape (both sides)
    images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    return images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')

def pred_video_thor(model, frame_0, task):
    channels=3
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image, text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    return (preds.numpy()*255).astype('uint8')

def pred_video_bridge(model, frame_0, task):
    channels=3
    transform = T.Compose([
        T.Resize((48, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image, text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    return (preds.numpy()*255).astype('uint8')