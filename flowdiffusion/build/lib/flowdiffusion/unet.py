from .guided_diffusion.guided_diffusion.unet import UNetModel, EncoderUNetModel
from torch import nn
import torch
from einops import repeat, rearrange


class UnetBridge(nn.Module):
    def __init__(self):
        super(UnetBridge, self).__init__()

        self.unet = UNetModel(
            image_size=(48, 64),
            in_channels=6,
            model_channels=160,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class UnetMW(nn.Module):
    def __init__(self):
        super(UnetMW, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetMWFlow(nn.Module):
    def __init__(self):
        super(UnetMWFlow, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=5,
            model_channels=128,
            out_channels=2,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, t, task_embed=None, **kwargs):
        f = x.shape[1] // 2 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', f=f) 
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class UnetMWFlow_proxy(nn.Module):
    def __init__(self):
        super(UnetMWFlow_proxy, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=3,
            model_channels=128,
            out_channels=2,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, task_embed=None, **kwargs):
        device = x.device
        t=torch.zeros(x.shape[0], dtype=torch.long).to(device)
        x = rearrange(x, 'b (f c) h w -> b c f h w', f=1) 
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class UnetThor(nn.Module):
    def __init__(self):
        super(UnetThor, self).__init__()

        self.unet = UNetModel(
            image_size=(64, 64),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
   
class EncoderSimple(nn.Module):
    def __init__(self):
        super(EncoderSimple, self).__init__()

        self.unet = EncoderUNetModel(
            image_size=128,
            in_channels=3,
            model_channels=64,
            out_channels=512,
            num_res_blocks=1,
            attention_resolutions=[],
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=3,
            pool='spatial_v2'
        )
        self.unet.convert_to_fp32()

    def forward(self, x, **kwargs):
        b = x.shape[0] # assume x: (b, c, f, h, w)
        t = torch.tensor([0]*b, device=x.device)
        out = self.unet(x, t, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class VideosEncoderSimple(nn.Module):
    def __init__(self):
        super(VideosEncoderSimple, self).__init__()
        out_size = 512

        self.unet = EncoderUNetModel(
            image_size=128,
            in_channels=3,
            model_channels=32,
            out_channels=out_size,
            num_res_blocks=1,
            attention_resolutions=[],
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            pool='spatial'
        )
        self.last_input = None
        self.last_output = None

        # random null embedding
        # self.null_embed = torch.randn(out_size)

    def forward(self, xs=None, **kwargs):
        # xs: (b, n, c, f, h, w) 
        # if xs[0] is None:
        #     b = len(xs)
        #     return self.null_embed.repeat(b, 1)
        ### -> (b*n, c, f, h, w)

        if self.last_input is not None and torch.equal(xs, self.last_input):
            return self.last_output
        else:
            self.last_input = xs


        device = next(self.unet.parameters()).device
        b, n, c, f, h, w = xs.shape
        xs = rearrange(xs, 'b n c f h w -> (b n) c f h w')
        t = torch.tensor([0]*b*n, device=device)

        out = self.unet(xs.to(device), t, **kwargs)
        out = rearrange(out, '(b n) d -> b n d', b=b, n=n)
        # mean pooling
        out = torch.mean(out, dim=1) # (b, d)
        self.last_output = out
        return out
    
class UnetMWSupp(nn.Module):
    def __init__(self):
        super(UnetMWSupp, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            use_scale_shift_norm=True
        )
        self.video_encoder = VideosEncoderSimple()
        
    def forward(self, x, t, task_embed=None, supp=None, **kwargs):
        # if supp[0] is None:
        #     print("no_supp_feat")
        device = x.device
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        if supp[0] is not None:
            supp_feat = self.video_encoder(supp) # (b, d)
            task_embed = torch.cat([task_embed, supp_feat[:, None]], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        # print(out.shape)
        return rearrange(out, 'b c f h w -> b (f c) h w')
  
class EncoderDINO(nn.Module):
    def __init__(self, out_dim=512):
        super(EncoderDINO, self).__init__()
        self.model = torch.hub.load(r'facebookresearch/dino:main', 'dino_vits16') # 384
        self.model.eval()
        # padding = 1
        # self.out_conv = nn.Conv1d(384, out_dim, kernel_size=1, stride=1, padding=1)
        self.out_conv = nn.Conv1d(384, out_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, **kwargs):
        # assume x: (b, c, f, h, w)
        b, n, f = x.shape[0], x.shape[1], x.shape[3]
        with torch.no_grad():
            x = rearrange(x, 'b n c f h w -> (b n f) c h w')
            x = self.model(x, **kwargs)
            x = rearrange(x, '(b n f) c -> (b n) c f', b=b, n=n, f=f)
        x = self.out_conv(x)
        x = rearrange(x, '(b n) c f -> b c (n f)', b=b, n=n)
        return self.pool(x).squeeze(2)

class UnetMWSuppDINO(nn.Module):
    def __init__(self):
        super(UnetMWSuppDINO, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            use_scale_shift_norm=True
        )
        self.video_encoder = EncoderDINO()
    def forward(self, x, t, task_embed=None, supp=None, **kwargs):
        # if supp[0] is None:
        #     print("no_supp_feat")
        device = x.device
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        if supp[0] is not None:
            supp_feat = self.video_encoder(supp) # (b, d)
            task_embed = torch.cat([task_embed, supp_feat[:, None]], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        # print(out.shape)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class EncoderDINOFlat(nn.Module):
    def __init__(self, out_dim=512, out_len=8):
        super(EncoderDINOFlat, self).__init__()
        self.model = torch.hub.load(r'facebookresearch/dino:main', 'dino_vits16') # 384
        self.model.eval()
        # padding = 1
        self.out_conv = nn.Conv1d(384*out_len, out_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, **kwargs):
        # assume x: (b, n, c, f, h, w)
        b, n, f = x.shape[0], x.shape[1], x.shape[3]
        with torch.no_grad():
            x = rearrange(x, 'b n c f h w -> (b n f) c h w')
            x = self.model(x, **kwargs)
            x = rearrange(x, '(b n f) c -> b (c f) n', b=b, n=n, f=f)
        x = self.out_conv(x)
        return self.pool(x).squeeze(2)

class UnetMWSuppDINOFlat(nn.Module):
    def __init__(self):
        super(UnetMWSuppDINOFlat, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            use_scale_shift_norm=True
        )
        self.video_encoder = EncoderDINOFlat()
    def forward(self, x, t, task_embed=None, supp=None, **kwargs):
        # if supp[0] is None:
        #     print("no_supp_feat")
        device = x.device
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        if supp[0] is not None:
            supp_feat = self.video_encoder(supp) # (b, d)
            task_embed = torch.cat([task_embed, supp_feat[:, None]], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        # print(out.shape)
        return rearrange(out, 'b c f h w -> b (f c) h w')
  
 
