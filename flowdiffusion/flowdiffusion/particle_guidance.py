import torch
from einops import rearrange

def load_dino_model(model_type="dino_vits16"):
    model = torch.hub.load('facebookresearch/dino:main', model_type)
    model.eval()
    # the model takes in [B, C, H, W] images and returns the embeddings
    # wrap a nn.module around the model to make it compatible with the video ([B, C, T, H, W]) input
    class DinoWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            b = x.shape[0]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.model(x)
            x = rearrange(x, '(b t) c -> b t c', b=b)
            return x
    model.eval()
    return DinoWrapper(model)

class ParticleGuidance():
    def __init__(self, feature_model=None, device="cuda"):
        if feature_model is None:
            self.feature_model = torch.nn.Identity()
        else:
            self.feature_model = feature_model
        self.feature_model.to(device)
        self.attraction = None
        self.repulsion = None
        self.device = device

    def criterion(self, input_features, target_features, h): # RBF kernel
        diff = input_features.flatten(1) - target_features.flatten(1)
        # out = torch.exp(-torch.norm(diff, dim=-1) / h)
        # simply mse
        out = -torch.nn.functional.mse_loss(input_features, target_features)
        return out

    def set_particles(self, attraction=None, repulsion=None):
        if attraction is not None:
            self.set_attraction(attraction)
        if repulsion is not None:
            self.set_repulsion(repulsion)
    
    def set_attraction(self, attraction): # (B, C, T, H, W)
        self.attraction = self.feature_model(attraction[:, :, 1:].clone().detach().to(self.device)) 
    
    def set_repulsion(self, repulsion): # (B, C, T, H, W)
        self.repulsion = self.feature_model(repulsion[:, :, 1:].clone().detach().to(self.device))

    def calc_grad_from_features(self, input_vid, input_features, attraction_features, h=10.0):
        loss = self.criterion(input_features, attraction_features, h=h)
        grad = torch.autograd.grad(loss, input_vid, retain_graph=True)[0]
        return grad

    def get_grad(self, input_vid, h=1.0, alpha=8.0): # (B, C, T, H, W)
        if self.attraction is None and self.repulsion is None:
            # return zero grad
            return {"attraction_grad": torch.zeros_like(input_vid), "repulsion_grad": torch.zeros_like(input_vid)}
        input_vid = torch.nn.Parameter(input_vid).to(self.device)
        input_features = self.feature_model(input_vid)
        out_dict = {}
        if self.attraction is not None:
            attraction_grad = self.calc_grad_from_features(input_vid, input_features, self.attraction, h)
            out_dict["attraction_grad"] = attraction_grad * alpha
        if self.repulsion is not None:
            repulsion_grad = -self.calc_grad_from_features(input_vid, input_features, self.repulsion, h)
            out_dict["repulsion_grad"] = repulsion_grad * alpha
        return out_dict

    def inner_gd_steps(self, input_vid, n_steps=10, h=1.0, alpha=8.0):
        for _ in range(n_steps):
            grads = self.get_grad(input_vid, h=h, alpha=alpha)
            input_vid = input_vid + torch.sum(torch.stack(list(grads.values())), dim=0)
        return input_vid