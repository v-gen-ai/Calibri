from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import ImageReward as RM

class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.model_path = "ImageReward-v1.0"
        self.device = device
        self.dtype = dtype
        self.model = RM.load(self.model_path, device=device).eval().to(dtype=dtype)
        self.model.requires_grad_(False)
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        rewards = []
        for prompt,image in zip(prompts, images):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(reward)
        return rewards
