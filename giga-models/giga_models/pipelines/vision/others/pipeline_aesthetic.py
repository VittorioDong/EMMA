import open_clip
import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop, InterpolationMode, Normalize, Resize

from ...pipeline import BasePipeline

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = Normalize(mean, std)
resize = Resize(size=(224), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)
centercrop = CenterCrop((224, 224))


class AestheticPipeline(BasePipeline):
    def __init__(self, model_path, pretrained_path):
        self.aesthetic_model = nn.Linear(768, 1)
        s = torch.load(model_path)
        self.aesthetic_model.load_state_dict(s)
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=pretrained_path, device='cpu')
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.aesthetic_model.to(self.device)
        self.clip_model.to(self.device)
        return self

    def __call__(self, image):
        image = self.clip_preprocess(image).unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            aesthetic_score = self.aesthetic_model(image_features)
        aesthetic_score = aesthetic_score.reshape(-1).item()
        return aesthetic_score
