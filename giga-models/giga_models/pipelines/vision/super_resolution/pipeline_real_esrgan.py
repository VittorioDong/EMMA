import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer

from ...pipeline import BasePipeline


class RealESRGANPipeline(BasePipeline):
    def __init__(self, model_paths, model_names):
        self.device = 'cpu'
        self.upsamplers = []
        for model_path, model_name in zip(model_paths, model_names):
            if model_name == 'x2plus':
                scale = 2
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
            elif model_name == 'x4plus':
                scale = 4
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            elif model_name == 'x4plus_anime_6B':
                scale = 4
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            else:
                assert False
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=192,
                tile_pad=8,
                device='cpu',
            )
            if scale == 2:
                self.upsamplers = [upsampler] + self.upsamplers
            else:
                self.upsamplers += [upsampler]

    def to(self, device):
        self.device = device
        for upsampler in self.upsamplers:
            upsampler.device = device
            upsampler.model.to(device)
        return self

    def half(self):
        for upsampler in self.upsamplers:
            upsampler.half = True
            upsampler.model.half()
        return self

    def __call__(self, image, scale=4):
        image = np.array(image)[:, :, ::-1]
        dst_height = int(image.shape[0] * scale)
        dst_width = int(image.shape[1] * scale)
        while True:
            upsampler = None
            for i in range(len(self.upsamplers)):
                if int(image.shape[0] * self.upsamplers[i].scale) < dst_height:
                    continue
                if int(image.shape[1] * self.upsamplers[i].scale) < dst_width:
                    continue
                upsampler = self.upsamplers[i]
                break
            if upsampler is None:
                upsampler = self.upsamplers[-1]
            image = upsampler.enhance(image)[0]
            if image.shape[0] >= dst_height and image.shape[1] >= dst_width:
                break
        image = Image.fromarray(image[:, :, ::-1])
        if image.height != dst_height or image.width != dst_width:
            image = image.resize((dst_width, dst_height), resample=Image.LANCZOS)
        return image
