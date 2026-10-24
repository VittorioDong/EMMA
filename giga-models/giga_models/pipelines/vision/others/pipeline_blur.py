import random
from typing import List, Optional

import attrs
import cv2
import numpy as np
from PIL import Image

from ...pipeline import BasePipeline


class BlurPipeline(BasePipeline):
    def __init__(self, strength) -> None:
        self.set_strength(strength)

    def set_strength(self, strength: str):
        blur_config = BilateralOnlyBlurAugmentorConfig[strength]
        self.set_blur_config(blur_config)

    def set_blur_config(self, config):
        probabilities = [combo.probability for combo in config.blur_combinations]
        total_prob = sum(probabilities)
        assert abs(total_prob - 1.0) < 1e-6, f'Probabilities must sum to 1.0, got {total_prob}'

        self.blur_combinations = config.blur_combinations
        self.downscale_factor = config.downscale_factor
        self.probabilities = probabilities
        self._set_blur_instances()

    def _set_blur_instances(self):
        if not self.blur_combinations:
            return
        self.blur_combinations_instances = []

        for blur_combination in self.blur_combinations:
            blur_mapping = {
                'gaussian': (GaussianBlur, blur_combination.gaussian_blur),
                'guided': (GuidedFilter, blur_combination.guided_filter),
                'bilateral': (BilateralFilter, blur_combination.bilateral_filter),
                'median': (MedianBlur, blur_combination.median_blur),
                'log': (LaplacianOfGaussian, blur_combination.log),
                'anisotropic': (AnisotropicDiffusion, blur_combination.anisotropic_diffusion),
            }

            cur_instances = []
            for blur_type in blur_combination.blur_types:
                assert blur_type in blur_mapping, f'Unknown {blur_type}. Needs to correct blur_type or blur_mapping.'

                blur_class, blur_config = blur_mapping[blur_type]
                cur_instances.append(blur_class(blur_config))

            self.blur_combinations_instances.append(cur_instances)

        assert len(self.blur_combinations_instances) == len(
            self.blur_combinations
        ), 'Number of blur_combinations_instances needs to match number of blur_combinations.'

    def __call__(self, images):
        frames = [np.array(image) for image in images]
        frames = np.stack(frames, axis=0)
        frames = frames.transpose((3, 0, 1, 2))  # (C, T, H, W)
        blur_instances = random.choices(self.blur_combinations_instances, weights=self.probabilities, k=1)[0]
        H, W = frames.shape[2], frames.shape[3]
        downscale_factor = random.choice(self.downscale_factor)
        if downscale_factor > 1:
            frames = [
                cv2.resize(_image_np, (W // downscale_factor, H // downscale_factor), interpolation=cv2.INTER_AREA)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))

        for ins in blur_instances:
            frames = ins(frames)

        if downscale_factor > 1:
            frames = [cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_LINEAR) for _image_np in frames.transpose((1, 2, 3, 0))]
            frames = np.stack(frames).transpose((3, 0, 1, 2))
        frames = frames.transpose((1, 2, 3, 0))
        frames = [Image.fromarray(frame) for frame in frames]
        return frames


@attrs.define
class GaussianBlurConfig:
    """Configuration for Gaussian blur."""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 25
    sigmaX: float = 12.5

    # if use_random is True, then optionally define the range
    ksize_min: int = 21
    ksize_max: int = 29
    sigmaX_min: float = 10.5
    sigmaX_max: float = 14.5


@attrs.define
class GuidedFilterConfig:
    """Configuration for Guided filter."""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    radius: int = 45
    eps: float = 0.15
    scale: int = 10

    # if use_random is True, then optionally define the range
    radius_min: int = 41
    radius_max: int = 49
    eps_min: float = 0.1
    eps_max: float = 0.2
    scale_min: int = 3
    scale_max: int = 18


@attrs.define
class BilateralFilterConfig:
    """Configuration for Bilateral filter."""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    d: int = 30
    sigma_color: int = 150
    sigma_space: int = 100
    iter: int = 1

    # if use_random is True, then optionally define the range
    d_min: int = 15
    d_max: int = 50
    sigma_color_min: int = 100
    sigma_color_max: int = 300
    sigma_space_min: int = 50
    sigma_space_max: int = 150
    iter_min: int = 1
    iter_max: int = 4


@attrs.define
class MedianBlurConfig:
    """Configuration for Median blur."""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 11

    # if use_random is True, then optionally define the range
    ksize_min: int = 9
    ksize_max: int = 15


@attrs.define
class LaplacianOfGaussianConfig:
    """Configuration for LoG filter."""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 5
    sigma: float = 1.4
    binarize: bool = False
    threshold: float = 0.0

    # if use_random is True, then optionally define the range
    ksize_min: int = 3
    ksize_max: int = 7
    sigma_min: float = 0.5
    sigma_max: float = 3.0
    threshold_min: float = 10.0
    threshold_max: float = 30.0


@attrs.define
class AnisotropicDiffusionConfig:
    """Configuration for Anisotropic Diffusion."""

    use_random: bool = False
    alpha: float = 0.25
    K: float = 0.15
    niters: int = 12

    # if use_random is True, then optionally define the range
    alpha_min: float = 0.2
    alpha_max: float = 0.3
    K_min: float = 0.1
    K_max: float = 0.2
    niters_min: int = 10
    niters_max: int = 14


@attrs.define
class BlurCombinationConfig:
    """Configuration for a combination of blurs with associated probability."""

    # list of choices are:  ["gaussian", "guided", "bilateral", "median", "log", "anisotropic"]
    # the corresponding config must be defined for each item in this blur_types list
    blur_types: List[str]
    probability: float
    gaussian_blur: Optional[GaussianBlurConfig] = None
    guided_filter: Optional[GuidedFilterConfig] = None
    bilateral_filter: Optional[BilateralFilterConfig] = None
    median_blur: Optional[MedianBlurConfig] = None
    log: Optional[LaplacianOfGaussianConfig] = None
    anisotropic_diffusion: Optional[AnisotropicDiffusionConfig] = None


@attrs.define
class BlurAugmentorConfig:
    """Configuration for blur augmentation with multiple combinations."""

    # probabilities from the list of combinations should add up to 1.0
    blur_combinations: List[BlurCombinationConfig] = []
    downscale_factor: List[int] = [1]


# random blur for training the VisControl
random_blur_config = BlurAugmentorConfig(
    downscale_factor=list(range(1, 5)),
    blur_combinations=[
        BlurCombinationConfig(
            blur_types=['bilateral'],
            probability=0.5,
            bilateral_filter=BilateralFilterConfig(use_random=True),
        ),
        BlurCombinationConfig(
            blur_types=['gaussian'],
            probability=0.3,
            gaussian_blur=GaussianBlurConfig(use_random=True),
        ),
        BlurCombinationConfig(
            blur_types=['bilateral', 'gaussian'],
            probability=0.2,
            bilateral_filter=BilateralFilterConfig(use_random=True),
            gaussian_blur=GaussianBlurConfig(use_random=True),
        ),
    ],
)

# predefined BilateralFilterConfig with different strength level
NoFilterConfig = BilateralFilterConfig(use_random=False, d=1, sigma_color=1, sigma_space=1, iter=1)

LowBilateralFilterConfig = BilateralFilterConfig(use_random=False, d=15, sigma_color=100, sigma_space=50, iter=1)

MediumBilateralFilterConfig = BilateralFilterConfig(use_random=False, d=30, sigma_color=150, sigma_space=100, iter=1)

HighBilateralFilterConfig = BilateralFilterConfig(use_random=False, d=50, sigma_color=300, sigma_space=150, iter=1)

BilateralOnlyBlurAugmentorConfig = {}
for blur_strength, blur_config in zip(
    ['none', 'very_low', 'low', 'medium', 'high', 'very_high'],
    [
        NoFilterConfig,
        LowBilateralFilterConfig,
        LowBilateralFilterConfig,
        MediumBilateralFilterConfig,
        HighBilateralFilterConfig,
        HighBilateralFilterConfig,
    ],
):
    BlurConfig = BlurCombinationConfig(
        blur_types=['bilateral'],
        probability=1.0,
        bilateral_filter=blur_config,
    )
    downscale_factor = {
        'none': 1,
        'very_low': 1,
        'low': 4,
        'medium': 2,
        'high': 1,
        'very_high': 4,
    }
    BilateralOnlyBlurAugmentorConfig[blur_strength] = BlurAugmentorConfig(
        blur_combinations=[BlurConfig],
        downscale_factor=[downscale_factor[blur_strength]],
    )


# Convert image into float32 type.
def to32F(img):
    if img.dtype == np.float32:
        return img
    return (1.0 / 255.0) * np.float32(img)


# Convert image into uint8 type.
def to8U(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(np.uint8(255.0 * img), 0, 255)


# Return if the input image is gray or not.
def _isGray(img):
    return len(img.shape) == 2


# Return down sampled image.
#  @param scale (w/s, h/s) image will be created.
#  @param shape I.shape[:2]=(h, w). numpy friendly size parameter.
def _downSample(img, scale=4, shape=None):
    if shape is not None:
        h, w = shape
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    h, w = img.shape[:2]
    return cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_NEAREST)


# Return up sampled image.
#  @param scale (w*s, h*s) image will be created.
#  @param shape I.shape[:2]=(h, w). numpy friendly size parameter.
def _upSample(img, scale=2, shape=None):
    if shape is not None:
        h, w = shape
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


# Fast guide filter.
class FastGuidedFilter:
    # Constructor.
    #  @param I Input guidance image. Color or gray.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    #  @param scale Down sampled scale.
    def __init__(self, img, radius=5, epsilon=0.4, scale=4):
        I_32F = to32F(img)
        self._I = I_32F

        I_sub = _downSample(I_32F, scale)

        self._I_sub = I_sub
        radius = int(radius / scale)

        if _isGray(img):
            self._guided_filter = GuidedFilterGray(I_sub, radius, epsilon)
        else:
            self._guided_filter = GuidedFilterColor(I_sub, radius, epsilon)

    # Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p_32F = to32F(p)
        shape_original = p.shape[:2]

        p_sub = _downSample(p_32F, shape=self._I_sub.shape[:2])

        if _isGray(p_sub):
            return self._filterGray(p_sub, shape_original)

        cs = p.shape[2]
        q = np.array(p_32F)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p_sub[:, :, ci], shape_original)
        return to8U(q)

    def _filterGray(self, p_sub, shape_original):
        ab_sub = self._guided_filter._computeCoefficients(p_sub)
        ab = [_upSample(abi, shape=shape_original) for abi in ab_sub]
        return self._guided_filter._computeOutput(ab, self._I)


# Common parts of guided filter.
#
#  This class is used by guided_filter class. GuidedFilterGray and GuidedFilterColor.
#  Based on guided_filter._computeCoefficients, guided_filter._computeOutput,
#  GuidedFilterCommon.filter computes filtered image for color and gray.
class GuidedFilterCommon:
    def __init__(self, guided_filter):
        self._guided_filter = guided_filter

    # Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p_32F = to32F(p)
        if _isGray(p_32F):
            return self._filterGray(p_32F)

        cs = p.shape[2]
        q = np.array(p_32F)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p_32F[:, :, ci])
        return q

    def _filterGray(self, p):
        ab = self._guided_filter._computeCoefficients(p)
        return self._guided_filter._computeOutput(ab, self._guided_filter._I)


# Guided filter for gray guidance image.
class GuidedFilterGray:
    #  @param I Input gray guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, img, radius=5, epsilon=0.4):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = to32F(img)
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    # Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        r = self._radius
        self._I_mean = cv2.blur(self._I, (r, r))
        I_mean_sq = cv2.blur(self._I**2, (r, r))
        self._I_var = I_mean_sq - self._I_mean**2

    def _computeCoefficients(self, p):
        r = self._radius
        p_mean = cv2.blur(p, (r, r))
        p_cov = p_mean - self._I_mean * p_mean
        a = p_cov / (self._I_var + self._epsilon)
        b = p_mean - a * self._I_mean
        a_mean = cv2.blur(a, (r, r))
        b_mean = cv2.blur(b, (r, r))
        return a_mean, b_mean

    def _computeOutput(self, ab, img):
        a_mean, b_mean = ab
        return a_mean * img + b_mean


# Guided filter for color guidance image.
class GuidedFilterColor:
    #  @param I Input color guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, img, radius=5, epsilon=0.2):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = to32F(img)
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    # Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = self._I[:, :, 0], self._I[:, :, 1], self._I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir**2, (r, r)) - self._Ir_mean**2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def _computeCoefficients(self, p):
        r = self._radius
        Ir, Ig, Ib = self._I[:, :, 0], self._I[:, :, 1], self._I[:, :, 2]

        p_mean = cv2.blur(p, (r, r))

        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov
        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar_mean = cv2.blur(ar, (r, r))
        ag_mean = cv2.blur(ag, (r, r))
        ab_mean = cv2.blur(ab, (r, r))
        b_mean = cv2.blur(b, (r, r))

        return ar_mean, ag_mean, ab_mean, b_mean

    def _computeOutput(self, ab, img):
        ar_mean, ag_mean, ab_mean, b_mean = ab

        Ir, Ig, Ib = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean

        return q


def apply_gaussian_blur(frames: np.ndarray, ksize: int = 5, sigmaX: float = 1.0) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = [cv2.GaussianBlur(_image_np, (ksize, ksize), sigmaX=sigmaX) for _image_np in frames.transpose((1, 2, 3, 0))]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class GaussianBlur:
    def __init__(self, config: GaussianBlurConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
            sigmaX = np.random.uniform(self.config.sigmaX_min, self.config.sigmaX_max)
        else:
            ksize = self.config.ksize
            sigmaX = self.config.sigmaX
        return apply_gaussian_blur(frames, ksize, sigmaX)


def apply_guided_filter(frames: np.ndarray, radius: int, eps: float, scale: float) -> np.ndarray:
    blurred_image = [FastGuidedFilter(_image_np, radius, eps, scale).filter(_image_np) for _image_np in frames.transpose((1, 2, 3, 0))]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class GuidedFilter:
    def __init__(self, config: GuidedFilterConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            radius = np.random.randint(self.config.radius_min, self.config.radius_max + 1)
            eps = np.random.uniform(self.config.eps_min, self.config.eps_max)
            scale = np.random.randint(self.config.scale_min, self.config.scale_max + 1)
        else:
            radius = self.config.radius
            eps = self.config.eps
            scale = self.config.scale
        return apply_guided_filter(frames, radius, eps, scale)


def apply_bilateral_filter(
    frames: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    iter: int = 1,
) -> np.ndarray:
    blurred_image = []
    for _image_np in frames.transpose((1, 2, 3, 0)):
        for _ in range(iter):
            _image_np = cv2.bilateralFilter(_image_np, d, sigma_color, sigma_space)
        blurred_image += [_image_np]

    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class BilateralFilter:
    def __init__(self, config: BilateralFilterConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        config = self.config
        if self.use_random:
            d = np.random.randint(config.d_min, config.d_max)
            sigma_color = np.random.randint(config.sigma_color_min, config.sigma_color_max)
            sigma_space = np.random.randint(config.sigma_space_min, config.sigma_space_max)
            iter = np.random.randint(config.iter_min, config.iter_max)
        else:
            d = config.d
            sigma_color = config.sigma_color
            sigma_space = config.sigma_space
            iter = config.iter
        return apply_bilateral_filter(frames, d, sigma_color, sigma_space, iter)


def apply_median_blur(frames: np.ndarray, ksize=5) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = [cv2.medianBlur(_image_np, ksize) for _image_np in frames.transpose((1, 2, 3, 0))]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class MedianBlur:
    def __init__(self, config: MedianBlurConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
        else:
            ksize = self.config.ksize
        return apply_median_blur(frames, ksize)


def apply_laplacian_of_gaussian(frames: np.ndarray, ksize: int = 5, sigma: float = 1.4, binarize: bool = False, threshold: float = 0.0) -> np.ndarray:
    """Apply Laplacian of Gaussian edge detection to a set of frames.

    Args:
    frames (np.ndarray): Input frames with shape (C, T, H, W)
    ksize (int): Size of the Gaussian kernel. Must be odd and positive.
    sigma (float): Standard deviation of the Gaussian distribution.
    binarize (bool): Whether to binarize the output edge map.
    threshold (float): Threshold for binarization (if binarize is True).

    Returns:
    np.ndarray: Edge-detected frames with shape (C, T, H, W).
    """
    # Ensure ksize is odd
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd

    edge_frames = []
    for frame in frames.transpose((1, 2, 3, 0)):  # (T, H, W, C)
        # Convert to grayscale if the image is in color
        if frame.shape[-1] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.squeeze()

        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if binarize:
            _, edge_map = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
        else:
            edge_map = normalized

        # Expand dimensions to match input shape
        edge_map = np.repeat(edge_map[..., np.newaxis], frames.shape[0], axis=-1)
        edge_frames.append(edge_map)
    return np.stack(edge_frames).transpose((3, 0, 1, 2))  # (C, T, H, W)


class LaplacianOfGaussian:
    """Applies Laplacian of Gaussian edge detection to images or video
    frames."""

    def __init__(self, config: LaplacianOfGaussianConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Apply LoG to input frames."""
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
            sigma = np.random.uniform(self.config.sigma_min, self.config.sigma_max)
            binarize = np.random.choice([True, False]) if not self.config.binarize else self.config.binarize
            threshold = np.random.uniform(self.config.threshold_min, self.config.threshold_max) if binarize else 0
        else:
            ksize = self.config.ksize
            sigma = self.config.sigma
            binarize = self.config.binarize
            threshold = self.config.threshold
        return apply_laplacian_of_gaussian(frames, ksize, sigma, binarize, threshold)


def apply_anisotropic_diffusion(frames: np.ndarray, alpha: float, K: float, niters: int) -> np.ndarray:
    """Apply Anisotropic Diffusion to a set of frames.

    Args:
    frames (np.ndarray): Input frames with shape (C, T, H, W)
    alpha (float): The amount of time to step forward on each iteration (between 0 and 1)
    K (float): Sensitivity to edges
    niters (int): Number of iterations

    Returns:
    np.ndarray: Anisotropic-diffused frames with shape (C, T, H, W).
    """
    blurred_image = [cv2.ximgproc.anisotropicDiffusion(_image_np, alpha, K, niters) for _image_np in frames.transpose((1, 2, 3, 0))]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))

    return blurred_image


class AnisotropicDiffusion:
    """Applies Anisotropic Diffusion to images or video frames."""

    def __init__(self, config: AnisotropicDiffusionConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            alpha = np.random.uniform(self.config.alpha_min, self.config.alpha_max)
            K = np.random.uniform(self.config.K_min, self.config.K_max)
            niters = np.random.randint(self.config.niters_min, self.config.niters_max + 1)
        else:
            alpha = self.config.alpha
            K = self.config.K
            niters = self.config.niters
        return apply_anisotropic_diffusion(frames, alpha, K, niters)
