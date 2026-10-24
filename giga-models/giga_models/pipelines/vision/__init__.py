from .depth_estimation import DepthAnythingForDepthEstimationPipeline, DPTForDepthEstimation2Pipeline, DPTForDepthEstimationPipeline
from .edge_detection import CannyPipeline, HEDPipeline, LineartPipeline, MLSDPipeline, PidiNetPipeline
from .face_swap import InsightFaceSwapPipeline
from .frame_interpolation import FilmPipeline
from .image_restoration import PromptIRPipeline
from .keypoints import DWposePipeline, OpenPosePipeline, RTMPosePipeline
from .lane_detection import LaneAFPipeline
from .others import AestheticPipeline, BlurPipeline, ContentShufflePipeline, LaplacePipeline, NormalBaePipeline
from .scene_detect import TransNetV2Pipeline
from .segmentation import SegmentAnythingPipeline, UperNetForSemanticSegmentationPipeline
from .super_resolution import RealESRGANPipeline, RGTPipeline
