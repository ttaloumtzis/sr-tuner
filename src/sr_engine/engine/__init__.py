"""Engine — training, inference, tiling, and metrics."""

from .tiling import stitch_tiles, tile_image
from .trainer import Trainer, TrainingCancelled
from .metrics import lpips, psnr, ssim
from .inference import infer_image, infer_video
