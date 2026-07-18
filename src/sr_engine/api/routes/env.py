import torch

from fastapi import APIRouter

from sr_engine.api.schemas import EnvInfo
from sr_engine.device.backend import (
    get_device,
    is_rocm,
    supports_flash_attn,
)

router = APIRouter(prefix="/api/env", tags=["environment"])


@router.get("", response_model=EnvInfo)
async def env_check():
    device = get_device()
    is_cuda = torch.cuda.is_available()
    dev_name = None
    vram_mb = None
    if is_cuda:
        dev_idx = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        vram_mb = torch.cuda.get_device_properties(dev_idx).total_memory // 1024**2

    return EnvInfo(
        torch_version=torch.__version__,
        device=str(device),
        cuda_available=is_cuda,
        rocm=is_rocm(),
        bf16_supported=torch.cuda.is_bf16_supported() if is_cuda else False,
        flash_attn=supports_flash_attn(),
        device_name=dev_name,
        vram_total_mb=vram_mb,
    )