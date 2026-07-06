"""Tiled inference utilities for VRAM-safe large-image processing."""

import torch


def tile_image(
    image: torch.Tensor,
    tile_size: int,
    overlap: int,
) -> list[tuple[torch.Tensor, tuple[int, int]]]:
    """Split an image tensor into overlapping tiles.

    Args:
        image: Input tensor of shape ``(C, H, W)``.
        tile_size: Tile width/height in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        List of ``(tile, (row, col))`` tuples where ``row, col`` is the
        tile's top-left position in the original image grid.
    """
    raise NotImplementedError("TODO: implement image tiling")


def stitch_tiles(
    tiles: list[tuple[torch.Tensor, tuple[int, int]]],
    output_size: tuple[int, int],
    overlap: int,
) -> torch.Tensor:
    """Stitch overlapping tiles back into a full image.

    Overlapping regions are averaged to avoid seam artifacts.

    Args:
        tiles: List of ``(tile, (row, col))`` tuples.
        output_size: ``(H, W)`` of the output image.
        overlap: Overlap used during tiling.

    Returns:
        Stitched image tensor of shape ``(C, H, W)``.
    """
    raise NotImplementedError("TODO: implement tile stitching")
