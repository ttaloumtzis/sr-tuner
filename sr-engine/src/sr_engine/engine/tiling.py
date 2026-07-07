"""Tiled inference utilities for VRAM-safe large-image processing."""

import torch


def _compute_starts(dim_size: int, tile_size: int, stride: int) -> list[int]:
    """Compute tile start offsets along one dimension.

    Walks forward in steps of *stride*, then clamps the final tile so it
    ends exactly at *dim_size* (rather than leaving a small ragged tile
    hanging off the edge). Every tile has exactly *tile_size* extent.
    """
    if dim_size <= tile_size:
        return [0]

    starts = list(range(0, dim_size - tile_size + 1, stride))
    last_valid_start = dim_size - tile_size
    if starts[-1] != last_valid_start:
        starts.append(last_valid_start)
    return starts


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
        tile's top-left pixel position in the original image.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")

    _, h, w = image.shape

    # Nothing to tile — the whole image already fits in one tile.
    if h <= tile_size and w <= tile_size:
        return [(image, (0, 0))]

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than tile_size ({tile_size})"
        )

    row_starts = _compute_starts(h, tile_size, stride)
    col_starts = _compute_starts(w, tile_size, stride)

    tiles: list[tuple[torch.Tensor, tuple[int, int]]] = []
    for row in row_starts:
        for col in col_starts:
            tile = image[:, row: row + tile_size, col: col + tile_size]
            tiles.append((tile, (row, col)))

    return tiles


def stitch_tiles(
    tiles: list[tuple[torch.Tensor, tuple[int, int]]],
    output_size: tuple[int, int],
    overlap: int,
) -> torch.Tensor:
    """Stitch overlapping tiles back into a full image.

    Overlapping regions are averaged (by accumulation count) to avoid seam
    artifacts.

    Args:
        tiles: List of ``(tile, (row, col))`` tuples.
        output_size: ``(H, W)`` of the output image.
        overlap: Overlap used during tiling (accepted for API symmetry with
            ``tile_image``; the current averaging strategy doesn't need the
            exact value, since it just counts contributions per pixel).

    Returns:
        Stitched image tensor of shape ``(C, H, W)``.
    """
    if not tiles:
        raise ValueError("Cannot stitch an empty list of tiles.")

    height, width = output_size
    first_tile, _ = tiles[0]
    channels = first_tile.shape[0]
    device = first_tile.device
    dtype = first_tile.dtype

    output = torch.zeros((channels, height, width), device=device, dtype=dtype)
    weight = torch.zeros((1, height, width), device=device, dtype=dtype)

    for tile, (row, col) in tiles:
        tile_h, tile_w = tile.shape[1], tile.shape[2]

        # Clip in case a tile would extend past the declared output bounds
        # (shouldn't normally happen if tiles came from tile_image with a
        # matching output_size, but guards against mismatched inputs).
        row_end = min(row + tile_h, height)
        col_end = min(col + tile_w, width)

        output[:, row:row_end, col:col_end] += tile[:, : row_end - row, : col_end - col]
        weight[:, row:row_end, col:col_end] += 1.0

    weight = weight.clamp(min=1e-8)
    return output / weight