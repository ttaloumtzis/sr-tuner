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


def _tile_blend_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    has_top: bool,
    has_bot: bool,
    has_left: bool,
    has_right: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a 2D linear-ramp weight map for one tile.

    Outside the overlap region the weight is 1.0.  In each overlap region
    the weight linearly ramps from 1.0 at the inner edge to 0.0 at the
    tile edge (when a neighbour exists).  Tile edges that form the image
    border (no neighbour) keep weight 1.0 so border pixels are preserved.

    Returns a ``(tile_h, tile_w)`` weight tensor.
    """
    row_w = torch.ones(tile_h, device=device, dtype=dtype)
    col_w = torch.ones(tile_w, device=device, dtype=dtype)

    if overlap > 0:
        if has_top:
            row_w[:overlap] = torch.linspace(0, 1, overlap, device=device, dtype=dtype)
        if has_bot:
            row_w[tile_h - overlap:] = torch.linspace(1, 0, overlap, device=device, dtype=dtype)
        if has_left:
            col_w[:overlap] = torch.linspace(0, 1, overlap, device=device, dtype=dtype)
        if has_right:
            col_w[tile_w - overlap:] = torch.linspace(1, 0, overlap, device=device, dtype=dtype)

    return row_w[:, None] * col_w[None, :]


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
    overlap: int = 0,
) -> torch.Tensor:
    """Stitch overlapping tiles back into a full image.

    Overlapping regions are blended with a position-aware linear ramp so
    that every pixel receives weight 1.0 in total — seam artefacts are
    eliminated while border pixels are preserved exactly.

    Args:
        tiles: List of ``(tile, (row, col))`` tuples.
        output_size: ``(H, W)`` of the output image.
        overlap: Overlap between adjacent tiles in pixels (used to
            determine the blend-ramp width).

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

    tile_h, tile_w = first_tile.shape[1], first_tile.shape[2]

    output = torch.zeros((channels, height, width), device=device, dtype=dtype)
    weight = torch.zeros((1, height, width), device=device, dtype=dtype)

    for tile, (row, col) in tiles:
        row_end = min(row + tile_h, height)
        col_end = min(col + tile_w, width)
        actual_h = row_end - row
        actual_w = col_end - col

        # Detect image borders — tiles at the image edge have no neighbor
        # on that side, so their overlap weight stays 1.0 (no blending).
        w = _tile_blend_weight(
            tile_h, tile_w, overlap,
            has_top=row > 0, has_bot=row_end < height,
            has_left=col > 0, has_right=col_end < width,
            device=device, dtype=dtype,
        )[:actual_h, :actual_w]

        output[:, row:row_end, col:col_end] += tile[:, :actual_h, :actual_w] * w
        weight[:, row:row_end, col:col_end] += w

    weight = weight.clamp(min=1e-8)
    return output / weight