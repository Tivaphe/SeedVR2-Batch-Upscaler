"""Découpage en tuiles avec recouvrement et fusion douce (feather blending).

Utilisé pour les GPU à faible VRAM : une image trop grande est découpée en
tuiles côté *entrée*, chaque tuile est upscalée indépendamment, puis le tout
est recousu. Les zones de recouvrement sont fusionnées par une rampe
linéaire afin d'éviter toute couture visible.

Toutes les coordonnées sont exprimées en pixels de l'image d'entrée ;
la mise à l'échelle par le facteur d'upscale est appliquée au moment
de la fusion (``stitch``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class Tile:
    """Une tuile côté image d'entrée (coordonnées en px d'entrée)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop_box(self) -> tuple[int, int, int, int]:
        """Boîte au format ``PIL.Image.crop``."""
        return (self.x0, self.y0, self.x1, self.y1)


def needs_tiling(out_w: int, out_h: int, threshold_mpx: float) -> bool:
    """Vrai si la sortie dépasse le seuil (en mégapixels) au-delà duquel on tuile."""
    return (out_w * out_h) / 1_000_000.0 > threshold_mpx


def compute_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Grille de tuiles couvrant l'image, avec recouvrement constant.

    La dernière tuile est plaquée sur le bord droit/bas lorsque la dimension
    n'est pas un multiple exact du pas, afin de couvrir l'image sans trou.
    """
    tile_size = max(64, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    step = tile_size - overlap

    def axis_positions(length: int) -> list[tuple[int, int]]:
        if length <= tile_size:
            return [(0, length)]
        starts = list(range(0, length, step))
        # Dernière tuile : plaquée sur le bord si besoin (recouvrement > overlap, bénin).
        return [(s, min(s + tile_size, length)) for s in starts]

    tiles = [
        Tile(x0, y0, x1, y1)
        for y0, y1 in axis_positions(height)
        for x0, x1 in axis_positions(width)
    ]
    return tiles


def _feather_mask(
    tw: int,
    th: int,
    ramp_x: int,
    ramp_y: int,
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
) -> np.ndarray:
    """Masque de fusion : 1 au centre, rampe linéaire vers 0 sur les bords de tuile.

    ``left/right/top/bottom`` indiquent si la tuile a un voisin de ce côté
    (les bords de l'image complète gardent un poids de 1).
    """
    wx = np.ones(tw, dtype=np.float32)
    wy = np.ones(th, dtype=np.float32)
    nx = max(1, min(ramp_x, tw // 2))
    ny = max(1, min(ramp_y, th // 2))
    rx = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32) + 1.0 / nx
    ry = np.linspace(0.0, 1.0, ny, endpoint=False, dtype=np.float32) + 1.0 / ny

    if left:
        wx[:nx] = np.minimum(wx[:nx], rx)
    if right:
        wx[tw - nx :] = np.minimum(wx[tw - nx :], rx[::-1])
    if top:
        wy[:ny] = np.minimum(wy[:ny], ry)
    if bottom:
        wy[th - ny :] = np.minimum(wy[th - ny :], ry[::-1])
    return wy[:, None] * wx[None, :]


def stitch(
    tiles: list[Tile],
    upscaled: list[PILImage],
    img_w: int,
    img_h: int,
    scale: float,
    overlap: int,
) -> PILImage:
    """Recoud les tuiles upscalées en une image complète (fusion pondérée).

    Args:
        tiles: tuiles en coordonnées d'entrée (même ordre que ``upscaled``).
        upscaled: images PIL (RGB) upscalées, une par tuile.
        img_w, img_h: dimensions de l'image d'entrée.
        scale: facteur d'upscale appliqué par le modèle.
        overlap: recouvrement demandé (px d'entrée) ; fixe la longueur des rampes.
    """
    out_w, out_h = round(img_w * scale), round(img_h * scale)
    # Longueur des rampes de fusion, exprimée en pixels de sortie.
    ramp_x = ramp_y = max(1, int(round(overlap * scale)))

    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)

    for tile, up in zip(tiles, upscaled):
        ox0, oy0 = round(tile.x0 * scale), round(tile.y0 * scale)
        tw, th = round(tile.width * scale), round(tile.height * scale)
        if up.size != (tw, th):  # tolérance aux arrondis du backend
            up = up.resize((tw, th), Image.LANCZOS)
        mask = _feather_mask(
            tw, th, ramp_x, ramp_y,
            left=tile.x0 > 0,
            right=tile.x1 < img_w,
            top=tile.y0 > 0,
            bottom=tile.y1 < img_h,
        )
        canvas[oy0 : oy0 + th, ox0 : ox0 + tw] += np.asarray(up, dtype=np.float32) * mask[:, :, None]
        weights[oy0 : oy0 + th, ox0 : ox0 + tw] += mask

    weights = np.maximum(weights, 1e-6)  # sécurité : jamais de division par 0
    result = np.clip(canvas / weights[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGB")
