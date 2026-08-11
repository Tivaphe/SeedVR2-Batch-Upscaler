"""Interface abstraite des backends d'upscaling + logique commune.

La classe ``UpscaleBackend`` gère de façon générique :
    - le canal alpha (upscalé séparément en lanczos) ;
    - le découpage en tuiles (recouvrement + fusion douce) ;
    - les arrondis de dimensions (multiple de 16, imposé par le VAE officiel).

Chaque backend concret n'a qu'à implémenter :
    - ``load(model)``         — chargement du modèle, **une seule fois** ;
    - ``_upscale_region(...)``— l'inférence d'une tuile ou de l'image entière ;
    - ``unload()``            — libération de la VRAM.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Callable

from PIL.Image import Image as PILImage

from ..constants import DIVISOR, TILING_AUTO_THRESHOLD_MPX
from ..images import merge_alpha, split_alpha
from ..models import ModelInfo, Precision, TilingConfig
from ..tiling import compute_tiles, needs_tiling, stitch

ProgressCb = Callable[[str], None]

# Régime d'entraînement de SeedVR2 : ~720x1280 ≈ 1 Mpx. Au-delà, l'attention
# explose en coût (coût ∝ tokens²) ET la qualité du modèle se dégrade — on
# bornage donc la dimension des tuiles de SORTIE à cette échelle par défaut.
MAX_TILE_OUT_SIDE = 1152


def ceil_div(value: float, divisor: int = DIVISOR) -> int:
    """Arrondit au multiple de ``divisor`` supérieur (contrainte DivisibleCrop)."""
    return max(divisor, int(math.ceil(value / divisor)) * divisor)


@dataclass(frozen=True)
class BackendOptions:
    """Options transverses passées au chargement du backend."""

    precision: Precision = Precision.AUTO
    low_vram: bool = True        # offload du DiT vers le CPU entre deux passes
    clear_cache: bool = True     # torch.cuda.empty_cache() entre les images
    tiling: TilingConfig = TilingConfig()


class BackendUnavailable(RuntimeError):
    """Le backend demandé n'est pas utilisable (dépendance manquante, etc.)."""


class UpscaleBackend(abc.ABC):
    """Contrat minimal d'un backend d'upscaling d'images."""

    name: str = "base"

    def __init__(self, options: BackendOptions | None = None) -> None:
        self.options = options or BackendOptions()
        self._model: ModelInfo | None = None

    # ------------------------------------------------------------------ #
    # Cycle de vie                                                         #
    # ------------------------------------------------------------------ #
    @property
    def loaded_model(self) -> ModelInfo | None:
        return self._model

    def is_loaded_for(self, model: ModelInfo) -> bool:
        """Vrai si le modèle demandé est déjà chargé (pas de rechargement)."""
        return self._model is not None and self._model.path == model.path

    @abc.abstractmethod
    def load(self, model: ModelInfo, log: ProgressCb = print) -> None:
        """Charge le modèle en VRAM. Appelé une seule fois par lot."""

    def unload(self) -> None:
        """Libère les ressources (surchargeable, no-op par défaut)."""
        self._model = None

    # ------------------------------------------------------------------ #
    # Upscaling                                                            #
    # ------------------------------------------------------------------ #
    def upscale(
        self,
        image: PILImage,
        scale: float,
        seed: int,
        log: ProgressCb = print,
        should_abort: Callable[[], bool] = lambda: False,
        tile_cb: "Callable[[int, int], None] | None" = None,
    ) -> PILImage:
        """Upscale une image PIL complète (tuiles + alpha gérés ici).

        Args:
            image: image source (RGB ou RGBA).
            scale: facteur d'agrandissement.
            seed: graine aléatoire (par image, pour la reproductibilité).
            log: journalisation.
            should_abort: callback d'annulation immédiate (entre les tuiles).
            tile_cb: notification « tuile i/N » (UI) ; reçoit ``(0, 0)`` en fin
                d'image pour réinitialiser l'affichage.
        """
        if self._model is None:
            raise RuntimeError("Aucun modèle chargé : appeler load() d'abord.")

        rgb, alpha = split_alpha(image)
        w, h = rgb.size
        out_w, out_h = ceil_div(w * scale), ceil_div(h * scale)
        tiling = self.options.tiling

        use_tiles = tiling.enabled or needs_tiling(out_w, out_h, TILING_AUTO_THRESHOLD_MPX)

        # Bornage de la tuile côté SORTIE : une tuile upscalée doit rester au
        # régime (≈1 Mpx) pour lequel le modèle est entraîné — sinon lenteur
        # quadratique en attention et qualité dégradée.
        max_in_side = max(64, min(tiling.tile_size, int(MAX_TILE_OUT_SIDE / scale)))
        if max_in_side < tiling.tile_size:
            log(f"Tuile réduite {tiling.tile_size}→{max_in_side} px d'entrée "
                f"(sortie/tuile ≤ ~{max_in_side * scale:.0f} px = régime du modèle).")
        overlap = max(0, min(tiling.overlap, max_in_side // 2))

        if use_tiles and (w > max_in_side or h > max_in_side):
            tiles = compute_tiles(w, h, max_in_side, overlap)
            upscaled_tiles: list[PILImage] = []
            for index, tile in enumerate(tiles, 1):
                if should_abort():
                    raise InterruptedError("Traitement annulé pendant le tuilage.")
                if tile_cb is not None:
                    tile_cb(index, len(tiles))
                log(f"  tuile {index}/{len(tiles)} ({tile.width}×{tile.height})")
                region = rgb.crop(tile.crop_box())
                tw = ceil_div(tile.width * scale)
                th = ceil_div(tile.height * scale)
                eff_seed = seed + index  # déterministe, différent par tuile
                upscaled_tiles.append(
                    self._upscale_region(region, tw, th, eff_seed, log)
                )
            result = stitch(tiles, upscaled_tiles, w, h, scale, overlap)
            if tile_cb is not None:
                tile_cb(0, 0)  # fin d'image : réinitialise le compteur de l'UI
        else:
            result = self._upscale_region(rgb, out_w, out_h, seed, log)

        # Le canal alpha n'est pas généré par le modèle : upscale bilinéaire séparé
        # (le bilinéaire évite les halos de Gibbs sur les masques).
        if alpha is not None:
            alpha = alpha.resize(result.size, resample=2)  # PIL.Image.BILINEAR
            result = merge_alpha(result, alpha)
        return result

    @abc.abstractmethod
    def _upscale_region(
        self,
        region: PILImage,
        out_w: int,
        out_h: int,
        seed: int,
        log: ProgressCb,
    ) -> PILImage:
        """Inférence effective sur une région RGB.

        ``region`` est déjà à la taille d'entrée choisie ; ``out_w``/``out_h``
        sont les dimensions cibles (multiples de 16). L'implémentation doit
        adresser le modèle en tenant compte de ses contraintes internes.
        """
