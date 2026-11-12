"""Lecture / écriture des images : formats, EXIF, ICC, canal alpha.

Points clés :
    - les métadonnées EXIF (et le profil ICC) sont préservées quand le
      format de sortie le permet (JPEG/WebP/PNG nativement via Pillow) ;
    - le canal alpha est traité séparément : le modèle de diffusion ne
      produit que du RGB, l'alpha est donc upscalé en bilinéaire puis
      réassemblé ;
    - la rotation EXIF est appliquée à la lecture pour éviter les images
      couchées en sortie.
"""
from __future__ import annotations

import os
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps
from PIL.Image import Image as PILImage

from .models import OutputFormat

@dataclass
class LoadedImage:
    """Image d'entrée avec ses métadonnées à réinjecter à l'écriture."""

    image: PILImage          # RVB ou RVBA, rotation EXIF appliquée
    path: Path
    exif_bytes: bytes | None # blob EXIF brut (tel que lu par PIL)
    icc_profile: bytes | None
    source_format: str       # « PNG », « JPEG »… (informatif, pour les logs)


def load_image(path: Path) -> LoadedImage:
    """Charge une image, applique l'orientation EXIF et normalise le mode.

    Le mode est ramené à RGB ou RGBA (P, LA, CMYK, 16 bits… sont convertis).
    Les TIFF multi-pages : seule la première page est utilisée.
    """
    # Image.open est paresseux : .load() explicite pour capter les erreurs ici.
    with Image.open(path) as raw:
        raw.load()
        icc_profile = raw.info.get("icc_profile")
        source_format = raw.format or "?"
        image = ImageOps.exif_transpose(raw)  # applique le tag Orientation
        exif = image.getexif()
        exif_bytes = exif.tobytes() if exif else None
        if image.mode == "CMYK" and icc_profile:
            try:
                srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
                image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
                                                  srgb, outputMode="RGB")
                icc_profile = srgb.tobytes()
            except (ImageCms.PyCmsError, OSError, ValueError):
                # Never label converted RGB pixels with a CMYK profile.
                icc_profile = None
        image = _normalize_mode(image)
    return LoadedImage(
        image=image,
        path=path,
        exif_bytes=exif_bytes,
        icc_profile=icc_profile,
        source_format=source_format,
    )


def _normalize_mode(image: PILImage) -> PILImage:
    """Normalise vers RGB / RGBA en préservant l'alpha s'il est réellement utilisé."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        converted = image.convert("RGBA")
        # Si l'alpha est 255 partout, autant travailler en RGB (plus léger).
        if converted.getchannel("A").getextrema() == (255, 255):
            return converted.convert("RGB")
        return converted
    return image.convert("RGB")


def split_alpha(image: PILImage) -> tuple[PILImage, PILImage | None]:
    """Sépare le RGB du canal alpha (None si l'image est opaque)."""
    if image.mode == "RGBA":
        return image.convert("RGB"), image.getchannel("A")
    return image, None


def merge_alpha(rgb: PILImage, alpha: PILImage | None) -> PILImage:
    """Réassemble RGB + alpha (l'alpha doit déjà être à la bonne taille)."""
    if alpha is None:
        return rgb
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def save_image(
    image: PILImage,
    path: Path,
    fmt: OutputFormat,
    quality: int,
    exif_bytes: bytes | None = None,
    icc_profile: bytes | None = None,
) -> None:
    """Sauvegarde l'image dans le format demandé avec ses métadonnées.

    Args:
        quality: 1-100. Qualité JPEG/WebP ; pour le PNG elle pilote le
            niveau de compression (qualité visuelle identique, taille vitesse).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = int(min(100, max(1, quality)))
    kwargs: dict[str, object] = {}

    if fmt is OutputFormat.JPEG:
        to_save = image.convert("RGB") if image.mode not in ("RGB", "L") else image
        kwargs.update(format="JPEG", quality=quality, optimize=True, progressive=True)
        if exif_bytes:
            kwargs["exif"] = exif_bytes  # JPEG : EXIF natif
        if icc_profile:
            kwargs["icc_profile"] = icc_profile

    elif fmt is OutputFormat.WEBP:
        to_save = image
        kwargs.update(format="WEBP", quality=quality, method=6, exact=True)
        if exif_bytes:
            kwargs["exif"] = exif_bytes  # WebP : EXIF supporté par PIL
        if icc_profile:
            kwargs["icc_profile"] = icc_profile

    else:  # PNG : sans perte, la « qualité » règle seulement l'effort de compression
        to_save = image
        # qualité 100 -> niveau 3 (rapide) … qualité 1 -> niveau 9 (dense)
        kwargs.update(format="PNG", compress_level=max(1, 9 - quality // 15))

    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    if exif_bytes:
        # Pillow supports PNG eXIf natively. Orientation was removed on load.
        exif = Image.Exif()
        exif.load(exif_bytes)
        exif.pop(274, None)
        for tag, value in ((40962, image.width), (40963, image.height)):
            if tag in exif:
                exif[tag] = value
        kwargs["exif"] = exif.tobytes()
    fd, temporary = tempfile.mkstemp(prefix=".seedvr2-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            to_save.save(handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def has_real_error(path: Path) -> str | None:
    """Vérifie rapidement qu'un fichier est une image lisible (pré-scan)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return None
    except ImageCms.PyCmsError as exc:  # profil ICC cassé : non bloquant ailleurs
        return str(exc)
    except Exception as exc:
        return str(exc)
