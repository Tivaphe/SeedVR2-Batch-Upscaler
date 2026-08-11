"""Lecture / écriture des images : formats, EXIF, ICC, canal alpha.

Points clés :
    - les métadonnées EXIF (et le profil ICC) sont préservées quand le
      format de sortie le permet (JPEG/WebP nativement ; PNG via piexif
      s'il est installé) ;
    - le canal alpha est traité séparément : le modèle de diffusion ne
      produit que du RGB, l'alpha est donc upscalé en bicubique puis
      réassemblé ;
    - la rotation EXIF est appliquée à la lecture pour éviter les images
      couchées en sortie.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps
from PIL.Image import Image as PILImage

from .models import OutputFormat

try:  # piexif : écriture EXIF dans les PNG (optionnel, non bloquant)
    import piexif  # type: ignore[import-untyped]

    _HAS_PIEXIF = True
except Exception:  # pragma: no cover - dépend de l'installation
    _HAS_PIEXIF = False


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
        exif_bytes = raw.info.get("exif")
        icc_profile = raw.info.get("icc_profile")
        source_format = raw.format or "?"
        image = ImageOps.exif_transpose(raw)  # applique le tag Orientation
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
    if image.mode in ("RGBA", "LA", "PA"):
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

    if fmt is OutputFormat.PNG and exif_bytes and _HAS_PIEXIF:
        # PNG n'a pas de slot EXIF standard pour PIL : injection via piexif.
        try:
            exif_dict = piexif.load(exif_bytes)
            kwargs["pnginfo"] = _pnginfo_with_exif(exif_dict)
        except Exception:
            pass  # on ne bloque jamais un lot pour des métadonnées

    to_save.save(path, **kwargs)


def _pnginfo_with_exif(exif_dict: dict) -> object:
    """Construit un PngInfo contenant la chunk eXIf (via piexif)."""
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add(b"eXIf", piexif.dump(exif_dict), zip=False)
    return info


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
