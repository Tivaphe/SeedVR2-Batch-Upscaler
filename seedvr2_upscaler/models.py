"""Dataclasses et énumérations partagées par l'ensemble de l'application.

Tout est typé et immuable lorsque c'est possible afin de sécuriser
le passage des paramètres entre l'UI, le moteur de lot et les backends.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class ModelKind(enum.Enum):
    """Famille de poids détectée à partir du nom/extension du fichier."""

    OFFICIAL = "Officiel (.pth)"      # checkpoint torch officiel ByteDance
    FP16 = "FP16"                     # safetensors fp16/bf16
    FP8 = "FP8 (e4m3)"                # safetensors fp8 e4m3fn
    GGUF = "GGUF"                     # version quantifiée GGUF


class ConflictPolicy(enum.Enum):
    """Que faire si le fichier de sortie existe déjà ?"""

    OVERWRITE = "overwrite"   # Écraser
    SKIP = "skip"             # Ignorer l'image
    RENAME = "rename"         # Renommer : "photo (1).png"

    @classmethod
    def from_label(cls, label: str) -> "ConflictPolicy":
        return {
            # libellés FR (CLI historique) + EN + valeurs canoniques de l'UI
            "écraser": cls.OVERWRITE, "overwrite": cls.OVERWRITE,
            "ignorer": cls.SKIP, "skip": cls.SKIP,
            "renommer (auto)": cls.RENAME, "auto-rename": cls.RENAME,
            "rename": cls.RENAME,
        }.get(label.strip().lower(), cls.RENAME)


class OutputFormat(enum.Enum):
    KEEP = "keep"     # comme l'entrée : extension et format strictement conservés
    PNG = "png"
    JPEG = "jpg"
    WEBP = "webp"

    @classmethod
    def from_label(cls, label: str) -> "OutputFormat":
        label = label.strip().lower()
        if label in {"c", "k", "keep", "comme l'entrée", "identique",
                     "comme l’entree", "same as input"}:
            return cls.KEEP
        if "entr" in label or "input" in label:  # « Comme l'entrée » / « Same as input »
            return cls.KEEP
        if label in {"jpg", "jpeg"}:
            return cls.JPEG
        if label == "webp":
            return cls.WEBP
        return cls.PNG


def format_for_source(source: Path, requested: OutputFormat) -> tuple[OutputFormat, str]:
    """Résout (format effectif, extension) pour un fichier source donné.

    En mode KEEP : PNG~→.png, JPG/JPEG~→.jpg, WEBP~→.webp ; les entrées
    BMP/TIFF (non réinscriptibles) sont exportées en PNG.
    """
    if requested is not OutputFormat.KEEP:
        ext = {OutputFormat.PNG: ".png", OutputFormat.JPEG: ".jpg",
               OutputFormat.WEBP: ".webp"}[requested]
        return requested, ext
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return OutputFormat.JPEG, ".jpg"
    if suffix == ".webp":
        return OutputFormat.WEBP, ".webp"
    return OutputFormat.PNG, ".png"


class Precision(enum.Enum):
    AUTO = "auto"     # bf16 si le GPU le supporte, sinon fp16
    BF16 = "bf16"
    FP16 = "fp16"


class RunnerState(enum.Enum):
    """État du moteur de traitement (pour l'UI et les verrous)."""

    IDLE = "idle"
    LOADING = "loading"     # chargement du modèle
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelInfo:
    """Description d'un fichier de poids détecté dans models/."""

    path: Path
    name: str                       # nom affiché dans la liste déroulante
    kind: ModelKind
    quant: str | None = None        # ex. "Q4_K_M" pour les GGUF
    size_bytes: int = 0

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def label(self) -> str:
        """Étiquette lisible : ``seedvr2_ema_3b-Q4_K_M.gguf  · GGUF Q4_K_M · 1,9 Go``."""
        extra = f" {self.quant}" if self.quant else ""
        return f"{self.path.name}  · {self.kind.value}{extra} · {self.size_gib:.1f} Go"


@dataclass(frozen=True)
class TilingConfig:
    """Paramètres du découpage en tuiles (GPU à faible VRAM)."""

    enabled: bool = False           # forcer l'activation (sinon : auto)
    tile_size: int = 512            # px, côté entrée
    overlap: int = 64               # px, côté entrée


@dataclass(frozen=True)
class JobConfig:
    """Configuration complète et immuable d'un lot d'upscaling."""

    input_dir: Path
    output_dir: Path
    model: ModelInfo
    scale: float = 4.0
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95                       # 1-100 (JPEG/WebP, compression PNG)
    conflict_policy: ConflictPolicy = ConflictPolicy.RENAME
    add_suffix: bool = False                # option explicite (décochée = nom inchangé)
    suffix: str = "_upscaled"               # utilisé seulement si add_suffix est vrai
    seed: int = 666
    precision: Precision = Precision.AUTO
    low_vram: bool = True                   # offload interne du DiT (dit_offload)
    clear_cache: bool = True                # vider le cache GPU entre les images
    parallel_io: bool = True                # écritures disque sur un thread séparé
    resume: bool = True                     # reprendre un lot interrompu
    tiling: TilingConfig = field(default_factory=TilingConfig)
    backend_name: str = "auto"              # "auto" | "official"


@dataclass
class ImageOutcome:
    """Résultat du traitement d'une image."""

    source: Path
    destination: Path | None
    ok: bool
    skipped: bool = False
    error: str | None = None
    seconds: float = 0.0


@dataclass
class BatchStats:
    """Statistiques agrégées en fin de lot (pour le rapport final)."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    avg_seconds_per_image: float = 0.0
    errors: list[str] = field(default_factory=list)
