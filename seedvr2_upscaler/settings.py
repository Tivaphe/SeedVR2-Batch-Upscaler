"""Persistance automatique des paramètres utilisateur (JSON dans config/).

Les valeurs sont sauvegardées à chaque changement pertinent de l'UI et
rechargées au démarrage. Le fichier est tolérant : toute clé inconnue ou
manquante est ignorée (compatibilité ascendante simple).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .constants import SETTINGS_PATH


@dataclass
class Settings:
    """Valeurs par défaut de l'interface (durables entre deux sessions)."""

    input_dir: str = ""
    output_dir: str = ""
    model_choice: str = ""            # nom du fichier de modèle sélectionné
    scale_choice: str = "x4"          # "x2" | "x4" | "x8" | "custom"
    scale_custom: float = 3.0
    output_format: str = "Comme l'entrée"  # "Comme l'entrée" | "PNG" | "JPG" | "WebP"
    quality: int = 95
    add_suffix: bool = False          # décoché = nom d'entrée strictement conservé
    suffix: str = "_upscaled"
    conflict_policy: str = "Renommer (auto)"
    tiling: bool = False
    tile_size: int = 512
    tile_overlap: int = 64
    seed: int = 666
    precision: str = "auto"           # auto | bf16 | fp16
    low_vram: bool = True
    clear_cache: bool = True
    parallel_io: bool = True
    resume: bool = True
    backend: str = "auto"             # auto | official
    language: str = "fr"              # langue de l'interface : fr | en
    # --- Réglages avancés natifs du pipeline officiel ---
    cfg_scale: float = 1.0            # guidage diffusion (1.0 = officiel ; > 1 accentue)
    cfg_rescale: float = 0.0          # rescale du guidage (0.0 = valeur officielle)
    cond_noise_scale: float = 0.0     # bruit de condition latent (0.0 = valeur officielle)
    color_fix: bool = True            # wavelet color fix officiel (si dispo dans le dépôt)

    # ------------------------------------------------------------------ #
    # Les radios de l'UI travaillent en valeurs canoniques (indépendantes
    # de la langue). Ces tables convertissent les libellés français des
    # anciens settings.json (v ≤ 1.1) vers ces valeurs canoniques.
    _MIGRATIONS = {
        "scale_choice": {"Personnalisé": "custom", "Custom": "custom"},
        "output_format": {"Comme l'entrée": "keep", "Same as input": "keep",
                          "PNG": "png", "JPG": "jpg", "WebP": "webp"},
        "conflict_policy": {"Écraser": "overwrite", "Ignorer": "skip",
                            "Renommer (auto)": "rename", "Overwrite": "overwrite",
                            "Skip": "skip", "Auto-rename": "rename"},
    }
    _VALID = {
        "scale_choice": {"x2", "x4", "x8", "custom"},
        "output_format": {"keep", "png", "jpg", "webp"},
        "conflict_policy": {"overwrite", "skip", "rename"},
    }
    _DEFAULTS = {"scale_choice": "x4", "output_format": "keep",
                 "conflict_policy": "rename"}

    def __post_init__(self) -> None:
        # Un settings.json écrit par une ancienne version peut contenir
        # « demo » (backend supprimé) : on le neutralise silencieusement.
        if self.backend not in {"auto", "official"}:
            self.backend = "auto"
        if self.language not in {"fr", "en"}:
            self.language = "fr"
        # Garde-fous des réglages avancés (bornes larges mais sûres — un
        # settings.json corrompu/édité à la main ne doit jamais planter l'appli).
        self.cfg_scale = min(4.0, max(0.5, float(self.cfg_scale)))
        self.cfg_rescale = min(1.0, max(0.0, float(self.cfg_rescale)))
        self.cond_noise_scale = min(1.0, max(0.0, float(self.cond_noise_scale)))
        # Migration des anciens libellés FR + garde-fous de validité.
        for attr, table in self._MIGRATIONS.items():
            value = getattr(self, attr)
            if value in table:
                setattr(self, attr, table[value])
            if getattr(self, attr) not in self._VALID[attr]:
                setattr(self, attr, self._DEFAULTS[attr])


    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Charge les paramètres ; retourne les valeurs par défaut si absent/invalide."""
    try:
        if path.exists():
            return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return Settings()


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    """Écrit les paramètres (atomique au possible, discret en cas d'échec)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(settings.to_json(), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass
