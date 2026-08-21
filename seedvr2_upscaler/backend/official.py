"""Backend officiel SeedVR2 — enroule **les scripts d'inférence officiels**
du dépôt ByteDance-Seed/SeedVR au lieu de réimplémenter le pipeline.

Ce que ce module fait :
    - importe ``projects.video_diffusion_sr.infer.VideoDiffusionInfer``
      (pipeline officiel) et la configuration ``configs_3b/main.yaml`` ;
    - reproduit, pour une image = vidéo d'1 frame, le flux du script officiel
      ``projects/inference_seedvr2_3b.py`` (NaResize → DivisibleCrop →
      Normalize → VAE encode → 1 step DiT → VAE decode) ;
    - étend UNIQUEMENT le chargement des poids : safetensors FP16/BF16,
      FP8-e4m3fn et GGUF (via ``gguf_loader``), le script officiel ne
      chargeant que des ``.pth`` ;
    - gère CUDA + BF16/FP16 automatiquement, ``torch.inference_mode()``,
      swap CPU/GPU du DiT et du VAE (faible VRAM), vidage de cache.

Configuration requise (voir README) : le repo officiel cloné, ses
dépendances installées (torch, apex éventuel, etc.), plus le VAE et les
embeddings texte du modèle (téléchargés automatiquement au besoin).
"""
from __future__ import annotations

import gc
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL.Image import Image as PILImage

from ..constants import SEEDVR2_REPO_CANDIDATES, SEEDVR2_REPO_ENV
from ..models import ModelInfo, ModelKind, Precision
from ..registry import ensure_aux_assets
from .base import BackendUnavailable, ProgressCb, UpscaleBackend

if TYPE_CHECKING:
    import torch


def find_repo_root(start: Path | None = None) -> Path | None:
    """Localise le clone du repo officiel SeedVR2 (sans l'importer)."""
    base = start or Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    env = os.environ.get(SEEDVR2_REPO_ENV)
    if env:
        candidates.append(Path(env))
    candidates += [base / c for c in SEEDVR2_REPO_CANDIDATES]
    candidates += [base.parent / "SeedVR"]
    for candidate in candidates:
        marker = candidate / "projects" / "video_diffusion_sr" / "infer.py"
        if marker.exists():
            return candidate.resolve()
    return None


REPO_HELP = (
    "Le dépôt officiel SeedVR2 est introuvable.\n"
    "  1) git clone https://github.com/ByteDance-Seed/SeedVR\n"
    "  2) placez-le dans le dossier de l'application sous le nom « SeedVR »\n"
    "     (ou définissez la variable d'environnement SEEDVR2_REPO vers son chemin)\n"
    "  3) installez ses dépendances (voir README du dépôt : torch, apex…)."
)

# Modules du dépôt officiel → paquet PyPI correspondant (épingle assouplie :
# son requirements.txt fige torch==2.3.0, incompatible avec les Python récents).
_PYPI_NAMES: dict[str, str] = {
    "rotary_embedding_torch": "rotary-embedding-torch>=0.5",
    "diffusers": "diffusers>=0.29",
    "transformers": "transformers>=4.38",
    "cv2": "opencv-python",
    "mediapy": "mediapy",
    "einops": "einops>=0.7",
    "omegaconf": "omegaconf>=2.3",
}

_APEX_HELP = (
    "Le module « apex » (NVIDIA) est requis par le dépôt officiel et ne peut "
    "pas être installé automatiquement sous Windows.\n"
    "  → Installez la wheel précompilée fournie dans le README du dépôt "
    "ByteDance-Seed/SeedVR (section « Install apex »), adaptée à votre "
    "Python et à votre CUDA."
)


def _pip_install(spec: str, log: ProgressCb) -> bool:
    """Installe ``spec`` dans l'interpréteur courant (le .venv) via pip."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        log(f"pip install {spec} : lancement impossible ({exc}).")
        return False
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-6:])
        log(f"pip install {spec} : échec.\n{tail}")
        return False
    return True


class SeedVR2OfficialBackend(UpscaleBackend):
    """Upscaling via le pipeline officiel SeedVR2 (DiT 3B one-step + VAE)."""

    name = "official"

    def __init__(self, options=None, models_dir: Path | None = None) -> None:
        super().__init__(options)
        self.models_dir = Path(models_dir) if models_dir else None
        self.repo_root: Path | None = None
        self.runner = None                     # VideoDiffusionInfer (officiel)
        self.device: str = "cuda"
        self.autocast_dtype: "torch.dtype | None" = None
        self._pos_emb: "torch.Tensor | None" = None
        self._neg_emb: "torch.Tensor | None" = None
        # Copies GPU des embeddings texte : calculées une fois au premier
        # appel au lieu d'être retransférées à chaque tuile.
        self._texts_dev: "tuple[torch.Tensor, torch.Tensor] | None" = None
        self._color_fix = None                 # wavelet_reconstruction si dispo
        self._autoinstall_attempted: dict[str, str] = {}  # modules déjà réparés

    # ------------------------------------------------------------------ #
    # Chargement (une seule fois par lot)                                  #
    # ------------------------------------------------------------------ #
    def load(self, model: ModelInfo, log: ProgressCb = print) -> None:  # noqa: C901
        import torch

        # Diagnostic pilote + build torch (nvidia-smi inclus) : en cas d'échec,
        # le message contient la cause probable ET la commande de réparation —
        # indispensable depuis la suppression du repli « démo ».
        from ..gpu_check import detect_gpu, status_line, unavailable_help

        gpu = detect_gpu()
        log(status_line(gpu))
        if not gpu.ok:
            raise BackendUnavailable(unavailable_help(gpu))
        # Toutes les tuiles d'un lot ont la même forme : l'autotune des
        # convolutions (VAE surtout) est calculé une fois puis réutilisé.
        torch.backends.cudnn.benchmark = True
        self.repo_root = find_repo_root()
        if self.repo_root is None:
            raise BackendUnavailable(REPO_HELP)

        # Le repo officiel s'importe depuis sa racine (paquets common/, data/…).
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        log(f"Dépôt officiel : {self.repo_root}")
        # Ajustement VRAM automatique : le swap CPU↔GPU du DiT/VAE (mode faible
        # VRAM) divise fortement la vitesse quand la carte est large (≥ 14 Go),
        # et est indispensable sur les petites cartes (< 10 Go).
        self._tune_low_vram(torch, log)
        # FlashAttention et apex ne se compilent pas sous Windows : on installe
        # d'emblée les replis natifs (équivalents ici) plutôt que d'attendre
        # les exceptions d'import.
        from .apex_fallback import install_apex_shim
        from .flash_fallback import install_flash_attn_shim

        install_flash_attn_shim(log)
        install_apex_shim(log)
        # Réparation automatique des autres dépendances du dépôt
        # (rotary-embedding-torch, diffusers…) : un essai par module manquant.
        self._with_autorepair(lambda: self._load_pipeline(model, torch, log), log)
        self._model = model
        vram_go = torch.cuda.memory_allocated() / 1024**3
        log(f"Modèle chargé. VRAM occupée : {vram_go:.2f} Go.")

    def _tune_low_vram(self, torch: "torch", log: ProgressCb) -> None:
        """Adapte le mode faible VRAM à la carte réellement présente."""
        import dataclasses

        try:
            total_go = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            return
        if self.options.low_vram and total_go >= 14.0:
            log(f"VRAM {total_go:.0f} Go ≥ 14 : mode faible VRAM désactivé "
                "(le swap CPU↔GPU multiplie le temps par image).")
            self.options = dataclasses.replace(self.options, low_vram=False)
        elif not self.options.low_vram and total_go < 10.0:
            log(f"VRAM {total_go:.0f} Go < 10 : mode faible VRAM activé "
                "automatiquement pour éviter les OOM.")
            self.options = dataclasses.replace(self.options, low_vram=True)

    def _tune_vae_memory_limit(self, torch: "torch", log: ProgressCb, runner) -> None:
        """Calibre le budget mémoire du décodeur VAE sur la VRAM disponible."""
        try:
            free_go = torch.cuda.mem_get_info()[0] / 1024**3
        except Exception:
            free_go = 8.0
        # 0,5 Go (config d'origine) → micro-slices très lentes. On monte à
        # ~50 % de la VRAM libre (plafond 7 Go) : au-delà, le nombre de
        # slices ne diminue plus significativement pour ~1,3 Go de pic DiT.
        budget = min(7.0, max(1.0, round(free_go * 0.5, 1)))
        ml = runner.config.vae.memory_limit
        if float(ml.get("conv_max_mem", 0.5)) < budget:
            ml.conv_max_mem = budget
            ml.norm_max_mem = min(6.0, budget)
            log(f"Budget mémoire VAE (décodage) : {budget:.1f} Go "
                f"(0,5 Go d'origine = micro-slices lentes).")

    def _with_autorepair(self, fn, log: ProgressCb):
        """Exécute ``fn`` ; sur ModuleNotFoundError, installe le paquet puis réessaie."""
        try:
            return fn()
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing == "apex":
                # Chemin de secours si l'import d'apex survivait au shim proactif.
                from .apex_fallback import install_apex_shim

                install_apex_shim(log)
                return fn()
            if missing == "flash_attn":
                # Cas où l'import surviendrait malgré tout (chemin précautionneux).
                from .flash_fallback import install_flash_attn_shim

                install_flash_attn_shim(log)
                return fn()
            pkg = _PYPI_NAMES.get(missing, missing)
            if self._autoinstall_attempted.get(missing):
                raise BackendUnavailable(
                    f"Module « {missing} » toujours introuvable malgré "
                    f"l'installation de « {pkg} ». Lancez manuellement :\n"
                    f"  {sys.executable} -m pip install \"{pkg}\""
                ) from exc
            self._autoinstall_attempted[missing] = pkg
            log(f"Module manquant « {missing} » — installation automatique : {pkg}")
            if not _pip_install(pkg, log):
                raise BackendUnavailable(
                    f"Impossible d'installer « {pkg} » automatiquement. Lancez :\n"
                    f"  {sys.executable} -m pip install \"{pkg}\""
                ) from exc
            importlib.invalidate_caches()
            try:
                importlib.import_module(missing)
            except Exception as exc2:
                raise BackendUnavailable(
                    f"« {pkg} » installé mais « {missing} » ne s'importe pas ({exc2})."
                ) from exc2
            log(f"« {missing} » installé et importé — reprise du chargement.")
            return fn()

    def _load_pipeline(self, model: ModelInfo, torch: "torch", log: ProgressCb) -> None:
        """Construit le runner officiel (config, DiT, VAE, diffusion, embeds)."""
        from omegaconf import OmegaConf

        # Certains fichiers de config officiels référencent des chemins
        # relatifs : on charge la config depuis la racine du dépôt.
        old_cwd = Path.cwd()
        try:
            os.chdir(self.repo_root)
            from common.config import load_config

            config_dir = self._config_dir_for(model)
            config = load_config(str(config_dir / "main.yaml"))
        finally:
            os.chdir(old_cwd)

        from projects.video_diffusion_sr.infer import VideoDiffusionInfer

        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)

        self._init_distributed_best_effort(torch, log)

        # Précision : bf16 si le GPU le gère, sinon fp16 (exigence utilisateur).
        self.autocast_dtype = self._resolve_dtype(torch)
        log(f"Précision de calcul : {self.autocast_dtype}")

        # Assets auxiliaires officiels (VAE + embeddings texte).
        vae_path, self._pos_emb, self._neg_emb = self._load_aux(torch, log)
        runner.config.vae.checkpoint = str(vae_path)

        # ------------------------- DiT ----------------------------- #
        if model.kind is ModelKind.OFFICIAL or model.path.suffix.lower() in (".pth", ".pt"):
            # Chemin strictement officiel : torch.load du checkpoint EMA.
            runner.configure_dit_model(device="cuda", checkpoint=str(model.path))
        else:
            # Extension : safetensors FP16/FP8 ou GGUF quantifié.
            state = self._read_state_dict(model, torch, log)
            self._build_dit_from_state(runner, state, torch, log)
            del state
            gc.collect()

        # ------------------------- VAE ----------------------------- #
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            # La config officielle bride le VAE à 0,5 Go (conv_max_mem) :
            # le décodage se fait alors en micro-slices très lentes
            # (~17 s/tuile mesuré sur 16 Go, pour 1,7 s de diffusion !).
            # On calibre ce budget sur la VRAM réellement disponible.
            self._tune_vae_memory_limit(torch, log, runner)
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        # ------------------ Diffusion (1 step, cfg réglable) --------------- #
        # Le nombre de steps (1) reste figé : c'est le mode « turbo » distillé
        # du script officiel (generation_loop) — le seul supporté par le
        # checkpoint. cfg.scale/rescale sont en revanche de vrais réglages du
        # pipeline officiel, exposés ici en options avancées (1.0/0.0 =
        # comportement identique au script d'origine).
        runner.config.diffusion.cfg.scale = float(self.options.cfg_scale)
        runner.config.diffusion.cfg.rescale = float(self.options.cfg_rescale)
        runner.config.diffusion.timesteps.sampling.steps = 1
        runner.configure_diffusion()
        if self.options.cfg_scale != 1.0 or self.options.cfg_rescale != 0.0:
            log(f"Guidage (cfg) personnalisé : scale={self.options.cfg_scale}, "
                f"rescale={self.options.cfg_rescale} (officiel : 1.0 / 0.0).")

        # ColorFix optionnel du script officiel (wavelet_reconstruction).
        self._load_color_fix(log)
        self.runner = runner

    def _config_dir_for(self, model: ModelInfo) -> Path:
        """configs_3b par défaut ; configs_7b si le nom du fichier l'indique."""
        name = model.path.name.lower()
        if "7b" in name and (self.repo_root / "configs_7b").exists():
            return self.repo_root / "configs_7b"
        return self.repo_root / "configs_3b"

    # ------------------------------------------------------------------ #
    # Utilitaires de chargement                                            #
    # ------------------------------------------------------------------ #
    def _resolve_dtype(self, torch: "torch") -> "torch.dtype":
        """AUTO → bf16 si supporté sinon fp16 ; force BF16/FP16 sinon."""
        if self.options.precision is Precision.BF16:
            return torch.bfloat16
        if self.options.precision is Precision.FP16:
            return torch.float16
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16

    def _init_distributed_best_effort(self, torch: "torch", log: ProgressCb) -> None:
        """Initialise un contexte distribué mono-processus (NCCL → GLOO → shims).

        Pourquoi : le dépôt n'est pas prévu que pour ``torchrun`` — son
        décorateur ``log_runtime`` (qui entoure ``configure_dit_model`` et
        ``configure_vae_model``) appelle ``torch.distributed.barrier()`` sans
        condition. Sous Windows, ``init_torch`` (NCCL) échoue : on crée donc
        un **vrai groupe gloo mono-processus** (CPU, natif Windows), ce qui
        rend tous les appels ``dist.*`` directs fonctionnels. Les shims restent
        l'ultime recours si gloo est indisponible.
        """
        import datetime

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return  # un runner précédent a déjà initialisé le groupe

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        try:
            from common.distributed import init_torch

            init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
            return
        except Exception as exc:
            log(f"Init distribuée officielle impossible ({exc}) — tentative gloo.")

        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            dist.init_process_group(
                backend="gloo",  # CPU, disponible dans les builds Windows
                init_method="tcp://127.0.0.1:29512",
                rank=0,
                world_size=1,
                timeout=datetime.timedelta(seconds=3600),
            )
            log("Groupe gloo mono-processus initialisé (repli Windows).")
            return
        except Exception as exc2:
            log(f"Init gloo impossible ({exc2}) — mode local par shims.")

        # Dernier recours : remplacement des helpers par des fonctions locales.
        device = torch.device(self.device)

        def _get_device(*_a, **_k):
            return device

        def _zero(*_a, **_k):
            return 0

        def _one(*_a, **_k):
            return 1

        patches = {
            "get_device": _get_device,
            "get_global_rank": _zero,
            "get_local_rank": _zero,
            "get_world_size": _one,
            "get_data_parallel_rank": _zero,
            "get_data_parallel_world_size": _one,
            "get_sequence_parallel_rank": _zero,
            "get_sequence_parallel_world_size": _one,
        }
        modules = []
        for mod_name in (
            "common.distributed",
            "common.distributed.advanced",
            "projects.video_diffusion_sr.infer",
        ):
            try:
                modules.append(__import__(mod_name, fromlist=["*"]))
            except Exception:
                continue
        for module in modules:
            for attr, fn in patches.items():
                if hasattr(module, attr):
                    setattr(module, attr, fn)

    def _load_aux(self, torch: "torch", log: ProgressCb) -> tuple[Path, "torch.Tensor", "torch.Tensor"]:
        """VAE + embeddings texte précalculés (téléchargés si absents)."""
        models_dir = self.models_dir or self._model_dir_fallback()
        models_dir.mkdir(parents=True, exist_ok=True)
        assets = ensure_aux_assets(models_dir, log=log)
        assert assets.vae and assets.pos_emb and assets.neg_emb  # assuré par ensure_*
        pos = torch.load(assets.pos_emb, map_location="cpu")
        neg = torch.load(assets.neg_emb, map_location="cpu")
        self._texts_dev = None  # nouveau modèle → cache GPU des embeddings à refaire
        return assets.vae, pos, neg

    def _model_dir_fallback(self) -> Path:
        from ..constants import DEFAULT_MODELS_DIR

        return DEFAULT_MODELS_DIR

    @staticmethod
    def _strip_prefixes(state: dict) -> dict:
        for prefix in ("model.", "dit.", "module.", "ema."):
            if all(k.startswith(prefix) for k in state):
                return {k[len(prefix):]: v for k, v in state.items()}
        return state

    def _read_state_dict(self, model: ModelInfo, torch: "torch", log: ProgressCb) -> dict:
        """Lit le state_dict (safetensors FP16/FP8, ou GGUF) dans la précision cible."""
        suffix = model.path.suffix.lower()
        if suffix == ".gguf":
            from .gguf_loader import cast_state_dict, load_gguf_state_dict

            state = load_gguf_state_dict(model.path, log=log)
        else:
            from safetensors import safe_open

            log(f"Lecture safetensors : {model.path.name}…")
            state = {}
            with safe_open(str(model.path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    state[key] = handle.get_tensor(key)
        state = self._strip_prefixes(state)
        # FP8/F16/BF16 → dtype de calcul (e4m3fn est casté proprement par .to).
        return {
            k: (v.to(self.autocast_dtype) if v.is_floating_point() else v)
            for k, v in state.items()
        }

    def _build_dit_from_state(self, runner, state: dict, torch: "torch", log: ProgressCb) -> None:
        """Injecte un state_dict externe dans le DiT construit par la config officielle.

        Reproduit la branche « meta device » de ``configure_dit_model``
        (méthode officielle), en remplaçant seulement la source des poids.
        """
        from common.config import create_object

        with torch.device("meta"):
            runner.dit = create_object(runner.config.dit.model)
        runner.dit.set_gradient_checkpointing(runner.config.dit.get("gradient_checkpoint", False))
        info = runner.dit.load_state_dict(state, strict=False, assign=True)
        if info.missing_keys:
            log(f"Clés manquantes ({len(info.missing_keys)}) : {info.missing_keys[:5]}…")
        if info.unexpected_keys:
            log(f"Clés ignorées ({len(info.unexpected_keys)}) : {info.unexpected_keys[:5]}…")
        if len(info.missing_keys) > 0.2 * len(state):
            raise RuntimeError(
                "Trop de clés manquantes : ce fichier ne semble pas correspondre "
                "au DiT SeedVR2 attendu par la configuration officielle."
            )
        self._materialize_buffers(runner.dit, log)
        runner.dit.to(device=self.device, dtype=self.autocast_dtype)

    def _materialize_buffers(self, dit, log: ProgressCb) -> None:
        """Matérialise les buffers restés sur 'meta' après le chargement.

        Extension robuste du ``meta_non_persistent_buffer_init_fn`` officiel :
        celui-ci ne gère que le buffer ``dummy`` des ``RotaryEmbedding`` de
        ``rotary-embedding-torch==0.5.3`` — or les versions récentes (≥ 0.8)
        ajoutent ``cached_freqs`` (non persistant, absent des poids) et le
        script officiel termine alors par ``AssertionError()`` silencieux.

        - ``freqs`` (RoPE) : si le checkpoint ne le fournit pas, il est
          RECOMPOSÉ exactement comme le fait la bibliothèque
          (``1 / theta^(2i/d)``, freqs_for='lang', theta=10000) ;
        - ``cached_freqs`` / ``dummy`` : simples placeholders → zéros
          (c'est leur contenu à l'initialisation d'un module neuf) ;
        - tout autre buffer meta restant : zéros + avertissement.
        """
        import torch

        try:
            from rotary_embedding_torch import RotaryEmbedding
        except Exception:  # pragma: no cover
            RotaryEmbedding = ()

        n_total, n_freqs, n_divers = 0, 0, 0
        inconnus: list[str] = []
        with torch.no_grad():
            for module_name, sub in dit.named_modules():
                for buf_name, buf in sub.named_buffers(recurse=False):
                    if not buf.is_meta:
                        continue
                    if isinstance(sub, RotaryEmbedding) and buf_name == "freqs":
                        # Recomposition exacte (cf. rotary-embedding-torch,
                        # branche freqs_for == 'lang', theta = 10000).
                        dim = buf.shape[-1] * 2
                        t = torch.arange(0, dim, 2)[: dim // 2].float() / dim
                        value = 1.0 / (10000.0**t)
                        setattr(sub, buf_name, value.to(buf.dtype)
                                if buf.dtype.is_floating_point else value)
                        n_freqs += 1
                    elif isinstance(sub, RotaryEmbedding) and buf_name in (
                        "cached_freqs", "cached_scales", "dummy", "scale"
                    ):
                        setattr(sub, buf_name, torch.zeros_like(buf, device="cpu"))
                        n_divers += 1
                    else:
                        setattr(sub, buf_name, torch.zeros_like(buf, device="cpu"))
                        inconnus.append(f"{module_name}.{buf_name}")
                    n_total += 1

        if n_freqs:
            log(f"{n_freqs} buffers 'freqs' RoPE recomposés (absents des poids).")
        restants = [n for n, b in dit.named_buffers() if b.is_meta]
        if restants:  # ne doit plus arriver ; l'officiel levait un assert muet
            log(f"Avertissement : buffers meta résiduels {restants[:5]}")
        if inconnus:
            log(f"Buffers meta matérialisés en zéros : {inconnus[:5]}")
        if n_freqs + n_divers + n_total == 0:
            log("Tous les buffers ont été matérialisés depuis les poids.")

    def _load_color_fix(self, log: ProgressCb) -> None:
        """Wavelet color fix officiel si le fichier a été ajouté au dépôt."""
        if not self.options.color_fix:
            log("Color fix désactivé (option avancée).")
            return
        color_fix_path = self.repo_root / "projects" / "video_diffusion_sr" / "color_fix.py"
        if not color_fix_path.exists():
            return
        try:
            from projects.video_diffusion_sr.color_fix import wavelet_reconstruction

            self._color_fix = wavelet_reconstruction
            log("Color fix wavelet officiel activé.")
        except Exception as exc:
            log(f"Color fix indisponible ({exc}), il sera ignoré.")


    # ------------------------------------------------------------------ #
    # Inférence d'une région (image = vidéo d'une frame)                   #
    # ------------------------------------------------------------------ #
    def _upscale_region(
        self, region: PILImage, out_w: int, out_h: int, seed: int, log: ProgressCb
    ) -> PILImage:
        # Les imports du dépôt (transforms d'images…) sont résolus ici aussi :
        # même réparation automatique qu'au chargement.
        return self._with_autorepair(
            lambda: self._upscale_region_impl(region, out_w, out_h, seed, log), log
        )

    def _upscale_region_impl(
        self, region: PILImage, out_w: int, out_h: int, seed: int, log: ProgressCb
    ) -> PILImage:
        import torch
        from einops import rearrange
        from torchvision.transforms import Compose, Lambda, Normalize
        from torchvision.transforms import functional as TF

        # Transforms officielles du repo (mêmes classes, même ordre).
        from data.image.transforms.divisible_crop import DivisibleCrop
        from data.image.transforms.na_resize import NaResize
        from data.video.transforms.rearrange import Rearrange

        runner = self.runner
        device = torch.device(self.device)

        # Reproductibilité par image (set_seed officiel, avec repli).
        try:
            from common.seed import set_seed

            set_seed(seed, same_across_ranks=True)
        except Exception:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Chaîne de transforms identique au script officiel (branche image) :
        # read_image → /255 → NaResize → clamp → DivisibleCrop → Normalize →
        # Rearrange "t c h w -> c t h w"  (T = 1 pour une image).
        video_transform = Compose(
            [
                NaResize(
                    resolution=(out_h * out_w) ** 0.5,
                    mode="area",
                    downsample_only=False,  # upscale autorisé (modèle entraîné haute rés.)
                ),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ]
        )

        # Swap VRAM officiel : VAE sur GPU pendant l'encodage, DiT sur CPU.
        def _sync() -> float:
            if device.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter()

        t_pre = _sync()
        frame = TF.to_tensor(region).unsqueeze(0)  # [T=1, C, H, W], 0..1
        cond_input = video_transform(frame.to(device))  # [C, 1, H', W']
        t0 = _sync()
        if self.options.low_vram:
            runner.dit.to("cpu")
        runner.vae.to(device)
        cond_latents = runner.vae_encode([cond_input])
        if self.options.low_vram:
            runner.vae.to("cpu")
            runner.dit.to(device)
        t1 = _sync()

        sample = self._generation_step(cond_latents)
        t2 = _sync()

        if self.options.low_vram:
            runner.dit.to("cpu")

        if self._color_fix is not None:
            # Comparaison wavelet avec l'entrée transformée à la résolution
            # cible (même flux que le script officiel, tolérant aux écarts).
            input_tc = rearrange(cond_input, "c t h w -> t c h w")
            try:
                sample = self._color_fix(sample.to("cpu"), input_tc.to("cpu"))
            except Exception as exc:
                log(f"Color fix ignoré ({exc}).")
                sample = sample.to("cpu")
        else:
            sample = sample.to("cpu")
        del cond_latents, cond_input

        array = (
            sample[0]
            .permute(1, 2, 0)
            .clamp(-1, 1)
            .mul_(0.5)
            .add_(0.5)
            .mul_(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        from PIL import Image

        t3 = _sync()
        log(f"    étapes — prétraitement {t0 - t_pre:.1f}s | encode VAE {t1 - t0:.1f}s | "
            f"diffusion+décodage {t2 - t1:.1f}s | conversion {t3 - t2:.1f}s")
        return Image.fromarray(np.ascontiguousarray(array), "RGB")

    def _generation_step(self, cond_latents: list) -> "torch.Tensor":
        """Réplique (avec bruit de condition réglable) de ``generation_step`` officiel.

        SeedVR2 est « one-step » : un seul pas de diffusion avec cfg réglable.
        Le script officiel fixe ``cond_noise_scale = 0.0`` ; c'est aussi la
        valeur par défaut ici, mais l'option avancée permet de l'augmenter
        légèrement (une petite valeur peut réintroduire un peu de texture).
        """
        import torch

        runner = self.runner
        device = torch.device(self.device)

        noises = [torch.randn_like(latent) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent) for latent in cond_latents]
        noises = [n.to(device) for n in noises]
        aug_noises = [n.to(device) for n in aug_noises]
        cond_latents = [c.to(device) for c in cond_latents]
        cond_noise_scale = float(self.options.cond_noise_scale)  # 0.0 = valeur officielle


        def _add_noise(x, aug_noise):
            t = torch.tensor([1000.0], device=device) * cond_noise_scale
            shape = torch.tensor(x.shape[1:], device=device)[None]
            t = runner.timestep_transform(t, shape)
            return runner.schedule.forward(x, aug_noise, t)

        conditions = [
            runner.get_condition(noise, task="sr", latent_blur=_add_noise(latent, aug))
            for noise, aug, latent in zip(noises, aug_noises, cond_latents)
        ]

        text_embeds = {
            # Tensors déjà sur GPU (cache de _texts_on_device) : aucun transfert
            # par tuile, contrairement à un .to(device) à chaque appel.
            "texts_pos": [self._texts_on_device(device)[0]],
            "texts_neg": [self._texts_on_device(device)[1]],
        }

        with torch.inference_mode(), torch.autocast("cuda", self.autocast_dtype, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=self.options.low_vram,  # offload interne officiel
                **text_embeds,
            )

        video = video_tensors[0]
        # Convention EXACTE du script officiel : ndim == 3 signifie [C, H, W]
        # (la dimension temps a été perdue à T == 1) — video[:, None] et non
        # video[None], sinon les axes C/T sont transposés silencieusement et
        # l'image de sortie n'a plus qu'un canal (« not enough image data »).
        sample = (
            rearrange_static(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange_static(video, "c t h w -> t c h w")
        )
        del video_tensors
        return sample

    def _texts_on_device(self, device: "torch.device") -> "tuple[torch.Tensor, torch.Tensor]":
        """Embeddings texte sur GPU, transférés une seule fois par chargement."""
        if self._texts_dev is None:
            self._texts_dev = (self._pos_emb.to(device), self._neg_emb.to(device))
        return self._texts_dev

    def unload(self) -> None:
        """Libère complètement la VRAM (appelé en fin de lot)."""
        import torch

        runner = self.runner
        if runner is not None:
            for attr in ("dit", "vae"):
                module = getattr(runner, attr, None)
                if module is not None:
                    module.to("cpu")
                    delattr(runner, attr)
        self.runner = None
        self._pos_emb = self._neg_emb = None
        self._texts_dev = None
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rearrange_static(tensor, pattern: str):
    """``einops.rearrange`` paresseux (évite l'import au niveau module)."""
    from einops import rearrange

    return rearrange(tensor, pattern)
