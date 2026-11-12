"""Interface graphique Gradio de SeedVR2 Batch Upscaler — bilingue FR/EN.

Architecture : l'UI ne fait AUCUN calcul lourd. Le ``BatchRunner`` tourne
dans son thread ; cet module se contente de :
    1. construire le ``JobConfig`` à partir des widgets (bouton Démarrer) ;
    2. relayer les événements du moteur (``EventBus``) vers les widgets,
       via un ``gr.Timer`` qui vidange la file toutes les 0,5 s.

Langue : tous les textes viennent de ``i18n.py``. Chaque widget concerné est
enregistré dans ``GuiSession.text_registry`` ; le sélecteur de langue applique
``gr.update`` à tout le registre — **sans recharger la page**.

Les radios utilisent des valeurs **canoniques** (ex. ``"keep"``, ``"custom"``)
indépendantes de la langue affichée : la logique et la persistance ne
dépendent jamais du libellé visible.

Inclus : sélection de dossiers, glisser-déposer, liste des modèles détectés,
téléchargement de presets, échelle, format/qualité, suffixe optionnel,
conflits, tuilage, progression + ETA + sous-progression tuiles, journal temps
réel, aperçu avant/après, Démarrer / Pause / Reprendre / Annuler, ouverture
du dossier de sortie, bandeau GPU.
"""
from __future__ import annotations

import threading
import dataclasses
import json
import tempfile
import shutil
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import gradio as gr

from .constants import DEFAULT_MODELS_DIR, DEFAULT_SEED, PRESET_DOWNLOADS
from .gpu_check import cause_key, detect_gpu
from .i18n import SUPPORTED_LANGUAGES, tr
from .models import (
    ConflictPolicy,
    JobConfig,
    ModelInfo,
    OutputFormat,
    Precision,
    RunnerState,
    TilingConfig,
)
from .registry import download_file, scan_models
from .settings import Settings, load_settings, save_settings
from .worker import BatchRunner, EventBus, _fmt_duration
from .jobs import preflight, scan_inputs, atomic_json

SETTING_KEYS = ["input_dir", "output_dir", "model_choice", "scale_choice", "scale_custom",
            "output_format", "quality", "add_suffix", "suffix", "conflict_policy",
            "tiling", "tile_size", "tile_overlap", "seed", "precision", "low_vram",
            "clear_cache", "parallel_io", "resume", "backend",
            "cfg_scale", "cfg_rescale", "cond_noise_scale", "color_fix",
            "recursive", "memory_profile", "retry_failed"]

_LOG_MAX_LINES = 600

# Valeurs canoniques (jamais traduites) ; libellés via i18n.
_SCALE_VALUES = ("x2", "x4", "x8", "custom")
_FORMAT_VALUES = ("keep", "png", "jpg", "webp")
_CONFLICT_VALUES = ("overwrite", "skip", "rename")

_CSS = """
#journal textarea {font-family: Consolas, "Cascadia Mono", monospace !important; font-size: 12px;}
.progress-outer {background: #2b2f3a; border-radius: 8px; height: 22px; overflow: hidden;}
.progress-inner {background: linear-gradient(90deg,#4f8ef7,#7bd88f); height: 100%;
                 transition: width .4s ease; color:#fff; font-size:12px; text-align:center; line-height:22px;}
"""


# --------------------------------------------------------------------------- #
# État partagé de la session (application locale mono-utilisateur)            #
# --------------------------------------------------------------------------- #
class GuiSession:
    """Mutable : runner courant, bus d'événements, langue et registre i18n."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.settings: Settings = load_settings()
        self.language: str = self.settings.language  # « fr » | « en »
        self.runner: BatchRunner | None = None
        self.bus: EventBus | None = None
        self.log_lines: deque[str] = deque(maxlen=_LOG_MAX_LINES)
        self.progress: tuple[int, int, float | None] = (0, 0, None)
        self.tile: tuple[int, int] = (0, 0)  # sous-progression « tuile i/N »
        self.state: RunnerState = RunnerState.IDLE
        self.preview: tuple[Any, Any] | None = None
        self.models_by_label: dict[str, ModelInfo] = {}
        self.pending_jobs: deque[JobConfig] = deque()
        self.latest_result: dict | None = None
        self.lock = threading.RLock()
        self.upload_folder: str | None = None
        # (composant, {prop: spec}) — spec = (« s », clé) | (« choices », …) | (« f », nom)
        self.text_registry: list[tuple[Any, dict[str, tuple[str, Any]]]] = []
        self.refresh_models()

    # --------------------------------------------------------------- modèles
    def refresh_models(self) -> list[str]:
        """Liste déroulante = fichiers réels de models/ uniquement (plus de démo)."""
        found = scan_models(self.models_dir)
        self.models_by_label = {m.label: m for m in found}
        return list(self.models_by_label)

    @property
    def busy(self) -> bool:
        if self.runner is not None and self.runner.is_alive:
            return True
        return self.state in {
            RunnerState.LOADING, RunnerState.RUNNING,
            RunnerState.PAUSED, RunnerState.CANCELLING,
        }


# --------------------------------------------------------------------------- #
# Helpers de rendu                                                             #
# --------------------------------------------------------------------------- #
def _progress_html(done: int, total: int) -> str:
    pct = (100.0 * done / total) if total else 0.0
    return (
        f'<div class="progress-outer"><div class="progress-inner" '
        f'style="width:{pct:.1f}%">{done}/{total} — {pct:.0f}%</div></div>'
    )


def _buttons_for(state: RunnerState) -> tuple[gr.update, gr.update, gr.update, gr.update]:
    """(Démarrer, Pause, Reprendre, Annuler) activés selon l'état du moteur."""
    busy = {
        RunnerState.LOADING, RunnerState.RUNNING,
        RunnerState.PAUSED, RunnerState.CANCELLING,
    }
    return (
        gr.update(interactive=state not in busy),
        gr.update(interactive=state is RunnerState.RUNNING),
        gr.update(interactive=state is RunnerState.PAUSED),
        gr.update(interactive=state in busy),
    )


def _scale_value(choice: str, custom: float) -> float:
    return float(custom) if choice == "custom" else float(str(choice).lstrip("x"))


def _gpu_banner(session: GuiSession) -> str:
    """Bandeau GPU *traduit* (contrairement au journal technique, en français)."""
    status = detect_gpu()
    lang = session.language
    if status.ok:
        bf16 = "✓" if status.bf16 else tr(lang, "gpu.bf16_no")
        return tr(lang, "gpu.ok", name=status.device_name, vram=status.vram_gib,
                  driver=status.driver_version or "?", torch=status.torch_version,
                  cuda=status.cuda_build, bf16=bf16)
    cause = cause_key(status) or "unknown"
    lines = [tr(lang, "gpu.ko"), f"- {tr(lang, f'gpu.cause.{cause}')}"]
    hint = tr(lang, f"gpu.hint.{cause}")
    if hint:
        lines.append(hint)
    lines.append(tr(lang, "gpu.ko.footer"))
    return "\n".join(lines)


# Contenus dynamiques référencés dans le registre de traduction.
_DYNAMIC_BUILDERS = {"gpu_banner": _gpu_banner}


def _language_updates(session: GuiSession) -> dict[Any, gr.update]:
    """Un ``gr.update`` par composant enregistré, dans la langue courante."""
    updates: dict[Any, gr.update] = {}
    for component, spec in session.text_registry:
        kwargs: dict[str, Any] = {}
        for prop, (kind, payload) in spec.items():
            if kind == "s":       # simple clé de traduction
                kwargs[prop] = tr(session.language, payload)
            elif kind == "choices":  # radios : (libellé affiché, valeur canonique)
                kwargs[prop] = [(tr(session.language, key), value)
                                for key, value in payload]
            elif kind == "f":     # contenu dynamique (bandeau GPU)
                kwargs[prop] = _DYNAMIC_BUILDERS[payload](session)
        updates[component] = gr.update(**kwargs)
    return updates


def _on_language_change(session: GuiSession, code: str) -> dict[Any, gr.update]:
    """Bascule la langue à chaud (persistée dans settings.json)."""
    code = code if code in SUPPORTED_LANGUAGES else "fr"
    session.language = code
    current = load_settings()  # recharge pour ne pas écraser l'état courant
    current.language = code
    save_settings(current)
    return _language_updates(session)


# --------------------------------------------------------------------------- #
# Construction de l'application                                                #
# --------------------------------------------------------------------------- #
def build_app(models_dir: Path | None = None,
              session: GuiSession | None = None) -> gr.Blocks:
    session = session or GuiSession(models_dir or DEFAULT_MODELS_DIR)
    s = session.settings
    lang = session.language

    def reg(component: Any, **spec: tuple[str, Any]) -> Any:
        """Enregistre un composant pour la bascule de langue à chaud.

        spec : ``label=("s", "clé.i18n")``, ``value=("s", ...)``,
        ``choices=("choices", [(clé_i18n, valeur_canonique), …])``,
        ``value=("f", "gpu_banner")`` pour un contenu dynamique.
        """
        session.text_registry.append((component, spec))
        return component

    with gr.Blocks(title="SeedVR2 Batch Upscaler", css=_CSS) as demo:
        with gr.Row():
            # Markdown n'accepte pas « scale » : on l'enrobe dans une Column.
            with gr.Column(scale=4):
                reg(gr.Markdown(tr(lang, "app.title")), value=("s", "app.title"))
            with gr.Column(scale=1, min_width=170):
                language = reg(gr.Dropdown(
                    choices=[(label, code) for code, label in SUPPORTED_LANGUAGES.items()],
                    value=session.language, label=tr(lang, "lang.label"),
                    interactive=True,
                ), label=("s", "lang.label"))
        # Bandeau GPU rempli au chargement de la page (demo.load ci-dessous) :
        # il importe torch et interroge nvidia-smi — d'où le texte d'attente.
        gpu_banner_md = reg(gr.Markdown(tr(lang, "gpu.wait")),
                            value=("f", "gpu_banner"))

        # ------------------------------------------------------------ dossiers
        with gr.Row():
            input_dir = reg(gr.Textbox(
                label=tr(lang, "input_dir.label"),
                value=s.input_dir, scale=3,
                placeholder=tr(lang, "input_dir.ph"),
            ), label=("s", "input_dir.label"), placeholder=("s", "input_dir.ph"))
            output_dir = reg(gr.Textbox(
                label=tr(lang, "output_dir.label"), value=s.output_dir, scale=3,
                placeholder=tr(lang, "output_dir.ph"),
            ), label=("s", "output_dir.label"), placeholder=("s", "output_dir.ph"))
            open_out_btn = reg(gr.Button(tr(lang, "open.btn"), scale=1, min_width=90),
                               value=("s", "open.btn"))
        drop = reg(gr.File(
            label=tr(lang, "drop.label"),
            file_count="directory", file_types=["image"], type="filepath",
        ), label=("s", "drop.label"))

        # ------------------------------------------------------------ modèle
        with gr.Row():
            model_dropdown = reg(gr.Dropdown(
                label=tr(lang, "model.label"),
                choices=session.refresh_models(),
                value=_initial_model_choice(session),
                interactive=True, scale=4,
            ), label=("s", "model.label"))
            refresh_btn = reg(gr.Button(tr(lang, "refresh.btn"), scale=1),
                              value=("s", "refresh.btn"))
        with gr.Row():
            preset = reg(gr.Dropdown(
                label=tr(lang, "preset.label"),
                choices=[p[0] for p in PRESET_DOWNLOADS],
                value=PRESET_DOWNLOADS[0][0], scale=3,
            ), label=("s", "preset.label"))
            download_btn = reg(gr.Button(tr(lang, "download.btn"), scale=1),
                               value=("s", "download.btn"))
            with gr.Column(scale=3):
                download_status = gr.Markdown("")

        # ------------------------------------------------------- réglages de base
        with gr.Row():
            scale_choice = reg(gr.Radio(
                [(tr(lang, f"scale.{v}"), v) for v in _SCALE_VALUES],
                value=s.scale_choice,
                label=tr(lang, "scale.label"), scale=2,
            ), label=("s", "scale.label"),
                choices=("choices", [(f"scale.{v}", v) for v in _SCALE_VALUES]))
            scale_custom = reg(gr.Number(value=s.scale_custom, minimum=1.0, maximum=16.0,
                                         step=0.5, label=tr(lang, "scale.custom_label")),
                               label=("s", "scale.custom_label"))
            output_format = reg(gr.Radio(
                [(tr(lang, f"fmt.{v}"), v) for v in _FORMAT_VALUES],
                value=s.output_format,
                label=tr(lang, "fmt.label"),
            ), label=("s", "fmt.label"),
                choices=("choices", [(f"fmt.{v}", v) for v in _FORMAT_VALUES]))
            quality = reg(gr.Slider(1, 100, value=s.quality, step=1,
                                    label=tr(lang, "quality.label")),
                          label=("s", "quality.label"))

        # ------------------------------------------------------------ avancé
        with reg(gr.Accordion(tr(lang, "adv.accordion"), open=False),
                 label=("s", "adv.accordion")):
            reg(gr.Markdown(tr(lang, "suffix.note")), value=("s", "suffix.note"))
            with gr.Row():
                add_suffix = reg(gr.Checkbox(value=s.add_suffix,
                                             label=tr(lang, "suffix.chk")),
                                 label=("s", "suffix.chk"))
                suffix = reg(gr.Textbox(value=s.suffix, label=tr(lang, "suffix.label"),
                                        scale=2),
                             label=("s", "suffix.label"))
            conflict = reg(gr.Radio(
                [(tr(lang, f"conflict.{v}"), v) for v in _CONFLICT_VALUES],
                value=s.conflict_policy,
                label=tr(lang, "conflict.label"),
            ), label=("s", "conflict.label"),
                choices=("choices", [(f"conflict.{v}", v) for v in _CONFLICT_VALUES]))
            reg(gr.Markdown(tr(lang, "tiling.md")), value=("s", "tiling.md"))
            with gr.Row():
                tiling = reg(gr.Checkbox(value=s.tiling, label=tr(lang, "tiling.chk")),
                             label=("s", "tiling.chk"))
                tile_size = reg(gr.Number(value=s.tile_size, minimum=256, maximum=2048,
                                          step=64, label=tr(lang, "tile.size_label")),
                                label=("s", "tile.size_label"))
                tile_overlap = reg(gr.Number(value=s.tile_overlap, minimum=16,
                                             maximum=256, step=16,
                                             label=tr(lang, "tile.overlap_label")),
                                   label=("s", "tile.overlap_label"))
            reg(gr.Markdown(tr(lang, "engine.md")), value=("s", "engine.md"))
            with gr.Row():
                seed = gr.Number(value=s.seed if s.seed else DEFAULT_SEED, precision=0,
                                 label=tr(lang, "seed.label"))
                precision = gr.Radio(["auto", "bf16", "fp16"], value=s.precision,
                                     label=tr(lang, "precision.label"))
                backend = gr.Radio(["auto", "official"], value=s.backend,
                                   label=tr(lang, "backend.label"))
                reg(seed, label=("s", "seed.label"))
                reg(precision, label=("s", "precision.label"))
                reg(backend, label=("s", "backend.label"))
            with gr.Row():
                low_vram = reg(gr.Checkbox(value=s.low_vram,
                                           label=tr(lang, "low_vram.label")),
                               label=("s", "low_vram.label"))
                clear_cache = reg(gr.Checkbox(value=s.clear_cache,
                                              label=tr(lang, "clear_cache.label")),
                                  label=("s", "clear_cache.label"))
                parallel_io = reg(gr.Checkbox(value=s.parallel_io,
                                              label=tr(lang, "parallel_io.label")),
                                  label=("s", "parallel_io.label"))
                resume = reg(gr.Checkbox(value=s.resume,
                                         label=tr(lang, "resume.label")),
                             label=("s", "resume.label"))

            with reg(gr.Accordion(tr(lang, "expert.accordion"), open=False),
                     label=("s", "expert.accordion")):
                reg(gr.Markdown(tr(lang, "expert.md")), value=("s", "expert.md"))
                with gr.Row():
                    cfg_scale = reg(gr.Slider(0.5, 3.0, value=s.cfg_scale, step=0.05,
                                              label=tr(lang, "cfg_scale.label")),
                                    label=("s", "cfg_scale.label"))
                    cfg_rescale = reg(gr.Slider(0.0, 1.0, value=s.cfg_rescale, step=0.05,
                                                label=tr(lang, "cfg_rescale.label")),
                                      label=("s", "cfg_rescale.label"))
                with gr.Row():
                    cond_noise_scale = reg(
                        gr.Slider(0.0, 0.5, value=s.cond_noise_scale, step=0.01,
                                 label=tr(lang, "cond_noise_scale.label")),
                        label=("s", "cond_noise_scale.label"))
                    color_fix = reg(gr.Checkbox(value=s.color_fix,
                                                label=tr(lang, "color_fix.label")),
                                    label=("s", "color_fix.label"))

        with gr.Accordion("Lots et mémoire / Batches & memory", open=False):
            recursive = gr.Checkbox(value=s.recursive, label="Inclure les sous-dossiers / Recursive")
            memory_profile = gr.Dropdown(choices=[("Automatique / Auto", "auto"),
                ("Économe / Low memory", "eco"), ("Équilibré / Balanced", "balanced"),
                ("Rapide, plus de VRAM / Fast", "fast")], value=s.memory_profile,
                label="Profil mémoire / Memory profile")
            retry_failed = gr.Checkbox(value=False, label="Uniquement les erreurs précédentes / Retry failed only")
            with gr.Row():
                check_btn = gr.Button("Vérifier le lot / Check batch")
                enqueue_btn = gr.Button("Ajouter à la file / Enqueue")
                queue_start = gr.Button("Lancer la file / Run queue")
                queue_clear = gr.Button("Vider la file / Clear queue")
            queue_status = gr.Textbox(label="File d'attente / Queue", interactive=False)
            check_status = gr.Textbox(label="Vérification / Preflight", interactive=False)
        with gr.Accordion("Préréglages / Presets", open=False):
            export_btn = gr.Button("Exporter les réglages / Export settings")
            export_file = gr.File(label="Préréglage exporté / Exported preset", interactive=False)
            import_file = gr.File(label="Importer un préréglage JSON / Import JSON preset", file_types=[".json"], type="filepath")
        with gr.Accordion("Essai sur une zone / Crop test", open=False):
            gr.Markdown("Vérifiez le lot pour choisir une image. Le crop peut différer du rendu complet. "
                        "Sorties isolées dans `.seedvr2_tests`. / Check batch to select an image; crop context differs.")
            test_source = gr.Dropdown(label="Image de test / Test image", choices=[])
            with gr.Row():
                crop_x = gr.Number(value=0, minimum=0, precision=0, label="X")
                crop_y = gr.Number(value=0, minimum=0, precision=0, label="Y")
                crop_w = gr.Number(value=256, minimum=1, precision=0, label="Largeur / Width")
                crop_h = gr.Number(value=256, minimum=1, precision=0, label="Hauteur / Height")
            test_btn = gr.Button("Tester cette zone / Test crop")

        # ------------------------------------------------------------ boutons
        with gr.Row():
            start_btn = reg(gr.Button(tr(lang, "btn.start"), variant="primary",
                                      scale=2), value=("s", "btn.start"))
            pause_btn = reg(gr.Button(tr(lang, "btn.pause"), interactive=False),
                            value=("s", "btn.pause"))
            resume_btn = reg(gr.Button(tr(lang, "btn.resume"), interactive=False),
                             value=("s", "btn.resume"))
            cancel_btn = reg(gr.Button(tr(lang, "btn.cancel"), interactive=False),
                             value=("s", "btn.cancel"))

        # ------------------------------------------------------------ statut
        with gr.Row():
            with gr.Column(scale=1):
                status = gr.Markdown(tr(lang, f"status.{RunnerState.IDLE.value}"))
            with gr.Column(scale=1):
                eta = gr.Markdown("")
            with gr.Column(scale=1):
                counter = gr.Markdown("")
        progress_html = gr.HTML(_progress_html(0, 0))

        # ------------------------------------------------------------ aperçu
        try:
            preview = gr.ImageSlider(
                label=tr(lang, "preview.label"),
                type="pil", interactive=False, slider_position=0.5,
            )
            reg(preview, label=("s", "preview.label"))
        except AttributeError:  # anciennes versions de Gradio
            with gr.Row():
                preview = gr.Image(label=tr(lang, "preview.before"), type="pil")
                gr.Image(label=tr(lang, "preview.after"), type="pil")

        with gr.Accordion("Inspection synchronisée 100 % / Pixel inspection", open=False):
            gr.Markdown("Coordonnées dans la sortie, zone 512 × 512. À 1× : un pixel image par pixel CSS. "
                        "/ Output coordinates; 1× shows one image pixel per CSS pixel.")
            with gr.Row():
                inspect_x = gr.Number(value=0, minimum=0, precision=0, label="X sortie / Output X")
                inspect_y = gr.Number(value=0, minimum=0, precision=0, label="Y sortie / Output Y")
                inspect_zoom = gr.Dropdown(choices=[1, 2, 4], value=1, label="Zoom")
                inspect_btn = gr.Button("Inspecter le dernier résultat / Inspect last result")
            inspection = gr.ImageSlider(type="pil", format="png", width=512, height=512,
                                        max_height=512, slider_position=50, interactive=False)
        with gr.Accordion("Historique / History", open=False):
            history_btn = gr.Button("Actualiser l'historique / Refresh history")
            history_view = gr.JSON(label="20 derniers lots / Last 20 batches")

        journal = reg(gr.Textbox(label=tr(lang, "journal.label"), lines=14,
                                 max_lines=14, interactive=False, autoscroll=True,
                                 elem_id="journal"),
                      label=("s", "journal.label"))

        # ------------------------------------------------------------ minuterie
        outputs_tick = [journal, progress_html, eta, counter, status, preview,
                        start_btn, pause_btn, resume_btn, cancel_btn]
        timer = gr.Timer(0.5)
        timer.tick(fn=lambda: _on_tick(session), outputs=outputs_tick)

        # ------------------------------------------------------------ événements
        setting_inputs = [input_dir, output_dir, model_dropdown, scale_choice,
                          scale_custom, output_format, quality, add_suffix, suffix,
                          conflict, tiling, tile_size, tile_overlap, seed, precision,
                          low_vram, clear_cache, parallel_io, resume, backend,
                          cfg_scale, cfg_rescale, cond_noise_scale, color_fix,
                          recursive, memory_profile, retry_failed]
        for component in setting_inputs:
            component.change(_persist_settings, inputs=setting_inputs)

        drop.change(lambda files: _on_drop(session, files), inputs=drop,
                    outputs=input_dir)
        refresh_btn.click(lambda: _on_refresh(session), outputs=model_dropdown)
        download_btn.click(lambda p: _on_download(session, p), inputs=preset,
                           outputs=[model_dropdown, download_status])
        open_out_btn.click(lambda t: _open_folder(session, t), inputs=output_dir)
        start_btn.click(
            lambda *vals: _on_start(session, *vals), inputs=setting_inputs,
            outputs=[status],
        )
        pause_btn.click(lambda: _safe_call(session, "pause"))
        resume_btn.click(lambda: _safe_call(session, "resume"))
        cancel_btn.click(lambda: _safe_call(session, "cancel"))

        check_btn.click(lambda *v: _check_job(session, *v), inputs=setting_inputs,
                        outputs=[check_status, test_source])
        enqueue_btn.click(lambda *v: _enqueue_job(session, *v), inputs=setting_inputs, outputs=queue_status)
        queue_start.click(lambda: _start_queue(session), outputs=queue_status)
        queue_clear.click(lambda: _clear_queue(session), outputs=queue_status)
        timer.tick(lambda: _queue_text(session), outputs=queue_status)
        export_btn.click(_export_preset, inputs=setting_inputs, outputs=export_file)
        import_file.upload(lambda f: _import_preset(session, f), inputs=import_file, outputs=setting_inputs)
        test_btn.click(lambda source, x, y, w, h, *v: _test_crop(session, source, x, y, w, h, *v),
                       inputs=[test_source, crop_x, crop_y, crop_w, crop_h] + setting_inputs, outputs=status)
        inspect_btn.click(lambda x, y, z: _inspect_result(session, x, y, z),
                          inputs=[inspect_x, inspect_y, inspect_zoom], outputs=inspection)
        history_btn.click(_history, inputs=output_dir, outputs=history_view)

        # Bascule de langue à chaud sur TOUT le registre enregistré ci-dessus.
        language.change(
            lambda code: _on_language_change(session, code),
            inputs=language,
            outputs=[c for c, _ in session.text_registry],
        )

        # Remplit le bandeau GPU après le premier rendu de la page
        # (import torch + nvidia-smi, quelques secondes au plus).
        demo.load(lambda: _gpu_banner(session), outputs=gpu_banner_md)

    return demo


# --------------------------------------------------------------------------- #
# Valeurs initiales                                                            #
# --------------------------------------------------------------------------- #
def _initial_model_choice(session: GuiSession) -> str | None:
    choices = session.refresh_models()
    saved = session.settings.model_choice
    for label in choices:
        if session.models_by_label[label].path.name == saved or label == saved:
            return label
    return choices[0] if choices else None


def _on_drop(session: GuiSession, files: list[str] | None) -> object:
    """Le glisser-déposer copie les fichiers dans un dossier temporaire Gradio :
    on utilise ce dossier comme entrée (noms de fichiers conservés)."""
    if not files:
        if session.upload_folder:
            session.upload_folder = None
            return gr.update(value="")
        return gr.update()
    # Gradio may store each upload in a separate hash directory.
    # Snapshot exactly the visible selection instead of scanning only the first parent.
    folder = Path(tempfile.mkdtemp(prefix="seedvr2-input-"))
    from .naming import resolve_output_path
    for filename in files:
        src = Path(filename)
        dest = resolve_output_path(folder, src.name, ConflictPolicy.RENAME)
        shutil.copy2(src, dest)
    session.upload_folder = str(folder)
    session.log_lines.append(tr(session.language, "drop.logged", folder=folder))
    return gr.update(value=str(folder))


def _on_refresh(session: GuiSession) -> object:
    choices = session.refresh_models()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _open_folder(session: GuiSession, folder_text: str) -> None:
    """Ouvre le dossier de sortie dans l'Explorateur (le serveur = ce PC)."""
    import os
    import subprocess

    text = str(folder_text or "").strip()
    if not text:
        session.log_lines.append(tr(session.language, "open.none"))
        return
    folder = Path(text)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(folder))  # explorateur Windows
        elif os.uname().sysname == "Darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        session.log_lines.append(tr(session.language, "open.ok", folder=folder))
    except Exception as exc:
        session.log_lines.append(
            tr(session.language, "open.fail", folder=folder, err=exc))


def _on_download(session: GuiSession, preset_label: str) -> tuple[object, object]:
    """Télécharge le preset HF sélectionné dans models/ puis actualise la liste."""
    lang = session.language
    entry = next((p for p in PRESET_DOWNLOADS if p[0] == preset_label), None)
    if entry is None:
        return gr.update(), gr.update(value=tr(lang, "dl.unknown"))
    _, repo_id, filename = entry
    session.models_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    event = threading.Event()

    def worker() -> None:
        try:
            download_file(repo_id, filename, session.models_dir,
                          log=messages.append)
        except Exception as exc:  # réseau, espace disque…
            messages.append(tr(lang, "dl.fail", err=exc))
        finally:
            event.set()

    threading.Thread(target=worker, daemon=True).start()
    event.wait()  # Gradio bloque proprement ce handler ; l'UI reste fluide ailleurs.
    choices = session.refresh_models()
    target_label = next((l for l in choices
                         if session.models_by_label[l].path.name == filename), choices[0])
    last = messages[-1] if messages else "OK"
    return (gr.update(choices=choices, value=target_label),
            gr.update(value=tr(lang, "dl.status", msg=last)))


# --------------------------------------------------------------------------- #
# Paramètres persistants                                                       #
# --------------------------------------------------------------------------- #
def _persist_settings(*values: Any) -> None:
    """Sauvegarde automatique des paramètres à chaque modification (callback).

    Les radios soumettent leurs **valeurs canoniques** — le JSON est donc
    indépendant de la langue d'affichage.
    """
    keys = SETTING_KEYS
    data = dict(zip(keys, values))
    # model_choice contient l'étiquette ; on ne garde que le nom du fichier.
    label = data["model_choice"] or ""
    if "·" in label:
        label = label.split("·")[0].strip()
    data["model_choice"] = label
    data["language"] = load_settings().language
    save_settings(Settings.from_dict(data))


# --------------------------------------------------------------------------- #
# Démarrage / contrôle du lot                                                  #
# --------------------------------------------------------------------------- #
def _build_job(session: GuiSession, *values: Any) -> JobConfig:
    """Construit le JobConfig depuis les widgets et lance le BatchRunner."""
    _persist_settings(*values)
    (inp, out, model_label, scale_choice, scale_custom, out_fmt, quality,
     add_suffix, suffix, conflict, tiling, tile_size, tile_overlap, seed,
     precision, low_vram, clear_cache, parallel_io, resume, backend_choice,
     cfg_scale, cfg_rescale, cond_noise_scale, color_fix, recursive, memory_profile, retry_failed) = values
    lang = session.language

    inp_dir = Path(str(inp).strip()) if str(inp).strip() else None
    if inp_dir is None or not inp_dir.exists():
        raise ValueError(tr(lang, "err.no_input"))
    out_text = str(out).strip() or str(inp_dir / "upscaled")
    out_dir = Path(out_text)

    session.refresh_models()
    model = session.models_by_label.get(str(model_label))
    if model is None:
        raise ValueError(tr(lang, "err.no_model"))

    # Un ancien settings.json peut contenir « demo » (backend supprimé) : on
    # retombe sur « auto » plutôt que de planter sur un backend fantôme.
    backend_name = str(backend_choice)
    if backend_name not in {"auto", "official"}:
        backend_name = "auto"

    job = JobConfig(
        input_dir=inp_dir,
        output_dir=out_dir,
        model=model,
        scale=_scale_value(str(scale_choice), float(scale_custom)),
        output_format=OutputFormat.from_label(str(out_fmt)),
        quality=int(quality),
        conflict_policy=ConflictPolicy.from_label(str(conflict)),
        add_suffix=bool(add_suffix),
        suffix=str(suffix or "_upscaled"),
        seed=int(seed),
        precision=Precision(str(precision)),
        low_vram=bool(low_vram),
        clear_cache=bool(clear_cache),
        parallel_io=bool(parallel_io),
        resume=bool(resume),
        tiling=TilingConfig(enabled=bool(tiling), tile_size=int(tile_size),
                            overlap=int(tile_overlap)),
        backend_name=backend_name,
        cfg_scale=float(cfg_scale),
        cfg_rescale=float(cfg_rescale),
        cond_noise_scale=float(cond_noise_scale),
        color_fix=bool(color_fix), recursive=bool(recursive),
        memory_profile=str(memory_profile), retry_failed=bool(retry_failed),
    )

    return job


def _launch_job(session: GuiSession, job: JobConfig) -> None:
    session.log_lines.clear()
    session.progress = (0, 0, None)
    session.tile = (0, 0)
    session.preview = None
    session.bus = EventBus()
    session.runner = BatchRunner(job, session.bus)
    session.state = RunnerState.LOADING
    session.runner.start()


def _on_start(session: GuiSession, *values: Any) -> object:
    with session.lock:
        if session.busy:
            return gr.update(value=tr(session.language, "err.already"))
        try:
            job = _build_job(session, *values)
            preflight(job)
            _launch_job(session, job)
            return gr.update(value=tr(session.language, "start.started"))
        except (ValueError, OSError) as exc:
            return gr.update(value=str(exc))


def _safe_call(session: GuiSession, action: str) -> None:
    """Pause / Reprendre / Annuler — jamais d'exception vers l'UI."""
    if action == "cancel":
        _clear_queue(session)
    runner = session.runner
    if runner is None:
        return
    try:
        getattr(runner, action)()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Rafraîchissement périodique (Timer)                                          #
# --------------------------------------------------------------------------- #
def _on_tick(session: GuiSession):  # noqa: C901
    """Vidange le bus d'événements et met à jour les widgets concernés."""
    lang = session.language
    runner = session.runner
    if runner is not None:
        session.state = runner.state
        if runner.state is RunnerState.CANCELLING and not runner.is_alive:
            session.state = RunnerState.IDLE

    if session.bus is not None:
        for event in session.bus.drain():
            kind = event.kind
            data = event.data
            if kind == "log":
                session.log_lines.append(data["message"])
            elif kind == "progress":
                session.progress = (data["done"], data["total"], data["eta"])
            elif kind == "preview":
                session.preview = (data["before"], data["after"])
            elif kind == "state":
                session.state = data["state"]
            elif kind == "tile":
                session.tile = (data["index"], data["total"])
            elif kind == "result":
                session.latest_result = data
            elif kind == "finished":
                stats = data["stats"]
                if session.state is RunnerState.CANCELLING:
                    session.state = RunnerState.IDLE
                session.log_lines.append(tr(
                    lang, "summary", ok=stats.succeeded, fail=stats.failed,
                    skip=stats.skipped, avg=f"{stats.avg_seconds_per_image:.2f}",
                ))

    with session.lock:
        if (session.pending_jobs and session.runner and session.runner.stats
                and not session.runner.is_alive):
            _launch_job(session, session.pending_jobs.popleft())

    done, total, remaining = session.progress
    preview_value = session.preview if session.preview else gr.update()
    start_b, pause_b, resume_b, cancel_b = _buttons_for(session.state)

    counter_text = tr(lang, "counter.images", done=done, total=total) if total else ""
    tile_index, tile_total = session.tile  # affiché seulement pendant le tuilage
    if tile_total > 0 and session.state is RunnerState.RUNNING:
        counter_text += tr(lang, "counter.tile", i=tile_index, n=tile_total)

    if session.state is RunnerState.RUNNING:
        eta_text = (tr(lang, "eta.fmt", d=_fmt_duration(remaining))
                    if remaining else tr(lang, "eta.pending"))
    else:
        eta_text = ""

    return (
        "\n".join(session.log_lines),
        _progress_html(done, total),
        eta_text,
        counter_text,
        tr(lang, f"status.{session.state.value}"),
        preview_value,
        start_b, pause_b, resume_b, cancel_b,
    )


def launch(models_dir: Path | None = None, **launch_kwargs: Any) -> None:
    """Construit et lance l'application Gradio (file d'attente activée)."""
    demo = build_app(models_dir)
    demo.queue()
    kwargs = {"server_name": "127.0.0.1", "inbrowser": True, **launch_kwargs}
    demo.launch(**kwargs)


def _check_job(session, *values):
    try:
        job = _build_job(session, *values)
        summary = preflight(job)
        files = [str(p) for p in scan_inputs(job)]
        return json.dumps(summary, ensure_ascii=False, indent=2), gr.update(choices=files, value=files[0])
    except (OSError, ValueError) as exc:
        return str(exc), gr.update(choices=[], value=None)


def _queue_text(session):
    return '\n'.join(f'{i}. {j.input_dir} → {j.output_dir} (×{j.scale:g})'
                     for i, j in enumerate(session.pending_jobs, 1)) or 'File vide / Empty queue'


def _enqueue_job(session, *values):
    try:
        job = _build_job(session, *values)
        preflight(job)
        with session.lock:
            session.pending_jobs.append(job)
        return _queue_text(session)
    except (OSError, ValueError) as exc:
        return str(exc)


def _start_queue(session):
    with session.lock:
        if not session.busy and session.pending_jobs:
            _launch_job(session, session.pending_jobs.popleft())
    return _queue_text(session)


def _clear_queue(session):
    with session.lock:
        session.pending_jobs.clear()
    return _queue_text(session)


def _export_preset(*values):
    data = dict(zip(SETTING_KEYS, values))
    for key in ('input_dir', 'output_dir', 'retry_failed'):
        data.pop(key, None)
    if data.get('model_choice'):
        data['model_choice'] = data['model_choice'].split('·')[0].strip()
    path = Path(tempfile.mkdtemp(prefix='seedvr2-preset-')) / 'preset.json'
    atomic_json(path, {'version': 1, 'settings': data})
    return str(path)


def _import_preset(session, filename):
    try:
        path = Path(filename)
        if path.stat().st_size > 65536:
            raise ValueError('Préréglage trop volumineux / Preset too large')
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('version') != 1 or not isinstance(payload.get('settings'), dict):
            raise ValueError('Format de préréglage inconnu / Unknown preset format')
        data = payload['settings']
        defaults = Settings()
        for key, value in data.items():
            if key in SETTING_KEYS and not isinstance(value, type(getattr(defaults, key))):
                # JSON may encode an integer for a float-valued setting.
                if not (isinstance(getattr(defaults, key), float) and type(value) in (int, float)):
                    raise ValueError(f'Type invalide : {key}')
        settings = Settings.from_dict(data)
        choices = session.refresh_models()
        result = []
        for key in SETTING_KEYS:
            if key in ('input_dir', 'output_dir', 'retry_failed') or key not in data:
                result.append(gr.update())
            elif key == 'model_choice':
                label = next((v for v in choices if session.models_by_label[v].path.name == settings.model_choice), None)
                result.append(gr.update(value=label) if label else gr.update())
            else:
                result.append(gr.update(value=getattr(settings, key)))
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc


def _test_crop(session, source, x, y, w, h, *values):
    with session.lock:
        if session.busy:
            return gr.update(value=tr(session.language, 'err.already'))
        try:
            job = _build_job(session, *values)
            files = scan_inputs(job)
            selected = Path(source).resolve() if source else files[0]
            if selected not in files:
                raise ValueError('Image hors sélection / Image outside selection')
            job = dataclasses.replace(job, input_files=(selected,), retry_failed=False, resume=False,
                crop=(int(x), int(y), int(w), int(h)),
                output_dir=job.output_dir / '.seedvr2_tests' / uuid.uuid4().hex[:12])
            preflight(job)
            _launch_job(session, job)
            return gr.update(value='Essai en cours / Testing crop…')
        except (OSError, ValueError, IndexError) as exc:
            return gr.update(value=str(exc))


def _inspect_result(session, x, y, zoom):
    from PIL import Image
    from .images import load_image
    if not session.latest_result:
        raise gr.Error('Aucun résultat disponible / No result yet')
    entry = session.latest_result
    try:
        before = load_image(Path(entry['source'])).image
        if entry.get('crop'):
            cx, cy, cw, ch = entry['crop']
            before = before.crop((cx, cy, cx + cw, cy + ch))
        with Image.open(entry['destination']) as after:
            x = max(0, min(int(x), after.width - 1))
            y = max(0, min(int(y), after.height - 1))
            side = 512 // int(zoom)
            box = (x, y, min(x + side, after.width), min(y + side, after.height))
            crop_after = after.crop(box)
            ratio_x, ratio_y = before.width / after.width, before.height / after.height
            # Resample only the inspection window, not the entire original.
            crop_before = before.transform(crop_after.size, Image.Transform.EXTENT,
                (box[0] * ratio_x, box[1] * ratio_y, box[2] * ratio_x, box[3] * ratio_y),
                resample=Image.Resampling.BICUBIC)
        target = (crop_after.width * int(zoom), crop_after.height * int(zoom))
        def pad(image):
            canvas = Image.new('RGB', (512, 512), '#222222')
            canvas.paste(image.resize(target, Image.Resampling.NEAREST), (0, 0))
            return canvas
        return pad(crop_before), pad(crop_after)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc


def _history(folder):
    if not folder:
        return []
    entries = sorted((Path(folder) / '.seedvr2_history').glob('*.json'), reverse=True)[:20]
    result = []
    for path in entries:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            result.append({'date': path.stem, 'state': payload['state'],
                           'stats': payload['stats'], 'config': payload['config']})
        except (OSError, ValueError, KeyError):
            continue
    return result
