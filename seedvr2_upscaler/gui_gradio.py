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
                    choices=list(SUPPORTED_LANGUAGES.items()),
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
                          low_vram, clear_cache, parallel_io, resume, backend]
        for component in setting_inputs:
            component.change(_persist_settings, inputs=setting_inputs)

        drop.upload(lambda files: _on_drop(session, files), inputs=drop,
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
        return gr.update()
    first = Path(files[0])
    folder = first.parent
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
    keys = ["input_dir", "output_dir", "model_choice", "scale_choice", "scale_custom",
            "output_format", "quality", "add_suffix", "suffix", "conflict_policy",
            "tiling", "tile_size", "tile_overlap", "seed", "precision", "low_vram",
            "clear_cache", "parallel_io", "resume", "backend"]
    data = dict(zip(keys, values))
    # model_choice contient l'étiquette ; on ne garde que le nom du fichier.
    label = data["model_choice"] or ""
    if "·" in label:
        label = label.split("·")[0].strip()
    data["model_choice"] = label
    save_settings(Settings.from_dict(data))


# --------------------------------------------------------------------------- #
# Démarrage / contrôle du lot                                                  #
# --------------------------------------------------------------------------- #
def _on_start(session: GuiSession, *values: Any) -> object:  # noqa: C901
    """Construit le JobConfig depuis les widgets et lance le BatchRunner."""
    _persist_settings(*values)
    (inp, out, model_label, scale_choice, scale_custom, out_fmt, quality,
     add_suffix, suffix, conflict, tiling, tile_size, tile_overlap, seed,
     precision, low_vram, clear_cache, parallel_io, resume, backend_choice) = values
    lang = session.language

    if session.busy:
        return gr.update(value=tr(lang, "err.already"))

    inp_dir = Path(str(inp).strip()) if str(inp).strip() else None
    if inp_dir is None or not inp_dir.exists():
        return gr.update(value=tr(lang, "err.no_input"))
    out_text = str(out).strip() or str(inp_dir / "upscaled")
    out_dir = Path(out_text)

    session.refresh_models()
    model = session.models_by_label.get(str(model_label))
    if model is None:
        return gr.update(value=tr(lang, "err.no_model"))

    session.log_lines.clear()
    session.progress = (0, 0, None)
    session.tile = (0, 0)
    session.preview = None

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
    )

    session.bus = EventBus()
    session.runner = BatchRunner(job, session.bus)
    session.state = RunnerState.LOADING
    session.runner.start()
    return gr.update(value=tr(lang, "start.started"))


def _safe_call(session: GuiSession, action: str) -> None:
    """Pause / Reprendre / Annuler — jamais d'exception vers l'UI."""
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
            elif kind == "finished":
                stats = data["stats"]
                session.log_lines.append(tr(
                    lang, "summary", ok=stats.succeeded, fail=stats.failed,
                    skip=stats.skipped, avg=f"{stats.avg_seconds_per_image:.2f}",
                ))

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
