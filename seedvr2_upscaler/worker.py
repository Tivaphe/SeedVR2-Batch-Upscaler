"""Moteur de traitement par lot.

Tourne dans un **thread dédié** pour ne jamais bloquer l'interface :
    - charge le modèle une seule fois, puis enchaîne toutes les images ;
    - Pause / Reprise / Annulation (coopératifs, réactifs aussi pendant les tuiles) ;
    - estimation du temps restant (moyenne glissante des durées par image) ;
    - journalisation horodatée vers l'UI **et** vers ``log.txt`` ;
    - sauvegarde des écritures disque sur un pool de threads (I/O parallèle) ;
    - état de reprise persistant (``<output>/.seedvr2_resume.json``) ;
    - rapport final ``report.txt`` (temps total, moyenne, erreurs) ;
    - repli automatique vers le tuilage en cas d'erreur VRAM (OOM).
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backend import BackendOptions, BackendUnavailable, create_backend
from .backend.base import UpscaleBackend
from .constants import (
    DEFAULT_MODELS_DIR,
    IMAGE_EXTENSIONS,
    LOG_FILENAME,
    REPORT_FILENAME,
    RESUME_FILENAME,
)
from .images import save_image
from .models import (
    BatchStats,
    ConflictPolicy,
    ImageOutcome,
    JobConfig,
    RunnerState,
    format_for_source,
)
from .naming import apply_suffix, resolve_output_path
_ETA_WINDOW = 5            # moyenne glissante sur les N dernières images
_MAX_PARALLEL_SAVES = 2    # files d'écriture simultanées max


# --------------------------------------------------------------------------- #
# Bus d'événements (worker -> UI, thread-safe)                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BusEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """File thread-safe de messages destinés à l'interface graphique."""

    def __init__(self) -> None:
        self._q: "queue.Queue[BusEvent]" = queue.Queue()

    def put(self, kind: str, **data: Any) -> None:
        self._q.put(BusEvent(kind, data))

    def drain(self) -> list[BusEvent]:
        """Récupère tous les événements en attente (appelé par l'UI)."""
        events: list[BusEvent] = []
        while True:
            try:
                events.append(self._q.get_nowait())
            except queue.Empty:
                return events

    # Raccourcis typés -----------------------------------------------------
    def log(self, message: str) -> None:
        self.put("log", message=message)

    def progress(self, done: int, total: int, eta_seconds: float | None) -> None:
        self.put("progress", done=done, total=total, eta=eta_seconds)

    def preview(self, before: Any, after: Any) -> None:
        self.put("preview", before=before, after=after)

    def state(self, state: RunnerState) -> None:
        self.put("state", state=state)

    def finished(self, stats: BatchStats) -> None:
        self.put("finished", stats=stats)


# --------------------------------------------------------------------------- #
# Moteur de lot                                                                #
# --------------------------------------------------------------------------- #
class BatchRunner:
    """Exécute un ``JobConfig`` dans un thread, par notifications EventBus."""

    def __init__(
        self,
        job: JobConfig,
        bus: EventBus | None = None,
        backend_factory: Callable[..., UpscaleBackend] = create_backend,
    ) -> None:
        self.job = job
        self.bus = bus or EventBus()
        self._backend_factory = backend_factory

        self._pause = threading.Event()   # levé = en marche
        self._pause.set()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = RunnerState.IDLE

        self._log_lines: list[str] = []
        self._durations: list[float] = []
        self._backend: UpscaleBackend | None = None

    # ------------------------------------------------------------- API UI #
    @property
    def state(self) -> RunnerState:
        return self._state

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Le traitement est déjà démarré.")
        self._thread = threading.Thread(target=self._run_guarded, daemon=True, name="batch-runner")
        self._thread.start()

    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.set()  # débloque un éventuel wait pour sortir proprement

    # ---------------------------------------------------------- internes #
    def _set_state(self, state: RunnerState) -> None:
        self._state = state
        self.bus.state(state)

    def _log(self, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        self._log_lines.append(line)
        self.bus.log(line)

    def _wait_if_paused(self) -> None:
        while not self._pause.is_set() and not self._cancel.is_set():
            if self._state is RunnerState.RUNNING:
                self._set_state(RunnerState.PAUSED)
            time.sleep(0.1)
        if self._state is RunnerState.PAUSED and not self._cancel.is_set():
            self._set_state(RunnerState.RUNNING)

    def _gate(self) -> bool:
        """Callback passé au backend : pause coopérative + test d'annulation."""
        self._wait_if_paused()
        return self._cancel.is_set()

    # ----------------------------------------------------------- exécution
    def _run_guarded(self) -> None:
        stats = BatchStats(started_at=time.time())
        self._inhibit_sleep()  # un lot peut durer des heures : pas de veille !
        try:
            self._run(stats)
        except Exception as exc:  # erreur inattendue du moteur lui-même
            import traceback

            details = "".join(traceback.format_exception(exc))
            self._log(f"ERREUR FATALE : {exc!r}\n{details}")
            stats.errors.append(f"fatal: {exc!r}")
            self._set_state(RunnerState.FAILED)
        finally:
            self._restore_sleep()
            self._finalize(stats)
            self.bus.finished(stats)

    def _run(self, stats: BatchStats) -> None:  # noqa: C901, PLR0912, PLR0915
        job = self.job
        self._set_state(RunnerState.LOADING)

        options = BackendOptions(
            precision=job.precision,
            low_vram=job.low_vram,
            clear_cache=job.clear_cache,
            tiling=job.tiling,
            cfg_scale=job.cfg_scale,
            cfg_rescale=job.cfg_rescale,
            cond_noise_scale=job.cond_noise_scale,
            color_fix=job.color_fix,
        )
        models_dir = job.model.path.parent if job.model.path.name else DEFAULT_MODELS_DIR
        self._backend = self._backend_factory(job.backend_name, options, models_dir)
        try:
            self._backend.load(job.model, log=self._log)
        except BackendUnavailable as exc:
            self._log(str(exc))
            stats.errors.append(str(exc))
            self._set_state(RunnerState.FAILED)
            return

        files = self._scan_inputs(job.input_dir)
        stats.total = len(files)
        if not files:
            self._log(f"Aucune image prise en charge dans {job.input_dir}.")
            self._set_state(RunnerState.FINISHED)
            return
        self._log(f"{len(files)} image(s) détectée(s) dans {job.input_dir}.")

        job.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_disk_space(files)
        done = self._load_resume(files)
        todo = [f for f in files if f.name not in done]
        if done:
            self._log(f"Reprise : {len(files) - len(todo)} image(s) déjà traitée(s) ignorée(s).")

        io_pool = ThreadPoolExecutor(max_workers=2) if job.parallel_io else None
        pending_saves: list[Future] = []
        reserved: set[str] = set()
        self._set_state(RunnerState.RUNNING)
        cancelled = False

        try:
            for index, src in enumerate(todo):
                if self._cancel.is_set():
                    cancelled = True
                    break
                self._wait_if_paused()
                if self._cancel.is_set():
                    cancelled = True
                    break

                outcome = self._process_one(src, index, reserved, pending_saves, io_pool)
                if outcome is None:  # annulation demandée pendant une tuile
                    cancelled = True
                    break

                self._register_outcome(outcome, stats, done)
                self._save_resume(done)
                self._emit_progress(done_count=stats.succeeded + stats.failed + stats.skipped,
                                    total_count=len(todo))
                if job.clear_cache:
                    self._clear_gpu_cache()

            if self._cancel.is_set():
                cancelled = True
        finally:
            for future in pending_saves:  # écritures en vol : on les termine
                try:
                    future.result()
                except Exception as exc:
                    self._log(f"Erreur d'écriture disque : {exc}")
                    stats.errors.append(f"io: {exc}")
            if io_pool is not None:
                io_pool.shutdown(wait=True)
            try:
                self._backend.unload()
            except Exception as exc:  # pragma: no cover - best effort
                self._log(f"Avertissement unload : {exc}")

        self._set_state(RunnerState.FINISHED if not cancelled else RunnerState.CANCELLING)
        self._log("Traitement annulé par l'utilisateur." if cancelled
                  else "Traitement terminé.")

    # ----------------------------------------------------------- une image
    def _process_one(
        self,
        src: Path,
        index: int,
        reserved: set[str],
        pending_saves: list[Future],
        io_pool: ThreadPoolExecutor | None,
    ) -> ImageOutcome | None:
        """Charge, u upscale, sauvegarde. None = annulation en cours de tuiles."""
        job = self.job
        started = time.time()

        # Format effectif : « Comme l'entrée » conserve extension ET format.
        out_format, out_ext = format_for_source(src, job.output_format)
        stem = apply_suffix(src.stem, job.suffix, job.add_suffix)
        dest = resolve_output_path(job.output_dir, f"{stem}{out_ext}",
                                   job.conflict_policy, _reserved=reserved)
        if dest is None:
            self._log(f"{src.name} : ignoré (fichier de sortie déjà présent).")
            return ImageOutcome(source=src, destination=None, ok=True, skipped=True)

        self._log(f"{src.name} → {dest.name}")
        try:
            from .images import load_image

            loaded = load_image(src)
            seed_img = job.seed + index  # graine déterministe par image
            result = self._backend.upscale(
                loaded.image, job.scale, seed_img, log=self._log,
                should_abort=self._gate, tile_cb=self._tile_cb,
            )
        except InterruptedError:
            return None
        except Exception as exc:
            if self._is_oom(exc) and not self._backend.options.tiling.enabled:
                self._log(f"VRAM insuffisante sur {src.name} — nouvel essai avec tuilage.")
                self._enable_tiling_fallback()
                self._clear_gpu_cache()
                try:
                    from .images import load_image

                    loaded = load_image(src)
                    result = self._backend.upscale(loaded.image, job.scale,
                                                   job.seed + index, log=self._log,
                                                   should_abort=self._gate,
                                                   tile_cb=self._tile_cb)
                except Exception as exc2:
                    return self._fail(src, exc2, started)
            else:
                return self._fail(src, exc, started)

        in_w, in_h = loaded.image.size
        out_w, out_h = result.size
        self._log(f"  dimensions : {in_w}×{in_h} → {out_w}×{out_h} "
                  f"(×{job.scale:g}) | {out_w * out_h / 1e6:.1f} Mpx")

        before_thumb = self._thumbnail(loaded.image)
        after_thumb = self._thumbnail(result)
        self.bus.preview(before_thumb, after_thumb)

        self._enqueue_save(pending_saves, io_pool, result, dest, loaded, out_format)
        return ImageOutcome(source=src, destination=dest, ok=True,
                            seconds=time.time() - started)

    def _fail(self, src: Path, exc: Exception, started: float) -> ImageOutcome:
        import traceback

        # traceback complet : indispensable pour diagnostiquer les erreurs
        # du pipeline (affiché dans le journal ET log.txt).
        details = "".join(traceback.format_exception(exc))
        self._log(f"ERREUR {src.name} : {exc}\n{details}— on continue le lot.")
        return ImageOutcome(source=src, destination=None, ok=False,
                            error=str(exc), seconds=time.time() - started)

    def _tile_cb(self, index: int, total: int) -> None:
        """Relais vers l'UI de la sous-progression « tuile i/N » de l'image courante."""
        self.bus.put("tile", index=index, total=total)

    def _check_disk_space(self, files: list[Path]) -> None:
        """Avertit (sans bloquer) si l'espace semble insuffisant pour le lot.

        Estimation volontairement grossière : octets d'entrée × échelle² × 0,8.
        Le but est d'éviter un disque plein à la 150ᵉ image d'un lot nocturne.
        """
        try:
            in_bytes = sum(f.stat().st_size for f in files)
            estimate = in_bytes * self.job.scale**2 * 0.8
            free = shutil.disk_usage(self.job.output_dir).free
            go = 1024**3
            if free < estimate * 1.2:
                self._log(f"⚠️ Espace disque possiblement insuffisant : ~{estimate / go:.1f} Go "
                          f"estimés contre {free / go:.1f} Go libres sur la destination.")
            else:
                self._log(f"Espace disque OK : ~{estimate / go:.1f} Go estimés, "
                          f"{free / go:.1f} Go libres.")
        except Exception:
            pass  # confort seulement, jamais bloquant

    # -------------------------------------------------- veille Windows ---- #
    @staticmethod
    def _set_execution_state(flags: int) -> None:
        """SetThreadExecutionState (Win32) — no-op hors Windows."""
        if os.name != "nt":
            return
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except Exception:
            pass

    @classmethod
    def _inhibit_sleep(cls) -> None:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED : écran libre, système éveillé.
        cls._set_execution_state(0x80000000 | 0x00000001)

    @classmethod
    def _restore_sleep(cls) -> None:
        cls._set_execution_state(0x80000000)  # ES_CONTINUOUS seul = restauration

    @staticmethod
    def _notify_done() -> None:
        """Petit bip de fin de lot (Windows uniquement, jamais bloquant)."""
        try:
            import winsound

            winsound.Beep(880, 180)
            winsound.Beep(660, 260)
        except Exception:
            pass

    def _enqueue_save(
        self,
        pending: list[Future],
        io_pool: ThreadPoolExecutor | None,
        result,
        dest: Path,
        loaded,
        out_format,
    ) -> None:
        """Écriture synchrone ou sur le pool d'I/O (borne le nombre de vols)."""
        job = self.job
        if io_pool is None:
            save_image(result, dest, out_format, job.quality,
                       loaded.exif_bytes, loaded.icc_profile)
            return
        while len(pending) >= _MAX_PARALLEL_SAVES:  # borne la RAM retenue
            pending.pop(0).result()
        pending.append(io_pool.submit(
            save_image, result, dest, out_format, job.quality,
            loaded.exif_bytes, loaded.icc_profile,
        ))

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _scan_inputs(input_dir: Path) -> list[Path]:
        if not input_dir.exists():
            return []
        return sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _thumbnail(image, max_side: int = 768):
        thumb = image
        if max(thumb.size) > max_side:
            thumb = thumb.copy()
            thumb.thumbnail((max_side, max_side))
        return thumb

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text

    def _enable_tiling_fallback(self) -> None:
        backend = self._backend
        backend.options = dataclasses.replace(
            backend.options,
            tiling=dataclasses.replace(backend.options.tiling, enabled=True),
        )
        self._log("Tuilage activé automatiquement pour la suite du lot.")

    @staticmethod
    def _clear_gpu_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------- progression & stats
    def _register_outcome(self, outcome: ImageOutcome, stats: BatchStats, done: list[str]) -> None:
        if outcome.ok and not outcome.skipped:
            stats.succeeded += 1
            self._durations.append(outcome.seconds)
            done.append(outcome.source.name)
        elif outcome.skipped:
            stats.skipped += 1
            done.append(outcome.source.name)
        else:
            stats.failed += 1
            stats.errors.append(f"{outcome.source.name}: {outcome.error}")
            # Les échecs ne sont pas marqués « done » : ils seront retentés
            # à la prochaine reprise.

    def _emit_progress(self, done_count: int, total_count: int) -> None:
        recent = self._durations[-_ETA_WINDOW:]
        avg = sum(recent) / len(recent) if recent else None
        remaining = max(0, total_count - done_count)
        eta = avg * remaining if avg else None
        self.bus.progress(done_count, total_count, eta)

    # -------------------------------------------------------------- reprise
    def _resume_path(self) -> Path:
        return self.job.output_dir / RESUME_FILENAME

    def _config_fingerprint(self) -> str:
        job = self.job
        payload = (
            f"{job.scale}|{job.model.path.name}|{job.output_format.value}|"
            f"{job.quality}|{job.add_suffix}:{job.suffix}"
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def _load_resume(self, files: list[Path]) -> list[str]:
        path = self._resume_path()
        if not (self.job.resume and path.exists()):
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self._config_fingerprint():
                self._log("État de reprise ignoré : configuration différente.")
                return []
            names = {p.name for p in files}
            return [n for n in payload.get("done", []) if n in names]
        except Exception as exc:
            self._log(f"État de reprise illisible ({exc}), nouveau départ.")
            return []

    def _save_resume(self, done: list[str]) -> None:
        if not self.job.resume:
            return
        try:
            self._resume_path().write_text(
                json.dumps({"fingerprint": self._config_fingerprint(), "done": done},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass  # la reprise est un confort, jamais un point de blocage

    # ------------------------------------------------------------ finalisation
    def _finalize(self, stats: BatchStats) -> None:
        stats.elapsed_seconds = time.time() - stats.started_at if stats.started_at else 0.0
        if self._durations:
            stats.avg_seconds_per_image = sum(self._durations) / len(self._durations)
        try:
            self.job.output_dir.mkdir(parents=True, exist_ok=True)
            self._write_log_file()
            if stats.total:
                self._write_report(stats)
                self._notify_done()
            # Lot achevé sans erreur bloquante : l'état de reprise n'a plus lieu d'être.
            if self._state is RunnerState.FINISHED:
                self._resume_path().unlink(missing_ok=True)
        except Exception as exc:
            self.bus.log(f"Impossible d'écrire log/rapport : {exc}")

    def _write_log_file(self) -> None:
        """Ajoute la session à log.txt (l'historique des lots précédents est gardé)."""
        target = self.job.output_dir / LOG_FILENAME
        header = (
            f"# Journal SeedVR2 Batch Upscaler — {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"# Modèle : {self.job.model.path.name} | Échelle : x{self.job.scale:g}\n"
        )
        with target.open("a", encoding="utf-8") as handle:
            handle.write(header + "\n".join(self._log_lines) + "\n\n")

    def _write_report(self, stats: BatchStats) -> None:
        job = self.job
        lines = [
            "=" * 60,
            "RAPPORT DE TRAITEMENT — SeedVR2 Batch Upscaler",
            "=" * 60,
            f"Date de fin        : {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"État               : {self._describe_final_state()}",
            f"Dossier d'entrée   : {job.input_dir}",
            f"Dossier de sortie  : {job.output_dir}",
            f"Modèle             : {job.model.path.name}",
            f"Échelle            : x{job.scale:g}",
            "-" * 60,
            f"Images détectées   : {stats.total}",
            f"Images réussies    : {stats.succeeded}",
            f"Images ignorées    : {stats.skipped} (conflit : ignorer / reprise)",
            f"Images en échec    : {stats.failed}",
            "-" * 60,
            f"Temps total        : {_fmt_duration(stats.elapsed_seconds)}",
            f"Temps moyen/image  : {stats.avg_seconds_per_image:.2f} s",
            f"Vitesse moyenne    : {_fmt_speed(stats)}",
            "-" * 60,
        ]
        if stats.errors:
            lines.append(f"Erreurs ({len(stats.errors)}) :")
            lines += [f"  - {err}" for err in stats.errors]
            lines.append("-" * 60)
        else:
            lines.append("Aucune erreur.")
        (job.output_dir / REPORT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.bus.log(f"Rapport écrit : {job.output_dir / REPORT_FILENAME}")

    def _describe_final_state(self) -> str:
        return {
            RunnerState.FINISHED: "terminé",
            RunnerState.CANCELLING: "annulé par l'utilisateur",
            RunnerState.FAILED: "échec (voir log.txt)",
        }.get(self._state, self._state.value)


# --------------------------------------------------------------------------- #
def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _fmt_speed(stats: BatchStats) -> str:
    if stats.succeeded and stats.avg_seconds_per_image > 0:
        return f"{60.0 / stats.avg_seconds_per_image:.1f} images/min"
    return "n/d"


def fmt_eta(eta_seconds: float | None) -> str:
    """Rendu texte du temps restant estimé (exposé pour l'UI)."""
    if not eta_seconds:
        return "estimation en cours…"
    return f"~{_fmt_duration(eta_seconds)} restantes"
