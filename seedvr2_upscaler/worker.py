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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backend import BackendOptions, create_backend
from .backend.base import UpscaleBackend
from .constants import (
    IMAGE_EXTENSIONS,
    LOG_FILENAME,
    REPORT_FILENAME,
    RESUME_FILENAME,
)
from .images import save_image
from .jobs import atomic_json, preflight, scan_inputs, serialize_job
from .memory import apply_auto_profile, remember_profile
from .models import (
    BatchStats,
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
        self._q: "queue.Queue[BusEvent]" = queue.Queue(maxsize=4096)
        self._preview_lock = threading.Lock()
        self._latest_preview: BusEvent | None = None

    def put(self, kind: str, **data: Any) -> None:
        event = BusEvent(kind, data)
        if kind == "preview":
            with self._preview_lock:
                self._latest_preview = event
            return
        # A closed browser must not grow RAM indefinitely or block inference.
        # The complete journal remains on disk; the UI only needs recent events.
        while True:
            try:
                self._q.put_nowait(event)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def drain(self) -> list[BusEvent]:
        """Récupère tous les événements en attente (appelé par l'UI)."""
        events: list[BusEvent] = []
        while True:
            try:
                events.append(self._q.get_nowait())
            except queue.Empty:
                with self._preview_lock:
                    if self._latest_preview:
                        events.append(self._latest_preview)
                        self._latest_preview = None
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

        self._log_lines: deque[str] = deque(maxlen=2000)
        self._durations: list[float] = []
        self._backend: UpscaleBackend | None = None
        self.stats: BatchStats | None = None
        self._records: dict[str, dict] = {}
        self._save_futures: dict[Path, Future] = {}
        self._log_handle = None
        self._journal_error: str | None = None


    # ------------------------------------------------------------- API UI #
    @property
    def state(self) -> RunnerState:
        return self._state

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Le traitement est déjà démarré.")
        self._thread = threading.Thread(target=self._run_guarded, daemon=True, name="batch-runner")
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def exit_code(self) -> int:
        if self._cancel.is_set():
            return 130
        return 1 if self._state is RunnerState.FAILED or (self.stats and self.stats.errors) else 0

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
        if self._log_handle is not None:
            try:
                self._log_handle.write(line + "\n")
                self._log_handle.flush()
            except OSError as exc:
                self._journal_error = str(exc)


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
        self.stats = stats
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
            if self._backend is not None:
                try:
                    self._backend.unload()
                except Exception as exc:
                    self._log(f"Avertissement unload : {exc}")
            self._restore_sleep()
            try:
                self._finalize(stats)
            finally:
                try:
                    if self._log_handle:
                        self._log_handle.close()
                except OSError as exc:
                    stats.errors.append(f"journal close: {exc}")
                finally:
                    self.bus.finished(stats)

    def _run(self, stats: BatchStats) -> None:  # noqa: C901, PLR0912, PLR0915
        job = self.job
        self._set_state(RunnerState.LOADING)
        summary = preflight(job)
        self._log_handle = (job.output_dir / LOG_FILENAME).open("a", encoding="utf-8")
        self._log(f"Pré-vérification : {summary}")
        files = scan_inputs(job)
        stats.total = len(files)
        self._check_disk_space(files)
        done = self._load_resume(files)
        todo = [f for f in files if self._key(f) not in done]
        if job.retry_failed:
            todo = [f for f in todo if self._records.get(self._key(f), {}).get("status") == "failed"]
        stats.skipped = len(files) - len(todo)
        if not todo:
            self._log("Aucune image à traiter avec cette sélection/reprise.")
            self._set_state(RunnerState.FINISHED)
            return
        tiling = job.tiling
        if job.memory_profile == "eco":
            tiling = dataclasses.replace(tiling, enabled=True, tile_size=min(192, tiling.tile_size))
        options = BackendOptions(
            precision=job.precision, low_vram=job.low_vram if job.memory_profile != "fast" else False,
            clear_cache=job.clear_cache, tiling=tiling, cfg_scale=job.cfg_scale,
            cfg_rescale=job.cfg_rescale, cond_noise_scale=job.cond_noise_scale, color_fix=job.color_fix,
        )
        self._backend = self._backend_factory(job.backend_name, options, job.model.path.parent)
        self._backend.load(job.model, log=self._log)
        memory_key = apply_auto_profile(self._backend, job, self._log)
        io_pool = ThreadPoolExecutor(max_workers=2) if job.parallel_io else None
        outcomes: list[ImageOutcome] = []
        reserved: set[str] = set()
        self._set_state(RunnerState.RUNNING)

        def commit(outcome):
            future = self._save_futures.pop(outcome.destination, None)
            if future is not None:
                try:
                    future.result()
                except Exception as exc:
                    outcome.ok = False
                    outcome.error = f"Écriture disque : {exc}"
                    self._log(f"{outcome.source.name} : {outcome.error}")
            if outcome.ok and not outcome.skipped:
                self.bus.put("result", source=str(outcome.source), destination=str(outcome.destination),
                             crop=self.job.crop)
            self._register_outcome(outcome, stats, done)
            self._save_resume(done)
            self._emit_progress(stats.succeeded + stats.failed + stats.skipped, stats.total)

        try:
            for src in todo:
                if self._gate():
                    break
                outcome = self._process_one(src, reserved, io_pool)
                if outcome is None:
                    break
                outcomes.append(outcome)
                if len(outcomes) >= _MAX_PARALLEL_SAVES:
                    commit(outcomes.pop(0))
                if job.clear_cache:
                    self._clear_gpu_cache()
        finally:
            try:
                for outcome in outcomes:
                    commit(outcome)
            finally:
                if io_pool:
                    io_pool.shutdown(wait=True)
        if stats.succeeded:
            remember_profile(memory_key, self._backend.options.tiling.tile_size)
        self._set_state(RunnerState.CANCELLING if self._cancel.is_set() else RunnerState.FINISHED)
        self._log("Traitement annulé." if self._cancel.is_set() else "Traitement terminé.")

    # ----------------------------------------------------------- une image
    def _process_one(
        self,
        src: Path,
        reserved: set[str],
        io_pool: ThreadPoolExecutor | None,
    ) -> ImageOutcome | None:
        """Charge, u upscale, sauvegarde. None = annulation en cours de tuiles."""
        job = self.job
        started = time.time()

        # Format effectif : « Comme l'entrée » conserve extension ET format.
        out_format, out_ext = format_for_source(src, job.output_format)
        stem = apply_suffix(src.stem, job.suffix, job.add_suffix)
        dest = resolve_output_path(job.output_dir / src.relative_to(job.input_dir.resolve()).parent, f"{stem}{out_ext}",
                                   job.conflict_policy, _reserved=reserved)
        if dest is None:
            self._log(f"{src.name} : ignoré (fichier de sortie déjà présent).")
            return ImageOutcome(source=src, destination=None, ok=True, skipped=True)

        self._log(f"{src.name} → {dest.name}")
        try:
            from .images import load_image

            loaded = load_image(src)
            if job.crop:
                x, y, w, h = job.crop
                loaded.image = loaded.image.crop((x, y, x + w, y + h))
            seed_img = self._seed(src)
            result = self._backend.upscale(
                loaded.image, job.scale, seed_img, log=self._log,
                should_abort=self._gate, tile_cb=self._tile_cb,
            )
        except InterruptedError:
            return None
        except Exception as exc:
            if self._is_oom(exc):
                exc.__traceback__ = None  # release failed inference tensors before retry
                result = None
                while self._reduce_tiles():
                    self._clear_gpu_cache()
                    try:
                        result = self._backend.upscale(loaded.image, job.scale, self._seed(src),
                            log=self._log, should_abort=self._gate, tile_cb=self._tile_cb)
                        break
                    except InterruptedError:
                        return None
                    except Exception as retry_exc:
                        if not self._is_oom(retry_exc):
                            return self._fail(src, retry_exc, started)
                        retry_exc.__traceback__ = None
                        exc = retry_exc
                if result is None:
                    return self._fail(src, exc, started)
            else:
                return self._fail(src, exc, started)

        in_w, in_h = loaded.image.size
        out_w, out_h = result.size
        self._log(f"  dimensions : {in_w}×{in_h} → {out_w}×{out_h} "
                  f"(×{job.scale:g}) | {out_w * out_h / 1e6:.1f} Mpx")

        before_thumb = self._thumbnail(loaded.image)
        after_thumb = self._thumbnail(result)
        self.bus.preview(before_thumb, after_thumb)

        try:
            self._enqueue_save(io_pool, result, dest, loaded, out_format)
        except Exception as exc:
            return self._fail(src, exc, started)
        return ImageOutcome(source=src, destination=dest, ok=True,
                            seconds=time.time() - started, effective_tiling=self._backend.options.tiling)

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
        self._save_futures[dest] = io_pool.submit(
            save_image, result, dest, out_format, job.quality,
            loaded.exif_bytes, loaded.icc_profile,
        )

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

    def _reduce_tiles(self) -> bool:
        from .backend.base import MAX_TILE_OUT_SIDE
        tiling = self._backend.options.tiling
        effective = max(64, min(tiling.tile_size, int(MAX_TILE_OUT_SIDE / self.job.scale)))
        if tiling.enabled and effective <= 64:
            return False
        size = max(64, effective // 2) if tiling.enabled else effective
        self._backend.options = dataclasses.replace(self._backend.options,
            tiling=dataclasses.replace(tiling, enabled=True, tile_size=size,
                                       overlap=min(tiling.overlap, size // 2)))
        self._log(f"Repli mémoire : tuiles de {size} px.")
        return True

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
            done.append(self._key(outcome.source))
        elif outcome.skipped:
            stats.skipped += 1
            done.append(self._key(outcome.source))
        else:
            stats.failed += 1
            stats.errors.append(f"{outcome.source.name}: {outcome.error}")
            # Les échecs ne sont pas marqués « done » : ils seront retentés
            # à la prochaine reprise.

        key = self._key(outcome.source)
        self._records[key] = {
            "status": "skipped" if outcome.skipped else "done" if outcome.ok else "failed",
            "source": self._signature(outcome.source), "seed": self._seed(outcome.source),
            "destination": str(outcome.destination.resolve()) if outcome.destination else None,
            "output": self._signature(outcome.destination) if outcome.destination and outcome.ok else None,
            "error": outcome.error,
            "effective_tiling": dataclasses.asdict(outcome.effective_tiling) if outcome.effective_tiling else None,
        }

    def _emit_progress(self, done_count: int, total_count: int) -> None:
        recent = self._durations[-_ETA_WINDOW:]
        avg = sum(recent) / len(recent) if recent else None
        remaining = max(0, total_count - done_count)
        eta = avg * remaining if avg else None
        self.bus.progress(done_count, total_count, eta)

    # -------------------------------------------------------------- reprise
    def _resume_path(self) -> Path:
        return self.job.output_dir / RESUME_FILENAME

    def _key(self, source: Path) -> str:
        return source.resolve().relative_to(self.job.input_dir.resolve()).as_posix()

    def _seed(self, source: Path) -> int:
        digest = hashlib.sha256(self._key(source).encode()).digest()
        return (self.job.seed + int.from_bytes(digest[:4], "big")) % (2**32)

    @staticmethod
    def _signature(path: Path | None):
        if path is None or not path.is_file():
            return None
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _config_fingerprint(self) -> str:
        payload = serialize_job(self.job)
        for name in ("resume", "retry_failed", "parallel_io", "input_files", "conflict_policy"):
            payload.pop(name, None)
        payload["input_dir"] = str(self.job.input_dir.resolve())
        payload["model_signature"] = self._signature(self.job.model.path)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _load_resume(self, files: list[Path]) -> list[str]:
        path = self._resume_path()
        if not ((self.job.resume or self.job.retry_failed) and path.exists()):
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self._config_fingerprint():
                self._log("État de reprise ignoré : configuration différente ou ancien format.")
                return []
            self._records = payload.get("records", {})
            done = []
            for source in files:
                key = self._key(source)
                entry = self._records.get(key, {})
                dest = Path(entry["destination"]) if entry.get("destination") else None
                if (entry.get("status") == "done" and entry.get("source") == self._signature(source)
                        and dest and self._signature(dest) is not None
                        and entry.get("output") == self._signature(dest)):
                    done.append(key)
            return done
        except (OSError, ValueError, TypeError) as exc:
            self._log(f"Reprise illisible : {exc}")
            return []

    def _save_resume(self, done: list[str]) -> None:
        if self.job.resume or self.job.retry_failed:
            atomic_json(self._resume_path(), {"version": 2,
                "fingerprint": self._config_fingerprint(), "records": self._records})

    # ------------------------------------------------------------ finalisation
    def _finalize(self, stats: BatchStats) -> None:
        stats.elapsed_seconds = time.time() - stats.started_at if stats.started_at else 0.0
        if self._durations:
            stats.avg_seconds_per_image = sum(self._durations) / len(self._durations)
        try:
            self.job.output_dir.mkdir(parents=True, exist_ok=True)
            if not self._log_handle:
                self._write_log_file()
            if self._journal_error:
                stats.errors.append(f"journal: {self._journal_error}")
            if stats.total:
                self._write_report(stats)
                self._notify_done()
            atomic_json(self.job.output_dir / ".seedvr2_history" /
                        (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json"),
                        {"config": serialize_job(self.job), "stats": dataclasses.asdict(stats),
                         "state": self._state.value, "records": self._records})
        except Exception as exc:
            stats.errors.append(f"finalisation: {exc}")
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
