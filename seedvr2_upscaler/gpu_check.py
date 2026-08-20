"""Diagnostic GPU/CUDA centralisé — sert l'interface, le backend et check_install.

Pourquoi ce module : sans le backend « démo » (supprimé), un GPU non détecté
doit produire une **erreur claire et actionnable**, jamais un repli silencieux.
Les causes typiques d'une détection qui « disparaît » après un redémarrage :
    1. le pilote NVIDIA ne répond plus (mise à jour Windows interrompue) ;
    2. torch a été remplacé par une build **CPU-only** (pip depuis PyPI sans
       l'index CUDA — ex. réinstallation sur un Python très récent) ;
    3. pilote fraîchement installé/maj **sans redémarrage** ;
    4. variable d'environnement ``CUDA_VISIBLE_DEVICES`` qui masque le GPU.

Le module ne lève JAMAIS d'exception : tout est capturé et rapporté.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

# Commande de (ré)installation recommandée, calquée sur install.bat :
# on vise l'interpréteur du venv pour ne pas polluer le Python système.
# Index cu130 : torch 2.9-2.13 (CUDA 13.0, pilote NVIDIA >= 580 requis sous
# Windows, wheels Python 3.10 a 3.14). Pilote plus ancien ? Utilisez cu128
# (torch <= 2.11, CUDA 12.8) : https://download.pytorch.org/whl/cu128
TORCH_CUDA_INSTALL_CMD = (
    r".venv\Scripts\python.exe -m pip install torch torchvision "
    "--index-url https://download.pytorch.org/whl/cu130"
)


@dataclass
class GpuStatus:
    """Photographie de la chaîne CUDA : pilote → build torch → disponibilité."""

    driver_ok: bool | None = None      # nvidia-smi a répondu
    driver_version: str | None = None
    torch_present: bool = False
    torch_version: str | None = None
    cuda_build: str | None = None      # ex. "12.8" ; None = build CPU-only
    cuda_available: bool = False       # torch.cuda.is_available()
    device_name: str | None = None
    vram_gib: float | None = None
    bf16: bool | None = None
    problems: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.torch_present and self.cuda_available


def _probe_nvidia_smi(timeout: float = 15.0) -> tuple[bool, str | None]:
    """Interroge le pilote via nvidia-smi (sans ouvrir de fenêtre sous Windows)."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, creationflags=flags,
        )
    except Exception:
        return False, None
    if proc.returncode != 0:
        return False, None
    first = proc.stdout.strip().splitlines()
    return True, (first[0].strip() if first else None) or None


def detect_gpu(smi_timeout: float = 15.0) -> GpuStatus:
    """Sonde pilote + torch. Jamais d'exception : les échecs sont décrits."""
    status = GpuStatus()
    status.driver_ok, status.driver_version = _probe_nvidia_smi(smi_timeout)

    try:
        import torch
    except Exception:
        torch = None  # import impossible : torch absent ou cassé

    if torch is not None:
        status.torch_present = True
        status.torch_version = str(torch.__version__)
        # None pour les builds « +cpu » / PyPI sans CUDA.
        status.cuda_build = getattr(torch.version, "cuda", None)
        try:
            status.cuda_available = bool(torch.cuda.is_available())
        except Exception:
            status.cuda_available = False
        if status.cuda_available:
            try:
                index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                status.device_name = torch.cuda.get_device_name(index)
                status.vram_gib = round(props.total_memory / 1024**3, 1)
                status.bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                pass

    if not status.ok:
        _diagnose(status)
    return status


def _diagnose(status: GpuStatus) -> None:
    """Remplit causes probables et remèdes, de la chaîne la plus amont à l'aval."""
    if not status.driver_ok:
        status.problems.append(
            "Le pilote NVIDIA ne répond pas (nvidia-smi en échec) — cause fréquente : "
            "mise à jour Windows ou pilote réinstallé sans redémarrage."
        )
        status.hints.append(
            "Réinstallez/mettez à jour le pilote depuis nvidia.fr/drivers puis "
            "REDÉMARREZ le PC."
        )
    if not status.torch_present:
        status.problems.append("PyTorch n'est pas installé dans cet environnement (.venv).")
        status.hints.append(f"Installez la build CUDA :\n  {TORCH_CUDA_INSTALL_CMD}")
        return
    if status.cuda_build is None:
        status.problems.append(
            f"torch {status.torch_version} est une build CPU-only (torch.version.cuda = None) "
            "— elle a probablement remplacé la build CUDA lors d'un pip install sans index CUDA "
            "(ou via le repli PyPI de install.bat : l'index cu128 ne publie plus de build "
            "CUDA depuis torch 2.12)."
        )
        status.hints.append(
            "Remplacez-la par la build CUDA (écrase la build CPU) :\n"
            f"  {TORCH_CUDA_INSTALL_CMD}"
        )
        status.hints.append(
            "Pilote NVIDIA antérieur au branch 580 ? Utilisez alors l'index cu128 "
            "(torch <= 2.11, CUDA 12.8) : remplacez cu130 par cu128 dans la commande."
        )
        return
    if not status.cuda_available:
        status.problems.append(
            f"Le pilote répond ({status.driver_version or 'version inconnue'}) et torch "
            f"{status.torch_version} embarque CUDA {status.cuda_build}, pourtant "
            "torch.cuda.is_available() = False."
        )
        status.hints.append(
            "Dans l'ordre :\n"
            "  1) redémarrez le PC (pilote en cours d'activation ?) ;\n"
            "  2) vérifiez que la variable CUDA_VISIBLE_DEVICES est absente ou vaut « 0 » ;\n"
            "  3) mettez à jour le pilote NVIDIA (torch récent exige un pilote récent)."
        )


def cause_key(status: GpuStatus) -> str | None:
    """Clé de cause stable (« driver », « cpu_build »…) pour l'UI bilingue.

    L'interface compose ses propres phrases traduites à partir de cette clé ;
    les messages longs (français) restent utilisés côté journal/CLI.
    """
    if status.ok:
        return None
    if not status.driver_ok:
        return "driver"
    if not status.torch_present:
        return "torch_missing"
    if status.cuda_build is None:
        return "cpu_build"
    if not status.cuda_available:
        return "cuda_false"
    return "unknown"


# --------------------------------------------------------------------------- #
# Rendus texte                                                                 #
# --------------------------------------------------------------------------- #
def gpu_report_lines(status: GpuStatus) -> list[str]:
    """Lignes factuelles (torch / pilote / GPU) pour check_install et le journal."""
    lines: list[str] = []
    if status.torch_present:
        build = f"CUDA {status.cuda_build}" if status.cuda_build else "build CPU-only"
        lines.append(f"torch {status.torch_version} ({build})")
    else:
        lines.append("torch : non installé")
    if status.driver_ok:
        lines.append(f"pilote NVIDIA : {status.driver_version or 'détecté'} (nvidia-smi OK)")
    else:
        lines.append("pilote NVIDIA : nvidia-smi en échec")
    if status.ok:
        bf16 = "supporté" if status.bf16 else "non supporté (fp16 sera utilisé)"
        lines.append(f"GPU : {status.device_name} — {status.vram_gib} Go — bf16 {bf16}")
    return lines


def status_line(status: GpuStatus) -> str:
    """Résumé une ligne pour le journal au chargement du modèle."""
    if status.ok:
        drv = f", pilote {status.driver_version}" if status.driver_version else ""
        bf16 = "bf16 ✓" if status.bf16 else "bf16 ✗"
        vram = f"{status.vram_gib:.1f} Go" if status.vram_gib else "VRAM n/d"
        return (f"GPU : {status.device_name} ({vram}{drv}) — "
                f"torch {status.torch_version} (CUDA {status.cuda_build}) — {bf16}")
    cause = status.problems[0] if status.problems else "cause indéterminée"
    return f"GPU non détecté — {cause}"


def gpu_banner(status: GpuStatus | None = None) -> str:
    """Bandeau Markdown affiché en haut de l'interface Gradio au démarrage."""
    status = status or detect_gpu()
    if status.ok:
        drv = f" · pilote {status.driver_version}" if status.driver_version else ""
        bf16 = "bf16 ✓" if status.bf16 else "bf16 ✗ (fp16 sera utilisé)"
        return (f"🟢 **GPU prêt** : {status.device_name} ({status.vram_gib} Go){drv} · "
                f"torch {status.torch_version} (CUDA {status.cuda_build}) · {bf16}")
    lines = ["🔴 **GPU CUDA non détecté — SeedVR2 ne pourra pas démarrer.**"]
    lines += [f"- {p}" for p in status.problems]
    lines.append("👉 Lancez `python check_install.py` pour le détail, ou cliquez "
                 "Démarrer : le journal affichera la réparation exacte.")
    return "\n".join(lines)


def unavailable_help(status: GpuStatus | None = None) -> str:
    """Texte complet de l'erreur levée quand on démarre sans GPU utilisable."""
    status = status or detect_gpu()
    parts = [
        "CUDA indisponible : le pipeline officiel SeedVR2 exige un GPU NVIDIA utilisable.",
        "",
        "Diagnostic :",
    ]
    parts += [f"  • {p}" for p in status.problems] or ["  • cause indéterminée"]
    if status.hints:
        parts.append("")
        parts.append("Réparation :")
        parts += [f"  • {h}" for h in status.hints]
    parts += [
        "",
        "Relancez ensuite l'application. Détails complets : python check_install.py",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
def main() -> int:
    """``python -m seedvr2_upscaler.gpu_check [--summary]`` — autonome."""
    import sys

    status = detect_gpu()
    if "--summary" in sys.argv[1:]:  # utilisé par run_app.bat : une seule ligne
        print(f"[GPU] {status_line(status)}")
        return 0 if status.ok else 1
    print("Diagnostic GPU/CUDA")
    print("-" * 56)
    for line in gpu_report_lines(status):
        print(f"  {line}")
    if not status.ok:
        print("  Causes probables :")
        for p in status.problems:
            print(f"   - {p}")
        print("  Réparation :")
        for h in status.hints:
            print(f"   - {h}")
    return 0 if status.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
