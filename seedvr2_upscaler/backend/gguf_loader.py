"""Chargement des poids SeedVR2 au format GGUF (versions quantifiées).

Les GGUF de SeedVR2 (ex. ``seedvr2_ema_3b-Q4_K_M.gguf``, dépôt HF
``cmeka/SeedVR2-GGUF``) contiennent les mêmes tenseurs que le checkpoint EMA,
stockés en blocs quantifiés. Ce module les **déquantise en mémoire** (via le
paquet de référence ``gguf``) pour produire un ``state_dict`` torch standard,
ensuite injecté dans le DiT officiel — on ne réimplémente pas le pipeline.

Note mémoire : la déquantisation est faite sur CPU. Comptez ~6,5 Go de RAM
libre pour un modèle 3B, quelle que soit la quantisation choisie.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # annotations uniquement : torch reste un import paresseux
    import torch

ProgressCb = Callable[[str], None]

# Préfixes cosmétiques parfois ajoutés par l'outil de conversion GGUF.
_PREFIXES_TO_STRIP = ("model.", "dit.", "module.", "ema.")


def _normalize_key(key: str) -> str:
    """Ramène une clé GGUF vers le nommage du state_dict officiel du DiT."""
    for prefix in _PREFIXES_TO_STRIP:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_gguf_state_dict(path: str | Path, log: ProgressCb = print) -> dict[str, "torch.Tensor"]:
    """Déquantise un fichier GGUF et retourne un ``state_dict`` torch (CPU).

    Détails du format (paquet ``gguf`` officiel) :
        - ``ReaderTensor.data`` est déjà reshapé dans l'ordre numpy (donc
          dans l'ordre torch d'origine) pour les types F32/F16 ;
        - les blocs quantifiés et le BF16 sont lus en ``uint8`` bruts puis
          restaurés à leur forme d'origine par ``gguf.quants.dequantize``,
          qui gère nativement Q3_K_M … Q8_0 et BF16.

    Returns:
        Dictionnaire ``{nom_de_paramètre: Tensor float32 (CPU)}``.
    """
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    path = Path(path)
    log(f"GGUF : lecture de {path.name}…")
    reader = GGUFReader(str(path))

    state: dict[str, torch.Tensor] = {}
    n_quants = 0
    for tensor in reader.tensors:
        key = _normalize_key(tensor.name)
        # dequantize() est transparent pour F32/F16 et couvre BF16 + quants.
        array = dequantize(tensor.data, tensor.tensor_type)
        state[key] = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        if tensor.tensor_type.name.startswith("Q"):
            n_quants += 1

    log(f"GGUF : {len(state)} tenseurs lus dont {n_quants} déquantifiés.")
    return state


def cast_state_dict(
    state: dict[str, "torch.Tensor"],
    dtype: "torch.dtype",
) -> dict[str, "torch.Tensor"]:
    """Cast les tenseurs flottants vers ``dtype`` (bf16/fp16) pour le DiT.

    Les tenseurs entiers (buffers d'index éventuels) sont conservés tels quels.
    """
    import torch

    return {
        key: (value.to(dtype) if value.is_floating_point() else value)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
