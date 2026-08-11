"""Repli FlashAttention → PyTorch SDPA (``scaled_dot_product_attention``).

Contexte : le DiT v2 du dépôt officiel importe
``from flash_attn import flash_attn_varlen_func`` au niveau module, et ce
paquet ne se compile pas sous Windows.

Pourquoi ce shim est sûr ici : le dépôt n'appelle **que**
``flash_attn_varlen_func(q, k, v, cu_seqlens_*, max_seqlen_*)`` **sans**
``causal`` ni ``window_size`` (seul ``deterministic`` est propagé).
SDPA est alors strictement équivalent mathématiquement ; PyTorch utilise
d'ailleurs des noyaux FlashAttention en interne quand c'est possible.
Coût : légèrement plus de VRAM / un peu plus lent que le noyau varlen
d'origine dans le pire des cas (séquences hétérogènes).

Le shim est injecté dans ``sys.modules`` AVANT l'import du dépôt officiel ;
si le vrai ``flash_attn`` est présent, il est conservé.
"""
from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from typing import Callable

ProgressCb = Callable[[str], None]


def _make_module(name: str, *, is_package: bool) -> types.ModuleType:
    """Module de shim avec métadonnées d'import complètes.

    ``importlib`` lève ``ValueError('….__spec__ is None')`` sur un module
    dépourvu de ``__spec__`` (résolution dynamique, find_spec…) : les modules
    injectés dans ``sys.modules`` doivent donc imiter des modules réels.
    """
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None, is_package=is_package)
    module.__loader__ = None
    if is_package:
        module.__path__ = []  # marque le « package » pour les sous-modules
    return module


def install_flash_attn_shim(log: ProgressCb = print) -> bool:
    """Injecte un module ``flash_attn`` de repli si le vrai est absent.

    Returns:
        True si l'environnement dispose d'un flash_attn utilisable
        (le vrai ou le shim), False si le shim n'a pas pu être installé.
    """
    try:
        import flash_attn  # noqa: F401

        return True  # le vrai paquet est disponible : rien à faire
    except Exception:
        pass

    if "flash_attn" in sys.modules and hasattr(
        sys.modules["flash_attn"], "flash_attn_varlen_func"
    ):
        return True  # shim déjà installé

    import torch
    import torch.nn.functional as F

    log("FlashAttention introuvable — repli PyTorch SDPA (numériquement "
        "équivalent ici, légèrement plus gourmand en VRAM).")

    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,  # accepté, ignoré (inférence)
        **_: object,
    ) -> torch.Tensor:
        """Même signature que ``flash_attn.flash_attn_varlen_func``.

        Layout compact « varlen » : q/k/v de forme (total, têtes, dim_tête),
        les bornes de séquences étant données par ``cu_seqlens_*``.
        """
        if float(dropout_p or 0.0) != 0.0:
            raise RuntimeError("Repli SDPA : dropout_p non supporté.")
        if window_size not in ((-1, -1), [-1, -1], None):
            raise RuntimeError("Repli SDPA : window_size non supporté.")

        lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
        nseq = len(lens_q)
        total, heads, head_dim = q.shape

        # Chemin rapide : séquences uniformes → une seule SDPA « dense ».
        if nseq > 0 and lens_q == lens_k and len(set(lens_q)) == 1 and not causal:
            length = lens_q[0]
            qb = q.view(nseq, length, heads, head_dim).transpose(1, 2)
            kb = k.view(nseq, length, heads, head_dim).transpose(1, 2)
            vb = v.view(nseq, length, heads, head_dim).transpose(1, 2)
            return _sdpa_best_kernel(qb, kb, vb, softmax_scale
                                     ).transpose(1, 2).reshape(total, heads, head_dim)

        # Chemin générique : boucle sur les séquences (longueurs variables).
        outs: list[torch.Tensor] = []
        for qi, ki, vi in zip(
            torch.split(q, lens_q, dim=0),
            torch.split(k, lens_k, dim=0),
            torch.split(v, lens_k, dim=0),
        ):
            qi4 = qi.transpose(0, 1).unsqueeze(0)      # (1, têtes, lq, d)
            ki4 = ki.transpose(0, 1).unsqueeze(0)      # (1, têtes, lk, d)
            vi4 = vi.transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(
                qi4, ki4, vi4,
                dropout_p=0.0,
                is_causal=causal and qi4.shape[2] == ki4.shape[2],
                scale=softmax_scale,
            )
            outs.append(oi.squeeze(0).transpose(0, 1))
        return torch.cat(outs, dim=0)

    module = _make_module("flash_attn", is_package=True)
    module.__doc__ = "Repli SDPA du paquet flash-attn (voir flash_fallback.py)."
    module.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore[attr-defined]
    sys.modules["flash_attn"] = module
    return True


def _sdpa_best_kernel(qb, kb, vb, scale):
    """Exécute SDPA en préférant explicitement les noyaux rapides.

    Ordre : noyau CUDA FlashAttention (intégré à PyTorch, sm_70+) → efficient
    attention → fallback math. Sans ce guidage, certaines combinaisons
    (formes longues + bf16) choisissent le noyau « math », beaucoup plus
    lent et gourmand en mémoire.
    """
    import torch
    import torch.nn.functional as F

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            try:
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(
                        qb, kb, vb, dropout_p=0.0, scale=scale
                    )
            except Exception:
                continue
    except Exception:
        pass  # vieux torch : on laisse le dispatcher décider
    return F.scaled_dot_product_attention(qb, kb, vb, dropout_p=0.0, scale=scale)
