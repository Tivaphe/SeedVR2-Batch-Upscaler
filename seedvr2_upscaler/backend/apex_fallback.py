"""Repli apex → couches de normalisation natives de PyTorch.

Contexte : ``models/dit_v2/normalization.py`` du dépôt officiel importe
``apex.normalization.FusedLayerNorm`` / ``FusedRMSNorm`` **uniquement** quand
la configuration demande ``fusedln`` / ``fusedrms`` — c'est le cas de
``configs_3b/main.yaml``. Or apex ne se compile pas sous Windows.

Équivalence :
    - ``FusedLayerNorm`` ≙ ``torch.nn.LayerNorm`` (mêmes paramètres
      ``weight``/``bias`` → state_dict compatible) ;
    - ``FusedRMSNorm`` ≙ ``torch.nn.RMSNorm`` (torch ≥ 2.4 ; paramètre
      ``weight`` → state_dict compatible). Une implémentation maison est
      fournie pour les anciens torch.

Comme pour FlashAttention : si le vrai ``apex`` est installable, il est
conservé ; sinon le shim est injecté dans ``sys.modules`` avant l'import du
dépôt officiel.
"""
from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from typing import Callable

ProgressCb = Callable[[str], None]


def _make_module(name: str, *, is_package: bool) -> types.ModuleType:
    """Module de shim avec métadonnées d'import complètes (``__spec__`` etc.),
    pour rester compatible avec les introspections ``importlib`` en aval."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None, is_package=is_package)
    module.__loader__ = None
    if is_package:
        module.__path__ = []
    return module


def install_apex_shim(log: ProgressCb = print) -> bool:
    """Injecte un module ``apex`` minimal si le vrai est absent."""
    try:
        import apex  # noqa: F401

        return True
    except Exception:
        pass
    if "apex" in sys.modules and hasattr(sys.modules["apex"], "normalization"):
        return True

    import torch
    from torch import nn

    log("apex introuvable — repli LayerNorm/RMSNorm natifs "
        "(numériquement équivalents, légèrement moins rapides).")

    class FusedLayerNorm(nn.LayerNorm):  # API et state_dict identiques à apex
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
            super().__init__(normalized_shape, eps=eps,
                             elementwise_affine=elementwise_affine, **kw)

    if hasattr(nn, "RMSNorm"):  # torch >= 2.4
        class FusedRMSNorm(nn.RMSNorm):  # type: ignore[no-redef]
            def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
                super().__init__(normalized_shape, eps=eps,
                                 elementwise_affine=elementwise_affine, **kw)
    else:
        class FusedRMSNorm(nn.Module):  # type: ignore[no-redef]
            """Implémentation de secours pour torch < 2.4."""

            def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
                super().__init__()
                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps
                self.elementwise_affine = elementwise_affine
                if elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(self.normalized_shape))
                else:
                    self.register_parameter("weight", None)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                dtype = x.dtype
                x32 = x.float()
                rms = x32.pow(2).mean(dim=-1, keepdim=True)
                out = x32 * torch.rsqrt(rms + self.eps)
                if self.weight is not None:
                    out = out * self.weight.float()
                return out.to(dtype)

    normalization = _make_module("apex.normalization", is_package=False)
    normalization.FusedLayerNorm = FusedLayerNorm  # type: ignore[attr-defined]
    normalization.FusedRMSNorm = FusedRMSNorm      # type: ignore[attr-defined]

    apex = _make_module("apex", is_package=True)
    apex.__doc__ = "Repli natif du paquet apex (voir apex_fallback.py)."
    apex.normalization = normalization  # type: ignore[attr-defined]

    sys.modules["apex"] = apex
    sys.modules["apex.normalization"] = normalization
    return True
