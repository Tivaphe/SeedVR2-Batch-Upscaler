"""SeedVR2 Batch Upscaler — application d'upscaling d'images par lot.

Structure du paquet :
    constants   : constantes globales (chemins, extensions, presets)
    models      : dataclasses et énumérations tyées partagées
    settings    : persistance JSON des paramètres utilisateur
    registry    : détection/classification automatique des modèles dans models/
    naming      : politique de nommage et résolution de conflits
    images      : lecture/écriture image (formats, EXIF, canal alpha)
    tiling      : découpage en tuiles avec recouvrement + fusion douce
    gpu_check   : diagnostic pilote NVIDIA / build torch / CUDA utilisable
    i18n        : chaînes de l'interface FR/EN (bascule à chaud sans rechargement)
    backend/    : backend d'inférence officiel SeedVR2 (+ chargement FP8/GGUF,
                  replis Windows flash-attn/apex, diagnostic GPU sans repli)
    worker      : moteur de traitement par lot (thread, pause/reprise/annulation)
    gui_gradio  : interface graphique Gradio bilingue
"""

__version__ = "1.3.0"
APP_NAME = "SeedVR2 Batch Upscaler"
