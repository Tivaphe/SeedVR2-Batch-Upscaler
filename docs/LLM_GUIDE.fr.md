# Guide LLM — SeedVR2 Batch Upscaler

> **Version française** | [English version](LLM_GUIDE.md)
>
> **Public visé** : ce document s'adresse à un assistant IA (LLM) chargé de
> comprendre, déboguer ou adapter ce projet à une autre machine. Lis-le en
> entier avant de modifier du code : architecture, carte des dépendances,
> historique complet des problèmes déjà résolus, et règles de non-régression.

---

## 1. Portrait du projet

- **Quoi** : application de bureau Windows qui upscale des images **par lot**
  avec le modèle de super-résolution en une étape de diffusion **SeedVR2-3B**
  de ByteDance.
- **Politique de code officiel** : l'application *enrobe les scripts
  d'inférence officiels* (dépôt `ByteDance-Seed/SeedVR`, cloné sous `SeedVR/`
  à côté de `app.py`). Elle ne **réimplémente pas** le pipeline. Seul le
  *chargement* des poids est étendu (safetensors FP16/FP8, GGUF), plus des
  cales de compatibilité Windows.
- **Stack** : Python 3.11–3.14 (3.12 le plus sûr), torch CUDA (index cu130 ;
  repli cu128 pour les pilotes antérieurs au branch 580 — cu128 s'arrête à
  torch 2.11), interface Gradio, moteur de lot threadé. Commentaires et logs
  utilisateur en **français** ; interface bilingue FR/EN (bascule à chaud via
  `i18n.py`).
- **Matériel de référence** (validé) : RTX 2000 Ada 16 Go, sm_89, bf16 OK,
  Windows 11, 32 Go RAM. Pipeline complet fonctionnel (GGUF Q8_0, ×4, tuiles 288 px).
- **Pas de mode CPU/démo, par conception** : sans GPU CUDA, l'application
  refuse de démarrer et affiche un diagnostic de réparation (§6.6). Ne pas
  réintroduire de repli silencieux.

## 2. Carte du dépôt

```
app.py                     Entrée GUI + CLI ; pose PYTORCH_CUDA_ALLOC_CONF
                           (=expandable_segments:True) AVANT tout import torch.
check_install.py           Diagnostic autonome (Python, dépendances, GPU, dépôt, modèles).
install.bat / run_app.bat  Amorçage Windows (sortie console en anglais).
requirements.txt           Dépendances de l'app — les commentaires DOIVENT
                           commencer par '#' (';' = marqueur d'environnement pip → bug n°1).
config/settings.json       Réglages utilisateur (ignoré par git ; migration auto
                           dans Settings.__post_init__).
models/                    Poids (ignoré par git). SeedVR/ = clone officiel (ignoré).

seedvr2_upscaler/
├── constants.py           Chemins, extensions, presets HF, nom de la variable SEEDVR2_REPO.
├── models.py              Dataclasses/énums typées : JobConfig, ModelInfo, ModelKind,
│                          OutputFormat/ConflictPolicy (from_label tolère FR/EN/canonique),
│                          Precision, TilingConfig, RunnerState, BatchStats, ImageOutcome.
├── settings.py            Persistance JSON. __post_init__ migre les anciens libellés
│                          FR des radios → valeurs canoniques, neutralise le backend
│                          « demo » supprimé, borne la langue à fr/en. LES NOUVEAUX
│                          CHAMPS SE DÉCLARENT ICI avec valeur par défaut.
├── i18n.py                Tables FR/EN (mêmes clés — test unitaire) + tr().
├── registry.py            scan_models/classify_model (type+quant depuis le nom),
│                          AuxAssets (VAE/pos/neg), ensure_aux_assets (téléchargement
│                          auto depuis HF ByteDance-Seed/SeedVR2-3B).
├── naming.py              apply_suffix (désactivé par défaut), resolve_output_path
│                          (écraser/ignorer/renommer « nom (1).ext », set de réservation).
├── images.py              load_image (transposition EXIF, ICC, bytes EXIF, split RGBA),
│                          save_image (PNG/JPEG/WebP, qualité, restauration EXIF+ICC,
│                          piexif pour PNG), split_alpha/merge_alpha.
├── tiling.py              compute_tiles (grille à recouvrement), stitch (fusion douce),
│                          needs_tiling (sortie > 4 Mpx → auto).
├── gpu_check.py           detect_gpu() → GpuStatus (pilote via nvidia-smi, build torch,
│                          cuda_available, bf16), cause_key() pour le bandeau bilingue,
│                          aide longue en français pour les logs/CLI. Ne lève jamais.
├── backend/
│   ├── base.py            ABC UpscaleBackend + BackendOptions ; upscale() générique :
│   │                      split alpha, arrondi ceil_div(·,16), boucle tuiles avec
│   │                      callback UI tile_cb(i,n), stitch. MAX_TILE_OUT_SIDE = 1152
│   │                      borne le côté des tuiles de SORTIE (régime d'entraînement) — bug n°9.
│   ├── official.py        Le cœur. Enrobe projects.video_diffusion_sr.infer.
│   │                      VideoDiffusionInfer + configs_3b/main.yaml. Voir §3.
│   ├── gguf_loader.py     GGUF → state_dict via le paquet officiel `gguf` ; déquantise
│   │                      les Q* sur CPU vers le dtype cible, renomme les clés style
│   │                      llama vers le schéma du checkpoint DiT. Pic ~6,5 Go RAM au chargement.
│   ├── flash_fallback.py  Injecte un faux module `flash_attn` (flash_attn_varlen_func)
│   │                      adossé à torch SDPA. Avec ModuleSpec (importlib exige
│   │                      __spec__/__path__) — bug n°7. Le vrai paquet prime s'il existe.
│   └── apex_fallback.py   Idem pour apex : FusedLayerNorm≈nn.LayerNorm,
│                          FusedRMSNorm≈nn.RMSNorm (state_dict compatible) — bug n°5.
├── worker.py              Thread BatchRunner + EventBus (log/progress/preview/state/
│                          tile/finished). Événements pause/reprise/annulation, ETA
│                          (moyenne glissante 5 images), fichier de reprise
│                          `.seedvr2_resume.json` (empreinte de config), repli OOM→tuilage,
│                          écritures parallèles (2), log.txt en ajout, report.txt,
│                          pré-contrôle d'espace disque, blocage de la veille Windows
│                          (SetThreadExecutionState) et bip de fin.
│                          Le modèle est chargé UNE SEULE FOIS par lot.
└── gui_gradio.py          Blocks Gradio. Timer 0,5 s vidant l'EventBus. Tous les textes
                           via le registre i18n (bascule de langue sans rechargement).
                           Les radios portent des valeurs canoniques (couples label/valeur).
                           Bandeau GPU construit depuis cause_key() dans la langue courante.
tests/selftest.py          21 tests unitaires sans GPU : nommage, registre, tuiles,
                           I/O, réglages (+migration), parité i18n, moteur de lot avec
                           un faux backend CPU injecté (BatchRunner(backend_factory=…)).
docs/LLM_GUIDE*.md         Ce fichier.
```

## 3. Comment le pipeline officiel est enrobé (backend/official.py)

Une image est traitée comme une **vidéo d'une frame** par les scripts officiels.

Chargement (`load()` → `_load_pipeline()`) :
1. Diagnostic `gpu_check` d'abord ; abandon avec consignes de réparation si pas de CUDA.
2. `torch.backends.cudnn.benchmark = True` (formes de tuiles constantes → autotune conv).
3. Installation proactive des cales flash/apex (Windows), puis chaque étape est
   enveloppée par `_with_autorepair` : sur ModuleNotFoundError, pip-install le
   paquet connu (table `_PYPI_NAMES`) dans l'interpréteur *courant*, une fois.
4. Charge `configs_3b/main.yaml` via `common.config.load_config` avec le CWD
   temporairement basculé à la racine du dépôt (chemins relatifs dans la config).
5. `_init_distributed_best_effort` : le décorateur officiel `@log_runtime` appelle
   `dist.barrier()` sans condition → création d'un **vrai groupe gloo
   mono-processus** (pas de NCCL sous Windows). Les shims restent l'ultime recours.
6. dtype : bf16 si `torch.cuda.is_bf16_supported()` sinon fp16.
7. Assets : `ensure_aux_assets` (VAE + embeddings texte pos/neg, téléchargement HF auto).
8. Poids DiT : `.pth` officiel → `runner.configure_dit_model(...)` verbatim ;
   safetensors/GGUF → `_build_dit_from_state` : construction sur meta-device
   (calque la branche officielle), `load_state_dict(strict=False, assign=True)`,
   puis `_materialize_buffers` (bug n°8), `.to(device, dtype)`.
9. VAE : `configure_vae_model()` puis `_tune_vae_memory_limit` remplace le
   `conv_max_mem: 0,5 Gio` officiel par `min(7,0, max(1,0, libre*0,5)) Gio` —
   le défaut 0,5 Gio décode en micro-slices (17 s/tuile mesuré) — bug n°10.
10. Diffusion : cfg 1,0, rescale 0, steps = 1, `configure_diffusion()`.
11. `color_fix.wavelet_reconstruction` optionnel si le fichier existe dans le dépôt.

Inférence par région (`_upscale_region_impl`), calquant la branche image de
`projects/inference_seedvr2_3b.py` :
`TF.to_tensor → [T=1,C,H,W] → NaResize(resolution=√(out_h·out_w), area,
downsample_only=False) → clamp01 → DivisibleCrop(16,16) → Normalize(0,5;0,5) →
Rearrange t c h w → c t h w` → `vae_encode([cond])` → `_generation_step`
(bruits randn_like, `set_seed(seed, same_across_ranks=True)` par tuile avec
repli manual_seed, cond_noise_scale=0,0, `get_condition(task='sr')`,
texts_pos/neg depuis les tenseurs GPU en cache, `runner.inference(...,
dit_offload=low_vram)` sous `inference_mode`+autocast bf16) → **branche de
sortie exactement officielle** : `video[:, None]` quand `video.ndim == 3`
(signifie [C,H,W], axe temps perdu) — `video[None]` transpose silencieusement
C/T → image 1 canal (« not enough image data ») — bug n°9. Puis
clamp [-1,1]→[0,255] uint8 RGB.

Swap VRAM identique au script officiel quand `low_vram` (dit↔cpu, vae↔cuda) ;
désactivé automatiquement si VRAM ≥ 14 Go, activé sous 10 Go (`_tune_low_vram`).

## 4. Dépendances

venv applicatif (`requirements.txt` + install.bat) : gradio, Pillow, numpy,
safetensors, gguf, huggingface_hub, einops, omegaconf (+ piexif optionnel pour
l'EXIF dans les PNG). Plus torch/torchvision depuis l'index **cu130**
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` ;
wheels Python 3.10-3.14, pilote NVIDIA >= 580 ; cu128 reste en repli pour les
pilotes plus anciens, torch <= 2.11).

Dépendances du dépôt officiel (sélection curée, **pas** son requirements.txt
qui épingle torch==2.3.0 — casserait les Python récents et rétrograderait le
torch CUDA) : `diffusers>=0.29`, `transformers>=4.38`,
`rotary-embedding-torch>=0.5`, `opencv-python`, `mediapy`. `apex` et
`flash-attn` sont remplacés par des cales.

Python ≥ 3.14 → wheels CUDA torch potentiellement absentes (`No matching
distribution found for torch`) → recréer le venv avec `py -3.12`.

## 5. Couche de compatibilité Windows (pourquoi chaque cale est sûre)

| Pièce manquante | Cale | Pourquoi équivalent ici |
|---|---|---|
| NCCL (absent de Windows) | vrai groupe **gloo** 1 processus (`init_process_group`, tcp://127.0.0.1:29512) | seuls `dist.barrier()`/getters de rang sont utilisés ; world_size=1 |
| flash-attn | `flash_attn_varlen_func` en SDPA (F.scaled_dot_product_attention par segment de `cu_seqlens`) | DiT v2 l'appelle **sans** causal ni fenêtre → SDPA mathématiquement identique (validé <1e-5 en fp32 vs attention naïve, longueurs uniformes et variables) |
| apex `fusedln`/`fusedrms` (norms de configs_3b) | `nn.LayerNorm` / `nn.RMSNorm` (eps transmis, affine par élément, mêmes noms de paramètres → state_dict compatible) | les kernels fusionnés sont identiques numériquement, juste plus rapides |

Les deux cales construisent de vrais `types.ModuleType` avec `ModuleSpec` +
loader, et `__path__=[]` pour les paquets, sinon `importlib`/`pkgutil` cassent
(bug n°7). Si le vrai paquet existe, il est utilisé tel quel.

## 6. Historique des bugs (symptôme → cause → correction) — À LIRE AVANT DE DÉBOGUER

1. **gradio ne s'installe jamais ; pip meurt avec `InvalidMarker`** — des
   commentaires de requirements.txt commençaient par `;` → interprétés comme
   marqueurs d'environnement. Correctif : commentaires préfixés `#` uniquement.
2. **`No matching distribution for torch` sur un index CUDA** — Python trop
   récent pour les wheels de cet index. Correctif : install.bat préfère
   `py -3.12`/`py -3.11`, chaîne cu130 → cu128 → PyPI.
3. **`ModuleNotFoundError: rotary_embedding_torch` au premier chargement** —
   dépendances du dépôt absentes. Correctif : installation curée non épinglée
   + `_with_autorepair` à l'exécution (un essai pip auto par module, puis
   erreur claire avec la commande exacte).
4. **flash-attn impossible à compiler sous Windows** → cale SDPA (§5).
5. **apex requis par configs_3b (`fusedln`, `fusedrms`, `norm: fusedrms`,
   `txt_in_norm: fusedln`)** → cale nn.LayerNorm/RMSNorm (§5).
6. **`ValueError: Default process group has not been initialized`** — le
   décorateur officiel `log_runtime` barrière sans condition ; `init_torch`
   impose NCCL. Correctif : env RANK/LOCAL_RANK/WORLD_SIZE + vrai groupe gloo (§5).
7. **`ValueError: flash_attn.__spec__ is None`** — modules factices sans
   métadonnées importlib. Correctif : fabrique ModuleSpec dans les deux cales.
8. **`AssertionError()` nu après le chargement des poids** — `rotary-embedding-torch`
   0.9.x enregistre des buffers non persistants supplémentaires
   (`cached_freqs` 8192×42, `freqs`), alors que le
   `meta_non_persistent_buffer_init_fn` officiel ne matérialise que les buffers
   nommés *dummy* avant d'asserter. Correctif : `_materialize_buffers`
   recompose `freqs` RoPE exactement (`1/θ^(2i/d)`, freqs_for='lang', θ=10000 ;
   vérifié 0,00 vs module neuf), zéros pour les placeholders cached/dummy.
9. **« not enough image data » après 7 min d'inférence correcte + sortie
   silencieusement 1 canal** — mon code utilisait `video[None]` au lieu du
   `video[:, None]` officiel ; ndim==3 ⇒ [C,H,W]. Transposition muette.
   Correctif + tests de régression sur les deux branches ndim. **Leçon : copier
   l'indexation officielle caractère par caractère.**
10. **4 031 s/image** — trois causes empilées : (a) swap `low_vram` actif sur
    carte 16 Go → seuils automatiques (§3) ; (b) tuiles au-delà du régime
    d'entraînement (~1 Mpx) → coût d'attention quadratique → bornage du côté
    de SORTIE des tuiles à 1152 px (`MAX_TILE_OUT_SIDE`, journalisé) ; (c) VAE
    `conv_max_mem: 0,5 Gio` → décodage micro-slicé → `_tune_vae_memory_limit`.
    Après correctifs : quelques secondes/tuile.
11. **« GPU non détecté » après redémarrage, et dégradation silencieuse de
    qualité** — l'application basculait silencieusement sur un backend démo CPU.
    Correctif : backend démo **entièrement supprimé** ; diagnostics
    `gpu_check.detect_gpu()` partout (ligne .bat, bandeau UI, erreur au
    chargement, check_install.py) avec les trois causes classiques (pilote
    muet / build torch CPU-only / absence de redémarrage après mise à jour du
    pilote) et les commandes exactes de réparation.
12. **Confusion du glisser-déposer Gradio** — les dossiers déposés atterrissent
    dans le dossier temporaire de l'OS → les sorties « disparaissaient » dans
    `…\Temp\gradio\…\upscaled`. Le libellé UI avertit ; conseiller un dossier
    de sortie explicite. De même : l'aperçu est une **vignette** ≤768 px — un
    utilisateur l'a prise pour une sortie « réduite » ; le journal affiche les
    dimensions réelles (`entrée×… → sortie×…`) pour chaque image.

## 7. Référence de configuration

- **Variables d'env** : `SEEDVR2_REPO` (chemin du dépôt),
  `PYTORCH_CUDA_ALLOC_CONF` (posée par app.py, ne pas retirer).
- **Fichier de réglages** `config/settings.json` : JSON simple ; clés inconnues
  ignorées ; valeurs canoniques des radios (`x2/x4/x8/custom`,
  `keep/png/jpg/webp`, `overwrite/skip/rename`, `auto/bf16/fp16`,
  `auto/official`, `fr/en`). Les anciens libellés FR migrent automatiquement
  (Settings.__post_init__).
- **Constantes clés** (`constants.py`/`base.py`) : `DIVISOR=16`,
  `TILING_AUTO_THRESHOLD_MPX=4,0`, `MAX_TILE_OUT_SIDE=1152`,
  `_ETA_WINDOW=5`, port gloo 29512.

## 8. Recettes d'adaptation

**Ajouter une option GUI de bout en bout (check-list — oublier une étape casse en silence) :**
1. champ dans `Settings` (+ validation dans `__post_init__` si énuméré) ;
2. ajouter la clé à la liste de `_persist_settings` **et** à `setting_inputs`
   et au tuple de dépaquetage de `_on_start` — **même ordre partout** ;
3. champ dans `JobConfig` ; câblage du job dans `_on_start` ;
4. consommation dans `worker.py`/`backend` ;
5. clés i18n dans les DEUX tables d'`i18n.py` (parité vérifiée par test) ;
6. enregistrer le widget avec `reg(comp, label=("s","ma.cle"), …)` ;
7. ajouter/étendre un test unitaire ; lancer `python tests/selftest.py`.

**Ajouter une langue** : ajouter une troisième table dans `i18n.py` avec
exactement les mêmes clés ; entrée `SUPPORTED_LANGUAGES` ; rien d'autre.

**Ajouter un format de poids** : étendre `registry.classify_model` ; apprendre
à `official._read_state_dict` à produire un state_dict au schéma DiT ; le
reste est générique.

**Ne jamais faire** : réimplémenter en silence les maths du pipeline (enrober
le code officiel et le citer) ; changer les défauts de conservation des noms ;
perdre l'ensemencement par tuile (seed+index par tuile = reproductibilité) ;
avaler les exceptions (les logs portent les tracebacks complets, par
conception) ; ajouter un repli CPU/démo (politique du projet).

## 9. Commandes de validation (après chaque modification)

```bash
python -m compileall -q app.py check_install.py seedvr2_upscaler tests
python tests/selftest.py            # 21 tests, tous doivent passer, sans GPU
python -m seedvr2_upscaler.gpu_check --summary
python check_install.py
# Fumée GUI headless : build_app(dossier models temp) doit construire ~70
# composants, et _on_language_change(session, "en") doit renvoyer une update
# par widget enregistré.
```

## 10. Symptôme → où regarder

| Symptôme | Fichier |
|---|---|
| Noms/suffixes de sortie incorrects | `naming.py`, `worker._process_one` |
| Coutures de tuiles visibles | `tiling.stitch` (recouvrement/fondu) |
| Temps par image lent | ligne `étapes —` du log ; `official._tune_vae_memory_limit`, `base.MAX_TILE_OUT_SIDE`, `_tune_low_vram` |
| OOM | `BackendOptions.tiling`, low_vram, variable d'alloc |
| Erreurs d'import du dépôt SeedVR | `_with_autorepair`, `_PYPI_NAMES`, cales |
| Langue non appliquée | widget non enregistré via `reg(...)` ou clé i18n manquante (le test le détecte) |
| Réglages perdus/ancien libellé | migrations de `settings.__post_init__` |
| EXIF perdues | `images.save_image` (+ piexif installé ?) |
| GPU disparu après redémarrage | `gpu_check` (lancer `python check_install.py`) |
