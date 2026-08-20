# SeedVR2 Batch Upscaler

> **🇫🇷 Français** | [🇬🇧 English](README.md)

Application **Windows / Python 3.11+** d'upscaling d'images **par lot** avec **SeedVR2-3B**
(une étape de diffusion — très rapide) : interface **Gradio** moderne **bilingue FR/EN**
(sélecteur en haut de l'interface), moteur de lot threadé (Pause / Reprise / Annuler),
et un backend qui **s'appuie sur les scripts d'inférence officiels** du dépôt ByteDance
au lieu de réimplémenter le pipeline.

```
input/                    output/
├── photo001.png          ├── photo001.png      ← noms conservés à l'identique
├── vacances.jpg    ──▶   ├── vacances.jpg
└── image_test.webp       └── image_test.webp
```

## Aperçu

Comparaisons réelles (production de l'application) — **gauche : original,
droite : SeedVR2 ×4**, affichés à taille identique :

| Cité flottante | Feuille à contre-jour |
|---|---|
| ![Comparaison — cité](docs/showcase-city.jpg) | ![Comparaison — feuille](docs/showcase-leaf.jpg) |

> 📷 Pour ajouter une capture de l'interface, enregistrez-la sous
> `docs/screenshot_ui.png` et insérez-la ici. Pour générer une comparaison
> composée automatiquement depuis un lot réel :
> `python docs/make_showcase.py --input <entrée> --output <sortie>`.

## Fonctionnalités

| Domaine | Détail |
|---|---|
| Modèles | Officiel `.pth`, safetensors **FP16/BF16**, **FP8 e4m3fn**, **GGUF** Q3_K_M · Q4_K_M · Q5_K_M · Q6_K · Q8_0 |
| Détection | Scan automatique de `models/` au démarrage + liste déroulante + bouton d'actualisation + téléchargement de presets HF intégré |
| Noms | Conservation **exacte** du nom d'entrée (aucun suffixe par défaut) ; option « Comme l'entrée » conserve aussi **l'extension** ; suffixe uniquement si activé explicitement ; conflits → Écraser / Ignorer / Renommer `photo (1).png` |
| Lot | Dossier d'entrée complet, formats PNG/JPG/JPEG/WEBP/BMP/TIFF, poursuite après échec, `log.txt` (journal complet) + `report.txt` (bilan : temps total, moyenne, erreurs) |
| UI Gradio | **Bilingue FR/EN à chaud** (sans rechargement), sélection de dossiers, **glisser-déposer**, facteur ×2/×4/×8/personnalisé, format + qualité de sortie, barre de progression, **ETA**, journal temps réel, **aperçu avant/après** (curseur comparatif), boutons **Démarrer / Pause / Reprendre / Annuler**, bandeau d'état GPU |
| VRAM | Modèle chargé **une seule fois**, `torch.inference_mode()`, BF16 auto (FP16 sinon), swap CPU↔GPU du DiT/VAE (comme le script officiel), vidage du cache entre images, **tuilage** automatique + repli automatique vers le tuilage en cas d'OOM, allocateur `expandable_segments` (anti-fragmentation), `cudnn.benchmark` (autotune des convs, formes constantes), embeddings texte gardés sur GPU, budget VAE calibré sur la VRAM libre |
| Bonus | Reprise d'un lot interrompu (`.seedvr2_resume.json`), sauvegarde automatique des paramètres, écritures disque en parallèle, EXIF + profil ICC préservés (JPEG/WebP natifs, PNG via piexif), canal alpha préservé, CLI complète |
| Confort | **Veille Windows bloquée pendant le lot** (restaurée après), bip de fin de lot, vérification de l'espace disque au départ, sous-progression « tuile i/N », bouton 📂 pour ouvrir le dossier de sortie, `log.txt` en mode ajout (historique conservé) |

## 1. Installation (Windows)

```bat
install.bat        :: crée .venv (Python 3.12/3.11 de préférence), installe torch CUDA
                   :: (essaie les index cu130, cu128, puis PyPI), les dépendances,
                   :: clone le dépôt officiel et vérifie l'installation
run_app.bat        :: lance l'interface graphique (avec contrôles préalables)
check_install.py   :: diagnostic complet (versions, CUDA, modèles, dépôt)
```

> **Python 3.14** : des wheels CUDA existent sur l'index **cu130** (torch 2.9-2.13,
> pilote ≥ 580). Si vous voyez `No matching distribution found for torch`,
> l'index utilisé n'a pas de wheel pour votre Python — recréez alors
> l'environnement avec Python 3.12 : `py -3.12 -m venv .venv`
> (install.bat le privilégie automatiquement).
> **Pilote NVIDIA ancien (< 580)** : utilisez plutôt l'index cu128
> (torch ≤ 2.11, CUDA 12.8) — remplacez `cu130` par `cu128` dans les commandes.
>
> **apex** : le dépôt officiel requiert `apex`. Sous Windows sa compilation
> échoue — c'est attendu et **non bloquant** : l'application injecte un repli
> `nn.LayerNorm`/`nn.RMSNorm` numériquement équivalent (voir FAQ).

Équivalent manuel :

```powershell
py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt mediapy
git clone https://github.com/ByteDance-Seed/SeedVR.git SeedVR
```

Le dépôt officiel est localisé automatiquement dans `SeedVR/`,
`seedvr2_repo/`, `vendor/SeedVR/` ou via la variable d'env `SEEDVR2_REPO`.

## 2. Les modèles (dossier `models/`)

Déposez n'importe lequel de ces poids dans `models/` — ils seront détectés et
classés automatiquement (ou utilisez le bouton **Télécharger** de l'interface) :

| Fichier | Type | VRAM indicative |
|---|---|---|
| `seedvr2_ema_3b.pth` | Officiel | ~16-24 Go |
| `seedvr2_ema_3b_fp16.safetensors` | FP16 | ~12-16 Go |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | FP8 | ~8-12 Go |
| `seedvr2_ema_3b-Q4_K_M.gguf` … `-Q8_0.gguf` | GGUF | ~6-8 Go (+6,5 Go RAM pour la déquantisation) |

Le VAE (`ema_vae.pth`) et les embeddings (`pos_emb.pt`, `neg_emb.pt`) du pipeline
officiel sont **téléchargés automatiquement** depuis `ByteDance-Seed/SeedVR2-3B`
au premier lancement, si absents de `models/`.

## 3. Utilisation

### Interface graphique

```powershell
python app.py           # ou run_app.bat ; --share pour exposer en réseau
```

1. Choisissez la **langue** (sélecteur en haut à droite), les dossiers
   d'entrée/sortie (ou **glissez un dossier** dans la zone prévue) ;
2. Sélectionnez un modèle dans la liste ;
3. Réglez le facteur (×2/×4/×8/personnalisé), le format et la qualité ;
4. **Démarrer** — suivez progression, temps restant et journal ; **Pause/Reprendre/Annuler**
   à tout moment ; l'aperçu avant/après se met à jour sur la dernière image.

Tous les paramètres sont **sauvegardés automatiquement** (`config/settings.json`).
En cas d'interruption, relancez simplement : les images déjà faites sont ignorées
(option « Reprendre un lot interrompu », activée par défaut).

### Ligne de commande

```powershell
python app.py --cli --input input --output output --model seedvr2_ema_3b-Q4_K_M.gguf `
    --scale 4 --format keep --quality 95 --tiling
```

> ⚠️ Un **GPU NVIDIA + CUDA** est obligatoire (il n'existe pas de mode CPU :
> sans SeedVR2 réel, les résultats n'ont aucun intérêt). Si le GPU n'est plus
> détecté, l'interface affiche le **diagnostic complet** (pilote, build torch,
> cause probable, commande de réparation) — voyez aussi `python check_install.py`
> et la section FAQ.

## 4. Architecture

```
seedvr2_upscaler/
├── constants.py        # chemins, extensions, presets HF, dépôt officiel
├── models.py           # dataclasses/énums typées (JobConfig, ModelInfo…)
├── settings.py         # persistance JSON + migration des anciens réglages
├── i18n.py             # chaînes FR/EN de l'interface (i18n à chaud)
├── registry.py         # détection/classification des modèles + téléchargement HF
├── naming.py           # conservation des noms + politique de conflits
├── images.py           # I/O : EXIF, ICC, alpha, PNG/JPEG/WebP, BMP/TIFF entrants
├── tiling.py           # tuiles à recouvrement + fusion douce (feather)
├── gpu_check.py        # diagnostic pilote NVIDIA / build torch / CUDA (UI + CLI)
├── backend/
│   ├── base.py         # contrat abstrait + alpha + tuilage génériques
│   ├── official.py     # ✦ pipeline OFFICIEL SeedVR2 (configs_3b, transforms,
│   │                   #   VideoDiffusionInfer, 1 step DiT) + chargement étendu
│   │                   #   safetensors FP16/FP8
│   ├── gguf_loader.py  # déquantisation GGUF → state_dict (paquet gguf officiel)
│   ├── flash_fallback.py # repli SDPA de flash_attn (Windows, équivalent ici)
│   └── apex_fallback.py  # repli LayerNorm/RMSNorm d'apex (Windows, équivalent)
├── worker.py           # moteur de lot threadé : pause/reprise/annulation, ETA,
│                       # reprise, repli OOM→tuilage, I/O parallèle, log+report
└── gui_gradio.py       # interface Gradio bilingue (Timer 0,5 s + registre i18n)
app.py                  # entrée GUI + CLI
tests/selftest.py       # 21 tests unitaires (nommage, registre, tuiles, i18n, lot…)
docs/LLM_GUIDE.md       # guide complet destiné aux LLM (adaptation du projet)
docs/make_showcase.py   # image de comparaison du README depuis vos lots réels
```

**Démarche d'intégration officielle.** Le backend importe
`projects.video_diffusion_sr.infer.VideoDiffusionInfer` et la config
`configs_3b/main.yaml`, puis reproduit la branche « image » (une frame) du script
`projects/inference_seedvr2_3b.py` : `NaResize → DivisibleCrop(16) → Normalize →
vae_encode → generation_step (noise/condition/1 step, cfg=1) → vae_decode`,
avec le même swap VRAM `dit↔vae` et le `empty_cache()` entre fichiers. Seul le
**chargement des poids** est étendu (safetensors FP16/FP8, GGUF déquantifié),
car le script officiel ne lit que des `.pth`.

## 5. FAQ / dépannage

- **« GPU non détecté » après un redémarrage** → lancez `python check_install.py` :
  il distingue les trois causes classiques et donne la réparation exacte —
  ① pilote NVIDIA muet (`nvidia-smi` en échec, souvent après une mise à jour
  Windows → réinstaller le pilote et **redémarrer**) ; ② torch remplacé par une
  build **CPU-only** (pip sans index CUDA — nota : l'index cu128 ne publie plus
  de build CUDA depuis torch 2.12, donc un `pip install torch` via PyPI donne
  désormais une 2.12/2.13 CPU-only →
  `.venv\Scripts\python.exe -m pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cu130`) ; ③ pilote frais sans redémarrage ou
  variable `CUDA_VISIBLE_DEVICES` mal réglée. L'interface affiche le même
  diagnostic dans son bandeau supérieur et dans le journal au démarrage d'un lot.
- **FlashAttention / apex sous Windows** → inconstructibles depuis les sources ;
  l'application injecte automatiquement des **replis natifs équivalents**
  (PyTorch SDPA pour `flash_attn_varlen_func` — le dépôt ne l'utilise ni causal
  ni fenêtré ; `nn.LayerNorm`/`nn.RMSNorm` pour les `fusedln`/`fusedrms` de
  `configs_3b`). Si le vrai paquet est présent, il est utilisé tel quel.
- **Quelle version de torch pour les modèles SeedVR2 ?** → Aucune en
  particulier : les checkpoints (`.pth`/`.safetensors`/`.gguf`) sont de
  simples tenseurs, indépendants de la build torch. Le dépôt officiel
  épingle `torch==2.3.0` pour son environnement d'entraînement —
  incompatible avec les Python modernes — donc l'application ignore cet
  épingle et tourne sur torch récent (validé 2.10–2.13, builds CUDA
  cu128/cu130 ; l'index cu128 s'arrête à 2.11). Seule une build CPU-only
  (`+cpu`) ne peut pas faire tourner le pipeline.
- **OOM malgré le mode faible VRAM** → laissez « Toujours découper en tuiles »
  activé et baissez la taille de tuile (256). Le moteur bascule aussi
  automatiquement en tuilage après un OOM.
- **Dimensions de sortie** → multiples de 16 (contrainte du VAE/DivisibleCrop
  officiel) ; l'écart vise le facteur demandé (ex. 60 px ×2 → 128 px).
- **GGUF lent au premier chargement** → déquantisation CPU (une seule fois par lot).
- **Aller encore plus vite ?** Le lotage de tuiles (2–4 par passage GPU) est à
  l'étude : sa validation exige de vérifier dans le dépôt officiel comment
  l'attention gère le packing multi-frames — il ne sera livré qu'en option
  désactivée par défaut, après tests numériques, car une fuite de contenu
  entre tuiles dégraderait silencieusement la qualité.
- **Journal en français ?** Les messages de diagnostic du journal (`log.txt`)
  sont en français ; l'interface est, elle, entièrement bilingue FR/EN.
- Le journal du lot et toutes les erreurs sont dans `<sortie>/log.txt` ;
  le bilan final dans `<sortie>/report.txt`.

## Adapter le projet avec un LLM

Une documentation complète destinée aux assistants IA (architecture, dépendances,
pièges Windows déjà résolus, recettes d'extension, règles de non-régression) est
fournie dans [`docs/LLM_GUIDE.fr.md`](docs/LLM_GUIDE.fr.md) —
[english version](docs/LLM_GUIDE.md). Donnez ce fichier à votre LLM en même temps
que votre demande d'adaptation.

## Licence

Cette application : MIT (voir `LICENSE`). SeedVR2 (code + poids) : Apache 2.0 — © ByteDance.
