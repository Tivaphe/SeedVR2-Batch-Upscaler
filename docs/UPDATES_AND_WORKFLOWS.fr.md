# Lots fiables, inspection et mise à jour

## Avant un gros traitement

1. Choisissez les dossiers, le modèle et les réglages.
2. Ouvrez **Lots et mémoire** et cliquez sur **Vérifier le lot**.
   Le contrôle intervient aussi au démarrage, **avant** le chargement du modèle.
3. Consultez le nombre d’images, les conflits, les images invalides, la plus grande
   sortie et l’estimation des tableaux d’assemblage. Cette estimation n’inclut
   pas le modèle, les copies, la compression ni toute la RAM nécessaire.
4. L’entrée et la sortie doivent être distinctes. La sortie peut être un
   sous-dossier de l’entrée : elle est exclue du scan récursif. Un lien de sortie
   ne doit pas permettre d’écraser un original ou de sortir du dossier choisi.

Les fichiers invalides sont signalés par le contrôle puis comptabilisés comme
échecs individuels pendant le lot. Ils ne bloquent pas les autres images.

### Essai sur une zone

Après **Vérifier le lot**, choisissez une image dans **Essai sur une zone**.
Indiquez X/Y/largeur/hauteur dans les coordonnées de l’image après orientation
EXIF, puis cliquez sur **Tester cette zone**. Le test utilise les réglages courants
et écrit dans `<sortie>/.seedvr2_tests/<identifiant>/`, sans modifier la reprise du
lot principal. Un crop peut différer de l’image entière : le contexte du modèle
et le découpage en tuiles changent.

### Inspection à 100 %

Après une sauvegarde réussie, **Inspection synchronisée 100 %** affiche une zone
avant/après de 512 × 512 pixels. X/Y sont exprimés dans la sortie. À 1×, un pixel
image correspond à un pixel CSS sur un écran assez large ; 2× et 4× agrandissent
les pixels sans ajouter de détail. Les deux vues sont alignées dans le même
comparateur. L’original est rééchantillonné dans cette zone pour comparer à taille
égale. Sur un écran étroit, le navigateur peut réduire le composant.

## Reprise, erreurs et historique

- Chaque image est écrite dans un fichier temporaire du dossier de destination,
  synchronisé sur disque puis renommé. Une erreur d’encodage ne remplace pas une
  ancienne sortie valide.
- En mode I/O parallèle, une réussite n’est enregistrée qu’après la fin effective
  de l’écriture. Deux résultats au maximum attendent leur validation.
- `.seedvr2_resume.json` contient les chemins relatifs, les destinations, la seed,
  les tailles/dates de modification des fichiers et les statuts. Les paramètres
  et l’identité du modèle font partie de l’empreinte de configuration.
- Une source modifiée ou une sortie manquante/modifiée est retraitée. La vérification
  utilise taille + date nanoseconde, **pas un hachage complet du contenu**.
- Le manifeste reste présent après la fin du lot : relancer à réglages identiques
  ignore les sorties valides. Décochez la reprise ou utilisez `--no-resume` pour
  tout refaire. La politique de conflit continue à s’appliquer.
- **Uniquement les erreurs précédentes** retente les entrées en échec du manifeste
  compatible. Ce n’est pas une recherche d’erreurs dans tous les historiques.
- Les anciennes reprises (avant le format v2) sont ignorées. La nouvelle dérivation
  stable des seeds change les résultats par rapport aux anciennes versions.
  La seed est stable par chemin relatif, indépendamment de la position dans la file.
  Cela ne garantit pas une identité bit à bit entre matériels, bibliothèques ou
  tailles de tuiles différentes.
- `log.txt` est alimenté en continu ; `report.txt` reste le dernier bilan.
  `.seedvr2_history/` conserve un JSON par exécution avec réglages, bilan et détail
  des images. Le panneau **Historique** affiche les 20 dernières exécutions du
  dossier indiqué. Ces fichiers contiennent les chemins locaux : vérifiez-les
  avant de les partager.

## Plusieurs lots et sous-dossiers

**Ajouter à la file** mémorise un instantané des réglages courants. Préparez les
lots suivants puis utilisez **Lancer la file**. Si un lot tourne déjà, les lots
ajoutés s’enchaînent à sa fin. **Annuler** arrête le lot courant et vide les lots
encore en attente ; **Vider la file** laisse le lot courant continuer.

La file est en mémoire : elle n’est pas restaurée après fermeture de l’application.
La reprise de chaque lot, elle, est persistante dans son dossier de sortie.

**Inclure les sous-dossiers** conserve les chemins relatifs. Deux images portant
le même nom dans deux sous-dossiers restent distinctes. Le glisser-déposer copie
exactement tous les fichiers visibles dans une sélection temporaire, même si
Gradio les a stockés séparément. Les homonymes sont renommés sans écrasement.
Ces copies temporaires ne modifient jamais les originaux et peuvent être nettoyées
avec le dossier temporaire système une fois l’application fermée.

## Préréglages

**Exporter les réglages** crée un JSON portable ; **Importer un préréglage JSON**
remplit les contrôles. L’import ne change ni les dossiers ni l’option de nouvelle
 tentative des erreurs. Un modèle absent localement ne remplace pas la sélection
courante. Les chemins d’entrée/sortie ne sont pas exportés.

## Mémoire et dimensions

- **Auto** choisit une limite conservatrice selon la VRAM libre après chargement
  et le dernier profil mémorisé pour le GPU/modèle/réglages correspondants.
  Ce n’est pas un benchmark garantissant le réglage optimal.
- **Économe** force des tuiles de 192 px d’entrée au maximum.
- **Équilibré** utilise les contrôles de tuilage/offload choisis manuellement.
- **Rapide** désactive le swap CPU/GPU et nécessite davantage de VRAM ; ce n’est
  pas une garantie d’accélération.
- Après un manque de mémoire, le moteur active le tuilage puis réduit les tuiles
  jusqu’à 64 px. Si cela échoue encore, l’image est déclarée en erreur et le lot
  continue. Le chargement d’un modèle trop volumineux n’est pas résolu par le tuilage.
- Les tuiles sont assemblées à mesure, sans conserver toute leur liste en RAM.
  Au-delà de 256 Mio de tableaux flottants, l’assemblage utilise des fichiers
  temporaires système (16 octets par pixel de sortie). Il faut donc de l’espace
  sur le disque temporaire, pas seulement sur le disque de sortie. Les temporaires
  sont supprimés à la sortie normale ou en cas d’exception Python.
- L’image RGB finale, son éventuel alpha et l’encodage occupent toujours de la RAM.
  Cette version n’est pas un encodeur de fichiers de taille illimitée.
- Les dimensions finales sont `round(entrée × facteur)`, que le tuilage soit
  activé ou non. L’alignement interne du VAE ne change plus les dimensions livrées.

## CLI

```powershell
python app.py --cli --input photos --output resultats --recursive --dry-run
python app.py --cli --input photos --output resultats --recursive --memory-profile eco
python app.py --cli --input photos --output resultats --recursive --retry-failed
python app.py --cli --input essai --output resultats-essai --crop 100 100 256 256
```

Le crop CLI s’applique à toutes les images du dossier sélectionné ; utilisez un
petit dossier de test. Codes de sortie : 0 = succès, 1 = échec de traitement ou
finalisation, 2 = configuration CLI invalide, 130 = annulation utilisateur.

## Mise à jour Windows, avec ou sans Git

Fermez l’application, puis double-cliquez sur **`update.bat`**.

1. Le script interroge les branches publiques du dépôt GitHub (liste paginée).
2. Choisissez un numéro ou un nom exact ; `main` est proposé par défaut.
3. Vérifiez le nom et le commit affichés, puis confirmez `oui` ou `yes`.
4. Le code est actualisé, puis les dépendances de `.venv` sont mises à jour.
5. Le script exécute `pip check` et le diagnostic d’installation. Une erreur ne
   sera pas annoncée comme une mise à jour entièrement réussie.

**Il faut que la branche et les modifications aient été publiées sur GitHub.**
Une branche uniquement locale n’apparaît pas dans cette liste. N’installez que du
code de confiance : les scripts et dépendances de la branche choisie seront exécutés.
L’API publique peut limiter le nombre de requêtes ; en cas d’erreur réseau, réessayez
plus tard. Aucun jeton GitHub n’est demandé ni enregistré.

### Installation issue d’un ZIP, sans Git

Aucune commande Git n’est nécessaire. Le script télécharge l’archive du commit
sélectionné, vérifie les chemins, puis remplace les fichiers de code. Les modèles,
`config/`, `.venv`, `SeedVR/`, `input/`, `output/` et les fichiers personnels non
gérés sont préservés. Les fichiers écrasés sont sauvegardés dans
`.update-backups/<date>/`. Les suppressions ne concernent que les anciens fichiers
gérés par un précédent manifeste de mise à jour ; au premier passage, aucun
ancien fichier inconnu n’est supprimé.

En cas d’erreur pendant le remplacement, les fichiers sauvegardés sont restaurés.
Une panne électrique peut interrompre cette restauration ; le manifeste de secours
permet aussi une restauration manuelle :

```powershell
python scripts/update.py --restore ".update-backups\<date>"
```

La restauration concerne **le code seulement**, pas les paquets déjà modifiés dans
la venv. `pip-freeze.txt` dans la sauvegarde conserve la liste préalable des paquets.

### Installation Git

Git est requis si le dossier contient `.git`. Le script refuse les modifications
locales du code et les branches divergentes ; il ne fait ni reset forcé ni stash
silencieux. Il sauvegarde et préserve `config/settings.json`, récupère la branche
choisie et n’autorise qu’une avance rapide. `git-state.json` conserve la branche
et le commit de départ. Il ne modifie pas le dépôt officiel `SeedVR/`.

Si une ancienne branche ne contient pas l’outil de mise à jour, les fichiers
`update.bat` et `scripts/update.py` sont conservés pour permettre une mise à jour
ultérieure.

### Venv et CUDA

Les versions installées de `torch`, `torchvision` et `torchaudio` servent de
contraintes lors de l’installation des dépendances. Une incompatibilité doit
provoquer une erreur plutôt qu’un remplacement silencieux de la build CUDA.

La venv est créée si elle n’existe pas, mais **le script ne choisit pas une build
PyTorch à votre place** : si PyTorch est absent, il demande de terminer l’installation
CUDA adaptée au GPU. Le code peut donc être actualisé alors que le diagnostic de
la venv échoue. Le message final distingue ce cas. Une mise à jour de paquets pip
n’est pas transactionnelle : elle n’est pas annulée automatiquement en cas d’échec.

```bat
update.bat --branch main
update.bat --no-venv
```

L’outil conserve le moteur officiel et les poids ; ce n’est pas une mise à niveau
automatique de SeedVR ou des modèles. Après un crash de l’outil, un verrou
`.update.lock` peut rester : ne le supprimez qu’après avoir vérifié qu’aucune mise
à jour ne tourne encore.

## Tests

```powershell
python tests/selftest.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Dépendances CPU : Pillow, numpy et Gradio 5. Les tests utilisent un backend simulé.
Le modèle `docs/ci-tests.yml.example` prévoit leur exécution sous Windows et Linux,
Python 3.11/3.12. GitHub Actions n’est pas activé : pour l’activer, un mainteneur
autorisé peut copier ce modèle dans `.github/workflows/tests.yml`. Ils ne remplacent pas
une validation de qualité/performance avec SeedVR2 et un GPU réel.
