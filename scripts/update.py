"""Interactive Windows/ZIP updater. Standard library only; never downloads models.

Git checkouts use fast-forward-only git operations, refusing local code changes.
ZIP installations use a pinned GitHub archive and a rollback-capable file manifest.
Run only with the app closed. Do not import application code before updating it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

REPOSITORY = 'Tivaphe/SeedVR2-Batch-Upscaler'
API = f'https://api.github.com/repos/{REPOSITORY}'
PROTECTED = {'.git', '.venv', 'models', 'config', 'input', 'output', 'upscaled',
             'seedvr', 'seedvr2_repo', 'vendor', '.update-backups', '.update-state.json',
             '.update.lock', '__pycache__'}
MAX_ARCHIVE = 128 * 1024**2


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'SeedVR2-Batch-Updater', 'Accept': 'application/vnd.github+json'}), timeout=60)


def api_json(url):
    with request(url) as response:
        return json.load(response)


def branches():
    result = []
    page = 1
    while True:
        batch = api_json(f'{API}/branches?per_page=100&page={page}')
        if not isinstance(batch, list):
            raise ValueError('Invalid GitHub branch response')
        result.extend((b['name'], b['commit']['sha']) for b in batch)
        if len(batch) < 100:
            break
        page += 1
    return sorted(result, key=lambda b: (b[0] != 'main', b[0]))


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if (not name or not pure.parts or pure.is_absolute() or any(p in ('..', '') for p in pure.parts)
            or '\\' in name or ':' in name or '\x00' in name):
        raise ValueError(f'Unsafe archive/manifest path: {name!r}')
    # Windows reserved device names, trailing dots/spaces and alternate streams.
    devices = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
               *(f'LPT{i}' for i in range(1, 10))}
    for part in pure.parts:
        if part.endswith(('.', ' ')) or part.split('.')[0].upper() in devices:
            raise ValueError(f'Unsafe Windows path: {name!r}')
    return Path(*pure.parts)


def protected(path: Path) -> bool:
    return path.parts[0].casefold() in PROTECTED


def contained(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes installation (symlink?): {relative}')
    # Refuse symlink/junction targets even when they point inside the root.
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError(f'Linked update target: {relative}')
    return path


def unpack_archive(archive: Path, destination: Path) -> list[Path]:
    files = []
    seen = set()
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if sum(info.file_size for info in members) > MAX_ARCHIVE:
            raise ValueError('Archive exceeds expanded size limit (128 MiB).')
        roots = {PurePosixPath(info.filename).parts[0] for info in members}
        if len(roots) != 1:
            raise ValueError('Expected one GitHub archive root')
        for info in members:
            safe_relative(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Archive symlinks are not allowed')
            parts = PurePosixPath(info.filename).parts[1:]
            if not parts or info.is_dir():
                continue
            rel = safe_relative('/'.join(parts))
            if protected(rel):
                continue
            key = rel.as_posix().casefold()
            if key in seen:
                raise ValueError(f'Duplicate archive path: {rel}')
            seen.add(key)
            target = contained(destination, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(info) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            files.append(rel)
    for required in ('app.py', 'requirements.txt', 'seedvr2_upscaler/__init__.py'):
        if Path(required) not in files:
            raise ValueError(f'Not an application archive: missing {required}')
    return files


def snapshot_update(root: Path, staged: Path, new_files: list[Path], backup: Path,
                    branch: str, commit: str):
    """Replace managed code only; preserve unknown files. Roll back on write errors."""
    state_path = root / '.update-state.json'
    old = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    old_files = [safe_relative(name) for name in old.get('files', [])]
    for rel in new_files + old_files:
        if protected(rel):
            raise ValueError(f'Protected path in update manifest: {rel}')
    affected = set(new_files + old_files)
    existing = []
    backup.mkdir(parents=True, exist_ok=False)
    for rel in sorted(affected):
        target = contained(root, rel)
        if target.exists():
            if not target.is_file():
                raise ValueError(f'Expected a file, found directory: {rel}')
            saved = backup / 'files' / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            existing.append(rel)
    if state_path.exists():
        shutil.copy2(state_path, backup / 'previous-state.json')
    recovery = {'existing': [p.as_posix() for p in existing],
                'added': [p.as_posix() for p in affected if p not in existing],
                'branch': branch, 'commit': commit}
    (backup / 'recovery.json').write_text(json.dumps(recovery, indent=2), encoding='utf-8')
    try:
        for rel in new_files:
            target = contained(root, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Same-filesystem replacement, never expose a partially copied file.
            fd, temporary = tempfile.mkstemp(prefix='.update-', dir=target.parent)
            os.close(fd)
            try:
                shutil.copy2(staged / rel, temporary)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        for rel in set(old_files) - set(new_files):
            contained(root, rel).unlink(missing_ok=True)
        payload = {'version': 1, 'branch': branch, 'commit': commit,
                   'files': [p.as_posix() for p in new_files]}
        temporary = root / '.update-state.tmp'
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        os.replace(temporary, state_path)
    except BaseException:
        restore_snapshot(root, backup)
        raise


def restore_snapshot(root: Path, backup: Path):
    data = json.loads((backup / 'recovery.json').read_text(encoding='utf-8'))
    for name in data['added']:
        rel = safe_relative(name)
        if protected(rel):
            raise ValueError('Invalid recovery manifest')
        contained(root, rel).unlink(missing_ok=True)
    for name in data['existing']:
        rel = safe_relative(name)
        if protected(rel):
            raise ValueError('Invalid recovery manifest')
        target = contained(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup / 'files' / rel, target)
    previous = backup / 'previous-state.json'
    if previous.exists():
        shutil.copy2(previous, root / '.update-state.json')
    else:
        (root / '.update-state.json').unlink(missing_ok=True)


def git(root, *args, capture=False):
    result = subprocess.run(['git', '-C', str(root), *args], check=True,
                            text=True, encoding='utf-8', stdout=subprocess.PIPE if capture else None)
    return result.stdout.rstrip('\r\n') if capture else None


def git_update(root: Path, branch: str, commit: str, backup: Path):
    """Only clean checkouts (apart from tracked settings) and fast-forward merges."""
    if not shutil.which('git'):
        raise RuntimeError('Ce dossier contient .git : installez Git, ou utilisez une installation ZIP séparée.')
    settings = root / 'config/settings.json'
    settings_bytes = settings.read_bytes() if settings.exists() else None
    changes = git(root, 'status', '--porcelain', '--untracked-files=all', capture=True).splitlines()
    bootstrap = {'update.bat', 'scripts/update.py'}
    artifacts = {'.update.lock', '.update-state.json', '.update-backups/'}
    unexpected = [line for line in changes if line[3:] != 'config/settings.json'
                  and not (line.startswith('?? ') and (line[3:] in bootstrap | artifacts
                           or line[3:].startswith('.update-backups/')))]
    if unexpected:
        raise RuntimeError('Modifications locales : sauvegardez/committez-les avant la mise à jour.\n' + '\n'.join(unexpected))
    current = git(root, 'branch', '--show-current', capture=True)
    if not current:
        raise RuntimeError('Detached HEAD : choisissez une branche locale avant de mettre à jour.')
    backup.mkdir(parents=True, exist_ok=False)
    old_commit = git(root, 'rev-parse', 'HEAD', capture=True)
    (backup / 'git-state.json').write_text(json.dumps({'branch': current, 'commit': old_commit}), encoding='utf-8')
    if settings_bytes is not None:
        (backup / 'settings.json').write_bytes(settings_bytes)
    git(root, 'fetch', f'https://github.com/{REPOSITORY}.git', f'refs/heads/{branch}')
    fetched = git(root, 'rev-parse', 'FETCH_HEAD', capture=True)
    if fetched != commit:
        raise RuntimeError('La branche a changé pendant la sélection. Relancez la mise à jour.')
    try:
        if settings_bytes is not None:
            git(root, 'restore', '--source=HEAD', '--staged', '--worktree', '--', 'config/settings.json')
        local = subprocess.run(['git', '-C', str(root), 'show-ref', '--verify', '--quiet', f'refs/heads/{branch}'])
        if local.returncode == 0:
            # Check ancestry BEFORE switching; never reset a diverged branch.
            subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', branch, commit], check=True)
            git(root, 'switch', branch)
            git(root, 'merge', '--ff-only', commit)
        else:
            git(root, 'switch', '-c', branch, commit)
    finally:
        if settings_bytes is not None:
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings.write_bytes(settings_bytes)


def refresh_venv(root: Path, backup: Path):
    python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(root / '.venv')], check=True)
    frozen = subprocess.run([str(python), '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True)
    (backup / 'pip-freeze.txt').write_text(frozen.stdout, encoding='utf-8')
    # Pin installed torch/torchvision for this transaction; never select another CUDA build.
    installed = subprocess.run([str(python), '-c',
        "import importlib.metadata as m,json; "
        "print(json.dumps({d.metadata['Name'].lower(): d.version for d in m.distributions() "
        "if d.metadata['Name'].lower() in ('torch','torchvision','torchaudio')}))"],
        check=True, capture_output=True, text=True)
    versions = json.loads(installed.stdout)
    if 'torch' not in versions:
        raise RuntimeError('Aucune installation PyTorch détectée : le code est actualisé, '
                           'mais installez d’abord PyTorch CUDA adapté à votre GPU avec install.bat. '
                           'La mise à jour ne choisit pas une build CPU/CUDA à votre place.')
    constraint_lines = [f'{name}=={version}' for name, version in versions.items()]
    constraints = backup / 'torch-constraints.txt'
    constraints.write_text('\n'.join(constraint_lines), encoding='utf-8')
    command = [str(python), '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed',
               '-r', str(root / 'requirements.txt'), '-c', str(constraints)]
    if (root / 'requirements-inference.txt').exists():
        command.extend(['-r', str(root / 'requirements-inference.txt')])
    subprocess.run(command, check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    subprocess.run([str(python), str(root / 'check_install.py')], cwd=root, check=True)


def run_update(args):
    root = Path(args.root).resolve()
    if not (root / 'app.py').is_file():
        raise ValueError('Dossier application invalide.')
    if args.restore:
        if (root / '.git').exists():
            raise ValueError('Restauration ZIP uniquement. Pour Git, consultez git-state.json dans la sauvegarde.')
        restore_snapshot(root, Path(args.restore).resolve())
        print('Fichiers restaurés. La venv n’a pas été modifiée par la restauration.')
        return
    print('Fermez l’application et tous les lots avant de continuer. / Close the app first.')
    choices = branches()
    for index, (name, sha) in enumerate(choices, 1):
        print(f'{index:3}. {name} ({sha[:10]})')
    value = args.branch or input('Branche : numéro ou nom [main] / Branch: ').strip() or 'main'
    if not args.branch and value.isdigit():
        index = int(value) - 1
        if not 0 <= index < len(choices):
            raise ValueError('Numéro de branche invalide.')
        value = choices[index][0]
    selected = next((b for b in choices if b[0] == value), None)
    if selected is None:
        raise ValueError('Branche inconnue. Aucune modification effectuée.')
    branch, commit = selected
    print(f'Installation de {branch} @ {commit}. Les modèles, images, réglages et SeedVR sont conservés.')
    print('La branche sélectionnée contient du code qui sera exécuté. Choisissez une branche de confiance.')
    if input('Confirmer / Confirm [oui/yes] : ').strip().lower() not in {'oui', 'yes'}:
        print('Annulé. Aucun fichier modifié.')
        return
    backup = root / '.update-backups' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if (root / '.git').exists():
        git_update(root, branch, commit, backup)
    else:
        with tempfile.TemporaryDirectory(prefix='seedvr2-update-') as folder:
            folder = Path(folder)
            archive = folder / 'source.zip'
            url = f'https://codeload.github.com/{REPOSITORY}/zip/{commit}'
            with request(url) as source, archive.open('wb') as target:
                total = 0
                while chunk := source.read(1024**2):
                    total += len(chunk)
                    if total > MAX_ARCHIVE:
                        raise ValueError('Archive trop volumineuse.')
                    target.write(chunk)
            staged = folder / 'staged'
            staged.mkdir()
            files = unpack_archive(archive, staged)
            snapshot_update(root, staged, files, backup, branch, commit)
    print(f'Code actualisé. Sauvegarde : {backup}')
    if not args.no_venv:
        print('Actualisation venv (PyTorch installé conservé)…')
        try:
            refresh_venv(root, backup)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Le code a été mis à jour, mais la venv/diagnostic a échoué. '
                               f'Consultez la sortie ci-dessus et {backup}. '
                               'La venv n’est pas restaurée automatiquement ; relancez install.bat si nécessaire.') from exc
    print('Mise à jour terminée.' if not args.no_venv else 'Code mis à jour ; venv non actualisée (--no-venv).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--branch', help='Nom exact (pas un numéro). Toujours confirmé avant installation.')
    parser.add_argument('--no-venv', action='store_true')
    parser.add_argument('--restore', help='Dossier de sauvegarde ZIP à restaurer (code seulement).')
    args = parser.parse_args()
    lock = Path(args.root).resolve() / '.update.lock'
    try:
        with lock.open('x') as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        print('Une mise à jour est déjà en cours. Si elle a crashé, supprimez .update.lock après vérification.')
        return 1
    root = Path(args.root).resolve()
    bootstrap = {name: (root / name).read_bytes() for name in ('update.bat', 'scripts/update.py')
                 if (root / name).is_file()}
    try:
        run_update(args)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERREUR / ERROR : {exc}', file=sys.stderr)
        return 1
    finally:
        try:
            # Older branches may not include the updater. Keep the entry points
            # so ZIP and Git installs can still update again without reinstalling.
            for name, content in bootstrap.items():
                target = root / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())
