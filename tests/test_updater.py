"""Offline updater tests: no repository/network mutations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import update


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'app'
        self.staged = Path(self.tmp.name) / 'stage'
        self.backup = self.root / '.update-backups' / 'one'
        self.root.mkdir()
        self.staged.mkdir()

    def test_invalid_paths(self):
        for path in ('../escape', '/absolute', 'C:/absolute', 'foo\\bar', 'name:stream', 'CON.txt', 'a./b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                update.safe_relative(path)

    def test_protected_data_and_unmanaged_files_survive(self):
        for name in ('config/settings.json', 'models/weights.pth', 'input/photo.png', 'personal.txt'):
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('keep')
        (self.root / 'app.py').write_text('old')
        (self.staged / 'app.py').write_text('new')
        update.snapshot_update(self.root, self.staged, [Path('app.py')], self.backup, 'feature/x', 'abc')
        self.assertEqual((self.root / 'app.py').read_text(), 'new')
        self.assertEqual((self.backup / 'files/app.py').read_text(), 'old')
        for name in ('config/settings.json', 'models/weights.pth', 'input/photo.png', 'personal.txt'):
            self.assertEqual((self.root / name).read_text(), 'keep')
        update.restore_snapshot(self.root, self.backup)
        self.assertEqual((self.root / 'app.py').read_text(), 'old')
        self.assertFalse((self.root / '.update-state.json').exists())

    def test_only_previously_managed_removed_files_are_deleted(self):
        (self.root / 'old.py').write_text('old')
        (self.root / 'personal.py').write_text('keep')
        (self.root / '.update-state.json').write_text(json.dumps({'files': ['old.py']}))
        (self.staged / 'new.py').write_text('new')
        update.snapshot_update(self.root, self.staged, [Path('new.py')], self.backup, 'main', 'abc')
        self.assertFalse((self.root / 'old.py').exists())
        self.assertTrue((self.root / 'personal.py').exists())
        update.restore_snapshot(self.root, self.backup)
        self.assertTrue((self.root / 'old.py').exists())
        self.assertFalse((self.root / 'new.py').exists())

    def test_replace_error_rolls_back(self):
        (self.root / 'app.py').write_text('old')
        (self.staged / 'app.py').write_text('new')
        with patch.object(update.os, 'replace', side_effect=OSError('locked')):
            with self.assertRaises(OSError):
                update.snapshot_update(self.root, self.staged, [Path('app.py')], self.backup, 'main', 'abc')
        self.assertEqual((self.root / 'app.py').read_text(), 'old')

    def test_archive_filters_protected_files(self):
        archive = self.root / 'archive.zip'
        with zipfile.ZipFile(archive, 'w') as package:
            for name in ('app.py', 'requirements.txt', 'seedvr2_upscaler/__init__.py', 'config/settings.json', '.git/config'):
                package.writestr('repo-sha/' + name, 'content')
        files = update.unpack_archive(archive, self.staged)
        self.assertEqual(len(files), 3)
        self.assertFalse((self.staged / 'config').exists())

    def test_archive_rejects_traversal_and_case_collisions(self):
        for names in (['repo/../escape'], ['repo/app.py', 'repo/APP.py']):
            archive = self.root / 'archive.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                for name in names:
                    package.writestr(name, 'content')
            with self.assertRaises(ValueError):
                update.unpack_archive(archive, self.staged)

    def test_manifest_cannot_delete_user_data(self):
        (self.root / '.update-state.json').write_text(json.dumps({'files': ['config/settings.json']}))
        with self.assertRaises(ValueError):
            update.snapshot_update(self.root, self.staged, [], self.backup, 'main', 'abc')

    def test_paginated_branches_main_first(self):
        first = [{'name': f'feature/{i}', 'commit': {'sha': str(i)}} for i in range(100)]
        with patch.object(update, 'api_json', side_effect=[first, [{'name': 'main', 'commit': {'sha': 'mainsha'}}]]):
            result = update.branches()
        self.assertEqual(len(result), 101)
        self.assertEqual(result[0], ('main', 'mainsha'))

    def test_git_status_preserves_leading_spaces(self):
        result = unittest.mock.Mock(stdout=' M app.py\n')
        with patch.object(update.subprocess, 'run', return_value=result):
            self.assertEqual(update.git(self.root, 'status', capture=True), ' M app.py')

    def test_git_dirty_checkout_refuses_before_changes(self):
        with patch.object(update.shutil, 'which', return_value='git'), \
             patch.object(update, 'git', return_value=' M app.py') as git:
            with self.assertRaisesRegex(RuntimeError, 'Modifications locales'):
                update.git_update(self.root, 'main', 'abc', self.backup)
            self.assertEqual(git.call_count, 1)

    def test_venv_update_constrains_existing_cuda_torch(self):
        import os
        python = self.root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        python.parent.mkdir(parents=True)
        python.touch()
        (self.root / 'requirements-inference.txt').touch()
        self.backup.mkdir(parents=True)
        results = [unittest.mock.Mock(stdout='torch==2.9.0+cu130\n'),
                   unittest.mock.Mock(stdout='{"torch": "2.9.0+cu130", "torchvision": "0.24.0+cu130"}'),
                   unittest.mock.Mock(), unittest.mock.Mock(), unittest.mock.Mock()]
        with patch.object(update.subprocess, 'run', side_effect=results) as run:
            update.refresh_venv(self.root, self.backup)
        constraints = (self.backup / 'torch-constraints.txt').read_text()
        self.assertIn('torch==2.9.0+cu130', constraints)
        install = run.call_args_list[2].args[0]
        self.assertIn('--upgrade', install)
        self.assertIn(str(self.root / 'requirements-inference.txt'), install)
        self.assertIn('-c', install)

    def test_missing_torch_does_not_install_cpu_build(self):
        import os
        python = self.root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        python.parent.mkdir(parents=True)
        python.touch()
        self.backup.mkdir(parents=True)
        with patch.object(update.subprocess, 'run', side_effect=[
                unittest.mock.Mock(stdout=''), unittest.mock.Mock(stdout='{}')]) as run:
            with self.assertRaisesRegex(RuntimeError, 'PyTorch'):
                update.refresh_venv(self.root, self.backup)
        self.assertEqual(run.call_count, 2)  # freeze/metadata only; no pip install


if __name__ == '__main__':
    unittest.main()
