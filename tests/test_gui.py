"""Gradio construction and workflow callbacks, with a CPU fake backend."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from selftest import _fake_backend_factory
from seedvr2_upscaler import gui_gradio as gui
from seedvr2_upscaler.settings import Settings
from seedvr2_upscaler.worker import BatchRunner


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        models = self.root / 'models'
        models.mkdir()
        (models / 'fake.pth').write_bytes(b'fake')
        self.source = self.root / 'input'
        self.source.mkdir()
        Image.new('RGB', (80, 60)).save(self.source / 'a.png')
        self.session = gui.GuiSession(models)
        settings = Settings(input_dir=str(self.source), output_dir=str(self.root / 'output'),
                            model_choice=next(iter(self.session.models_by_label)), clear_cache=False)
        self.values = [getattr(settings, key) for key in gui.SETTING_KEYS]
        self.save_patch = patch.object(gui, 'save_settings')
        self.save_patch.start()
        self.addCleanup(self.save_patch.stop)
        self.backend_patch = patch.object(gui, 'BatchRunner',
            side_effect=lambda job, bus: BatchRunner(job, bus, backend_factory=_fake_backend_factory))
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    def test_build_both_languages(self):
        for language in ('fr', 'en'):
            self.session.language = language
            self.session.text_registry.clear()
            app = gui.build_app(session=self.session)
            self.assertGreater(len(app.blocks), 100)

    def test_check_returns_file_selection(self):
        summary, files = gui._check_job(self.session, *self.values)
        self.assertEqual(json.loads(summary)['images'], 1)
        self.assertEqual(files['choices'], [str(self.source / 'a.png')])

    def test_queue_snapshots_and_cancel_clears(self):
        gui._enqueue_job(self.session, *self.values)
        self.values[gui.SETTING_KEYS.index('scale_choice')] = 'x2'
        self.assertEqual(self.session.pending_jobs[0].scale, 4)
        gui._start_queue(self.session)
        self.session.runner.wait(10)
        self.assertEqual(self.session.runner.stats.succeeded, 1)
        gui._enqueue_job(self.session, *self.values)
        gui._safe_call(self.session, 'cancel')
        self.assertFalse(self.session.pending_jobs)

    def test_crop_and_pixel_inspection(self):
        gui._test_crop(self.session, str(self.source / 'a.png'), 1, 2, 16, 16, *self.values)
        self.session.runner.wait(10)
        gui._on_tick(self.session)
        self.assertIsNotNone(self.session.latest_result)
        result = gui._inspect_result(self.session, 0, 0, 1)
        self.assertEqual([im.size for im in result], [(512, 512), (512, 512)])
        self.assertIn('.seedvr2_tests', self.session.latest_result['destination'])

    def test_export_import_does_not_override_directories(self):
        path = Path(gui._export_preset(*self.values))
        self.addCleanup(shutil.rmtree, path.parent)
        imported = gui._import_preset(self.session, path)
        self.assertNotIn('value', imported[0])
        self.assertNotIn('value', imported[1])
        self.assertEqual(imported[gui.SETTING_KEYS.index('scale_choice')]['value'], 'x4')

    def test_drop_uses_all_files_even_with_same_name(self):
        second = self.root / 'another'
        second.mkdir()
        Image.new('RGB', (10, 10)).save(second / 'a.png')
        value = gui._on_drop(self.session, [str(self.source / 'a.png'), str(second / 'a.png')])
        folder = Path(value['value'])
        self.addCleanup(shutil.rmtree, folder)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ['a (1).png', 'a.png'])
        self.assertEqual(gui._on_drop(self.session, [])['value'], '')


if __name__ == '__main__':
    unittest.main()
