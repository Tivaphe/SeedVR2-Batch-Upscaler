"""CPU regressions: python -m unittest discover -s tests -p 'test_*.py' -v"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from selftest import _fake_backend_factory, _fake_model, _FakeBackend
from seedvr2_upscaler.images import load_image, save_image
from seedvr2_upscaler.jobs import preflight
from seedvr2_upscaler.models import JobConfig, OutputFormat, TilingConfig
from seedvr2_upscaler.worker import BatchRunner, EventBus
from seedvr2_upscaler.tiling import compute_tiles, stitch


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'input'
        self.source.mkdir()
        self.output = self.root / 'output'
        self.job = JobConfig(self.source, self.output, _fake_model('fake.pth'),
                             scale=2, clear_cache=False, output_format=OutputFormat.KEEP)
        for name in ('a.png', 'b.png'):
            Image.new('RGB', (73, 59), (10, 20, 30)).save(self.source / name)

    def run_job(self, job=None, factory=_fake_backend_factory):
        runner = BatchRunner(job or self.job, EventBus(), backend_factory=factory)
        runner.start()
        runner.wait(10)
        self.assertFalse(runner._thread.is_alive())
        return runner

    def manifest(self):
        return json.loads((self.output / '.seedvr2_resume.json').read_text())

    def test_completed_manifest_and_exact_dimensions(self):
        runner = self.run_job()
        self.assertEqual(runner.exit_code, 0)
        self.assertEqual(runner.stats.succeeded, 2)
        with Image.open(self.output / 'a.png') as image:
            self.assertEqual(image.size, (146, 118))
        self.assertEqual(self.manifest()['records']['a.png']['status'], 'done')
        self.assertEqual(len(list((self.output / '.seedvr2_history').glob('*.json'))), 1)
        again = self.run_job()
        self.assertEqual(again.stats.succeeded, 0)
        self.assertEqual(again.stats.skipped, 2)
        self.assertIsNone(again._backend)  # no GPU load for an already completed job

    def test_missing_output_is_reprocessed(self):
        self.run_job()
        (self.output / 'b.png').unlink()
        again = self.run_job()
        self.assertEqual(again.stats.succeeded, 1)
        self.assertTrue((self.output / 'b.png').is_file())

    def test_source_change_invalidates_only_that_entry(self):
        self.run_job()
        Image.new('RGB', (23, 32)).save(self.source / 'a.png')
        again = self.run_job()
        self.assertEqual(again.stats.succeeded, 1)
        self.assertEqual(again.stats.skipped, 1)

    def test_changed_seed_invalidates_resume(self):
        self.run_job()
        again = self.run_job(dataclasses.replace(self.job, seed=123))
        self.assertEqual(again.stats.succeeded, 2)

    def test_seed_independent_of_remaining_list(self):
        runner = BatchRunner(self.job)
        seed = runner._seed(self.source / 'b.png')
        self.run_job()
        (self.output / 'b.png').unlink()
        self.run_job()
        self.assertEqual(self.manifest()['records']['b.png']['seed'], seed)

    def test_async_write_failure_is_not_success_and_retry_only_failed(self):
        original = save_image
        def broken(image, dest, *args):
            if dest.name == 'b.png':
                raise OSError('disk full')
            original(image, dest, *args)
        with patch('seedvr2_upscaler.worker.save_image', side_effect=broken):
            runner = self.run_job()
        self.assertEqual((runner.stats.succeeded, runner.stats.failed), (1, 1))
        self.assertEqual(runner.exit_code, 1)
        self.assertEqual(self.manifest()['records']['b.png']['status'], 'failed')
        again = self.run_job(dataclasses.replace(self.job, retry_failed=True))
        self.assertEqual((again.stats.succeeded, again.stats.skipped), (1, 1))

    def test_success_not_recorded_before_write_finishes(self):
        entered, release = threading.Event(), threading.Event()
        original = save_image
        def delayed(*args):
            entered.set()
            release.wait(5)
            original(*args)
        runner = BatchRunner(self.job, backend_factory=_fake_backend_factory)
        with patch('seedvr2_upscaler.worker.save_image', side_effect=delayed):
            runner.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(runner.stats.succeeded, 0)
                self.assertFalse((self.output / '.seedvr2_resume.json').exists())
                self.assertTrue((self.output / 'log.txt').read_text())
            finally:
                release.set()
                runner.wait(10)
        self.assertEqual(runner.stats.succeeded, 2)

    def test_recursive_preserves_tree_and_excludes_outputs(self):
        sub = self.source / 'nested'
        sub.mkdir()
        Image.new('RGB', (10, 10)).save(sub / 'a.png')
        job = dataclasses.replace(self.job, recursive=True, output_dir=self.source / 'upscaled')
        first = self.run_job(job)
        self.assertEqual(first.stats.total, 3)
        self.assertTrue((job.output_dir / 'nested/a.png').is_file())
        self.assertEqual(self.run_job(job).stats.total, 3)

    def test_empty_input_does_not_load_backend(self):
        for p in self.source.iterdir():
            p.unlink()
        factory = unittest.mock.Mock(side_effect=AssertionError('must not load'))
        runner = self.run_job(factory=factory)
        factory.assert_not_called()
        self.assertEqual(runner.exit_code, 1)

    def test_invalid_config_and_same_folder_rejected(self):
        for changes in ({'scale': 0}, {'scale': float('nan')}, {'quality': 101},
                        {'output_dir': self.source}, {'suffix': '../bad'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                preflight(dataclasses.replace(self.job, **changes))

    def test_crop_has_exact_dimensions(self):
        job = dataclasses.replace(self.job, crop=(2, 3, 20, 15),
                                   input_files=(self.source / 'a.png',))
        runner = self.run_job(job)
        self.assertEqual(runner.stats.succeeded, 1)
        with Image.open(self.output / 'a.png') as image:
            self.assertEqual(image.size, (40, 30))

    def test_oom_reduces_tiles_and_succeeds(self):
        class OOMBackend(_FakeBackend):
            def _upscale_region(self, region, *args):
                if max(region.size) > 64:
                    raise RuntimeError('CUDA out of memory')
                return super()._upscale_region(region, *args)
        runner = self.run_job(factory=lambda name, options, models: OOMBackend(options))
        self.assertEqual(runner.stats.succeeded, 2)
        self.assertEqual(runner._backend.options.tiling.tile_size, 64)

    def test_cancellation_during_tile_is_not_an_image_failure(self):
        bus = EventBus()
        runner = BatchRunner(dataclasses.replace(self.job, tiling=TilingConfig(True, 64, 16)),
                             bus, backend_factory=_fake_backend_factory)
        def cancel(index, total):
            runner.cancel()
        runner._tile_cb = cancel
        runner.start()
        runner.wait(10)
        self.assertEqual(runner.exit_code, 130)
        self.assertEqual(runner.stats.failed, 0)

    def test_pause_then_cancel_unblocks(self):
        runner = BatchRunner(self.job, backend_factory=_fake_backend_factory)
        runner.pause()
        runner.start()
        runner.cancel()
        runner.wait(10)
        self.assertFalse(runner._thread.is_alive())
        self.assertEqual(runner.exit_code, 130)

    def test_unload_even_after_load_failure(self):
        backend = _FakeBackend()
        backend.load = unittest.mock.Mock(side_effect=RuntimeError('bad model'))
        backend.unload = unittest.mock.Mock()
        runner = self.run_job(factory=lambda *a: backend)
        backend.unload.assert_called_once()
        self.assertEqual(runner.exit_code, 1)


class FidelityTests(unittest.TestCase):
    def test_orientation_and_png_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dest = Path(folder) / 'in.jpg', Path(folder) / 'out.png'
            exif = Image.Exif()
            exif[274] = 6
            exif[270] = 'description'
            Image.new('RGB', (30, 20)).save(src, exif=exif, icc_profile=b'test-profile')
            loaded = load_image(src)
            self.assertEqual(loaded.image.size, (20, 30))
            save_image(loaded.image, dest, OutputFormat.PNG, 95, loaded.exif_bytes, loaded.icc_profile)
            with Image.open(dest) as result:
                self.assertIsNone(result.getexif().get(274))
                self.assertEqual(result.getexif()[270], 'description')
                self.assertEqual(result.info['icc_profile'], b'test-profile')
                self.assertEqual(result.size, (20, 30))

    def test_palette_transparency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'palette.png'
            image = Image.new('P', (4, 4), 0)
            image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
            image.save(path, transparency=0)
            loaded = load_image(path)
            self.assertEqual(loaded.image.mode, 'RGBA')
            self.assertEqual(loaded.image.getchannel('A').getextrema(), (0, 0))

    def test_jpeg_extension_case_preserved(self):
        from seedvr2_upscaler.models import format_for_source
        self.assertEqual(format_for_source(Path('Photo.JPEG'), OutputFormat.KEEP),
                         (OutputFormat.JPEG, '.JPEG'))

    def test_atomic_write_keeps_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'out.png'
            path.write_bytes(b'previous')
            with patch.object(Image.Image, 'save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_image(Image.new('RGB', (8, 8)), path, OutputFormat.PNG, 90)
            self.assertEqual(path.read_bytes(), b'previous')
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_memmap_stitch_and_fractional_scale(self):
        tiles = compute_tiles(129, 97, 64, 16)
        def generated():
            for tile in tiles:
                yield Image.new('RGB', (round(tile.width * 1.3), round(tile.height * 1.3)), (12, 34, 56))
        image = stitch(tiles, generated(), 129, 97, 1.3, 16, memory_limit_bytes=1024)
        self.assertEqual(image.size, (round(129 * 1.3), round(97 * 1.3)))
        self.assertLessEqual(np.abs(np.asarray(image).astype(int) - [12, 34, 56]).max(), 1)

    def test_edge_tiles_not_tiny(self):
        tiles = compute_tiles(513, 513, 256, 64)
        self.assertTrue(all(t.width == 256 and t.height == 256 for t in tiles))


if __name__ == '__main__':
    unittest.main()
