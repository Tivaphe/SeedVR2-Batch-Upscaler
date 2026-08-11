"""Tests unitaires rapides des modules sans GPU ni modèle.

Exécution :  python tests/selftest.py
Couvre : nommage (conservation exacte + conflits), registre des modèles,
tuilage (recouvrement + fusion), I/O image (formats, conservation des noms),
paramètres persistants, et un lot complet via un backend de test injecté.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seedvr2_upscaler.backend.base import UpscaleBackend  # noqa: E402
from seedvr2_upscaler.images import load_image, save_image  # noqa: E402
from seedvr2_upscaler.models import (  # noqa: E402
    ConflictPolicy,
    JobConfig,
    ModelInfo,
    ModelKind,
    OutputFormat,
    TilingConfig,
)
from seedvr2_upscaler.naming import apply_suffix, resolve_output_path  # noqa: E402
from seedvr2_upscaler.registry import classify_model  # noqa: E402
from seedvr2_upscaler.settings import Settings, load_settings, save_settings  # noqa: E402
from seedvr2_upscaler.tiling import compute_tiles, stitch  # noqa: E402
from seedvr2_upscaler.worker import BatchRunner, EventBus  # noqa: E402

from PIL import Image  # noqa: E402


def _fake_model(name: str, size: int = 1000) -> ModelInfo:
    return ModelInfo(path=Path(name), name=Path(name).stem,
                     kind=classify_model(Path(name)).kind,
                     quant=classify_model(Path(name)).quant, size_bytes=size)


class TestNaming(unittest.TestCase):
    def test_nom_conserve_par_defaut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = resolve_output_path(Path(tmp), "photo001.png", ConflictPolicy.RENAME)
            self.assertEqual(out.name, "photo001.png")

    def test_suffixe_desactive_par_defaut(self) -> None:
        self.assertEqual(apply_suffix("vacances", "_upscaled", False), "vacances")

    def test_suffixe_active(self) -> None:
        self.assertEqual(apply_suffix("vacances", "_x4", True), "vacances_x4")

    def test_conflit_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "photo001.png").write_bytes(b"x")
            out = resolve_output_path(Path(tmp), "photo001.png", ConflictPolicy.RENAME)
            self.assertEqual(out.name, "photo001 (1).png")

    def test_conflit_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "photo001.png").write_bytes(b"x")
            self.assertIsNone(resolve_output_path(Path(tmp), "photo001.png", ConflictPolicy.SKIP))

    def test_conflit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "photo001.png").write_bytes(b"x")
            out = resolve_output_path(Path(tmp), "photo001.png", ConflictPolicy.OVERWRITE)
            self.assertEqual(out.name, "photo001.png")

    def test_rename_deux_collisions_dans_le_lot(self) -> None:
        reserved: set[str] = set()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.png").write_bytes(b"x")
            p1 = resolve_output_path(Path(tmp), "a.png", ConflictPolicy.RENAME, reserved)
            p2 = resolve_output_path(Path(tmp), "a.png", ConflictPolicy.RENAME, reserved)
        self.assertEqual(p1.name, "a (1).png")
        self.assertEqual(p2.name, "a (2).png")


class TestRegistry(unittest.TestCase):
    def test_classification(self) -> None:
        cases = {
            "seedvr2_ema_3b.pth": (ModelKind.OFFICIAL, None),
            "seedvr2_ema_3b_fp16.safetensors": (ModelKind.FP16, None),
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors": (ModelKind.FP8, None),
            "seedvr2_ema_3b-Q4_K_M.gguf": (ModelKind.GGUF, "Q4_K_M"),
            "seedvr2_ema_3b-Q6_K.gguf": (ModelKind.GGUF, "Q6_K"),
            "seedvr2_ema_3b-Q8_0.gguf": (ModelKind.GGUF, "Q8_0"),
        }
        for name, (kind, quant) in cases.items():
            info = _fake_model(name)
            self.assertEqual((info.kind, info.quant), (kind, quant), name)


class TestTiling(unittest.TestCase):
    def test_couverture_complete(self) -> None:
        tiles = compute_tiles(1000, 700, 512, 64)
        covered = any(t.x0 == 0 and t.y0 == 0 for t in tiles)
        reaches_end = any(t.x1 == 1000 and t.y1 == 700 for t in tiles)
        self.assertTrue(covered and reaches_end)
        for t in tiles:
            self.assertLessEqual(t.width, 512)
            self.assertLessEqual(t.height, 512)

    def test_stitch_identity(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        base = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), "RGB")
        tiles = compute_tiles(400, 300, 200, 32)
        upscaled = [base.crop(t.crop_box()) for t in tiles]  # scale 1 = identité
        merged = stitch(tiles, upscaled, 400, 300, 1.0, 32)
        self.assertEqual(merged.size, base.size)
        diff = abs(np.asarray(merged, int) - np.asarray(base, int)).max()
        self.assertLessEqual(diff, 1)  # fusion quasi parfaite hors recouvrement


class TestImages(unittest.TestCase):
    def test_io_names_and_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src_dir = tmp / "input"
            (tmp / "output").mkdir()
            src_dir.mkdir()
            names = ["photo001.png", "vacances.jpg", "image_test.webp"]
            for n in names:
                Image.new("RGB", (64, 48), (123, 50, 200)).save(src_dir / n)
            for n in names:
                loaded = load_image(src_dir / n)
                dest = tmp / "output" / n  # nom identique à l'entrée
                save_image(loaded.image, dest,
                           OutputFormat.from_label(dest.suffix.lstrip(".")), 90)
                self.assertTrue(dest.exists())
                self.assertEqual(Image.open(dest).size, (64, 48))


class TestSettings(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            original = Settings(input_dir="/x", scale_choice="x8", quality=80)
            save_settings(original, path)
            loaded = load_settings(path)
            self.assertEqual(loaded.input_dir, "/x")
            self.assertEqual(loaded.scale_choice, "x8")
            self.assertEqual(loaded.quality, 80)

    def test_inconnu_ou_invalide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"foo": 1}', encoding="utf-8")
            self.assertEqual(load_settings(path), Settings())  # tolérant


class _FakeBackend(UpscaleBackend):
    """Double de test CPU (lanczos) injecté à la place du pipeline réel.

    L'ancien backend « démo » de l'application a été supprimé (l'app exige
    désormais le vrai SeedVR2) ; ce double reste local aux tests et ne fait
    pas partie du produit — il sert uniquement à valider le moteur de lot
    (nommage, formats, reprise, journal, rapport) sans GPU.
    """

    name = "fake-test"

    def load(self, model, log=print) -> None:
        self._model = model

    def _upscale_region(self, region, out_w, out_h, seed, log):
        return region.resize((out_w, out_h), Image.LANCZOS)


def _fake_backend_factory(name, options=None, models_dir=None) -> _FakeBackend:
    """Signature identique à backend.create_backend (injectée au BatchRunner)."""
    return _FakeBackend(options)


class TestI18n(unittest.TestCase):
    def test_parite_cles_fr_en(self) -> None:
        from seedvr2_upscaler.i18n import STRINGS

        self.assertEqual(set(STRINGS["fr"]), set(STRINGS["en"]),
                         "les tables FR/EN doivent couvrir exactement les mêmes clés")
        for lang, table in STRINGS.items():
            for key, text in table.items():
                self.assertIsInstance(text, str, f"{lang}:{key}")
                if key == "gpu.hint.unknown":
                    continue  # seule chaîne volontairement vide (pas de conseil)
                self.assertNotEqual(text.strip(), "", f"{lang}:{key} vide")

    def test_tr_format_et_repli(self) -> None:
        from seedvr2_upscaler.i18n import tr

        self.assertEqual(tr("en", "counter.images", done=1, total=2),
                         "📦 1/2 images")
        self.assertEqual(tr("zz", "btn.start"), "▶️ Démarrer")     # langue inconnue → FR
        self.assertEqual(tr("fr", "cle.inexistante"), "cle.inexistante")  # repli = clé
        self.assertIn("images", tr("en", "counter.images", done=0, total=0))


class TestSettingsMigration(unittest.TestCase):
    def test_anciens_libelles_fr_migres(self) -> None:
        migrated = Settings.from_dict({
            "scale_choice": "Personnalisé",
            "output_format": "Comme l'entrée",
            "conflict_policy": "Renommer (auto)",
            "backend": "demo",          # backend supprimé
            "language": "klingon",      # langue inconnue
        })
        self.assertEqual(migrated.scale_choice, "custom")
        self.assertEqual(migrated.output_format, "keep")
        self.assertEqual(migrated.conflict_policy, "rename")
        self.assertEqual(migrated.backend, "auto")
        self.assertEqual(migrated.language, "fr")

    def test_valeurs_canoniques_inchangees(self) -> None:
        kept = Settings.from_dict({"scale_choice": "x8", "output_format": "webp",
                                   "conflict_policy": "skip", "language": "en"})
        self.assertEqual((kept.scale_choice, kept.output_format,
                          kept.conflict_policy, kept.language),
                         ("x8", "webp", "skip", "en"))


class TestBatchRunner(unittest.TestCase):
    def test_progression_tuiles_relayee_au_bus(self) -> None:
        """Le worker relaie « tuile i/N » pendant le tuilage, puis réinitialise."""
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "input"
            src_dir.mkdir()
            # 300×200 avec des tuiles de 128 px → forcément plusieurs tuiles.
            Image.new("RGB", (300, 200), (40, 120, 200)).save(src_dir / "mosaique.png")

            job = JobConfig(
                input_dir=src_dir, output_dir=Path(tmp) / "output",
                model=_fake_model("fake.pth"), scale=2.0,
                tiling=TilingConfig(enabled=True, tile_size=128, overlap=32),
            )
            bus = EventBus()
            runner = BatchRunner(job, bus, backend_factory=_fake_backend_factory)
            runner.start()
            runner._thread.join(timeout=60)

            tile_events = [e.data for e in bus.drain() if e.kind == "tile"]
            self.assertTrue(any(d["total"] > 1 for d in tile_events), tile_events)
            self.assertTrue(any((d["index"], d["total"]) == (0, 0)
                                for d in tile_events), tile_events[-1])

    def test_lot_complet_noms_conserves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "input"
            out_dir = Path(tmp) / "output"
            src_dir.mkdir()
            for n in ("a.png", "b.jpg", "c.webp"):
                Image.new("RGB", (96, 64), (10, 200, 30)).save(src_dir / n)

            job = JobConfig(
                input_dir=src_dir, output_dir=out_dir,
                model=_fake_model("fake.pth"), scale=2.0,
                output_format=OutputFormat.PNG, quality=90,
                tiling=TilingConfig(enabled=False),
            )
            runner = BatchRunner(job, EventBus(),
                                 backend_factory=_fake_backend_factory)
            runner.start()
            runner._thread.join(timeout=60)

            # Format PNG demandé : le nom de base est conservé, l'extension suit.
            for n in ("a.png", "b.png", "c.png"):
                out = out_dir / n
                self.assertTrue(out.exists(), out)
                self.assertEqual(Image.open(out).size, (192, 128))
            self.assertTrue((out_dir / "log.txt").exists())
            self.assertTrue((out_dir / "report.txt").exists())
            report = (out_dir / "report.txt").read_text(encoding="utf-8")
            self.assertIn("Images réussies", report)

    def test_mode_keep_conserve_nom_et_extension(self) -> None:
        """Exemple du cahier des charges : photo001.png / vacances.jpg / image_test.webp."""
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "input"
            out_dir = Path(tmp) / "output"
            src_dir.mkdir()
            for n in ("photo001.png", "vacances.jpg", "image_test.webp"):
                Image.new("RGB", (80, 60), (50, 50, 200)).save(src_dir / n)

            job = JobConfig(
                input_dir=src_dir, output_dir=out_dir,
                model=_fake_model("fake.pth"), scale=2.0,
                output_format=OutputFormat.KEEP,   # « Comme l'entrée »
            )
            runner = BatchRunner(job, EventBus(),
                                 backend_factory=_fake_backend_factory)
            runner.start()
            runner._thread.join(timeout=60)

            for n in ("photo001.png", "vacances.jpg", "image_test.webp"):
                out = out_dir / n  # extension ET nom strictement identiques
                self.assertTrue(out.exists(), out)
                self.assertEqual(Image.open(out).size, (160, 128))  # arrondi /16

    def test_reprise_ignore_deja_faits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "input"
            out_dir = Path(tmp) / "output"
            src_dir.mkdir()
            out_dir.mkdir()
            for n in ("a.png", "b.png"):
                Image.new("RGB", (64, 64), (200, 10, 10)).save(src_dir / n)
            # a.png déjà présent → politique « ignorer ».
            Image.new("RGB", (64, 64), (0, 0, 0)).save(out_dir / "a.png")

            job = JobConfig(
                input_dir=src_dir, output_dir=out_dir,
                model=_fake_model("fake.pth"), scale=2.0,
                conflict_policy=ConflictPolicy.SKIP,
            )
            runner = BatchRunner(job, EventBus(),
                                 backend_factory=_fake_backend_factory)
            runner.start()
            runner._thread.join(timeout=60)

            # a.png (le placeholder 64x64) ne doit pas avoir été remplacé.
            self.assertEqual(Image.open(out_dir / "a.png").size, (64, 64))
            self.assertEqual(Image.open(out_dir / "b.png").size, (128, 128))


def main() -> int:
    try:
        import numpy, PIL  # noqa: F401
    except ImportError:
        print("numpy/Pillow requis : pip install numpy Pillow")
        return 1
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    shutil.rmtree  # (référence pour linters : non utilisé)
    sys.exit(main())
