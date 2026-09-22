from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import generator
from src.config import MODEL_DIR, load_generator_settings
from src.generation_context import VerifiedEvidenceItem, VerifiedEvidencePackage


class GeneratorConfigurationTests(unittest.TestCase):
    def test_defaults_use_7b_development_profile_and_portable_first_shard(self) -> None:
        settings = load_generator_settings({})
        self.assertEqual(Path(settings.model_path), MODEL_DIR / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
        self.assertEqual(settings.profile, "DEVELOPMENT_8GB")
        self.assertEqual(settings.context_size, 4096)
        self.assertEqual(settings.threads, 8)
        self.assertEqual(settings.batch_size, 128)
        self.assertEqual(settings.gpu_layers, 0)
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.top_p, 1.0)
        self.assertEqual(settings.max_tokens, 180)
        self.assertTrue(settings.use_mmap)
        self.assertFalse(settings.use_mlock)

    def test_every_generator_setting_is_configurable(self) -> None:
        settings = load_generator_settings({
            "GENERATOR_MODEL_PATH": "models/custom.gguf",
            "GENERATOR_CONTEXT_SIZE": "4096",
            "GENERATOR_THREADS": "3",
            "GENERATOR_BATCH_SIZE": "128",
            "GENERATOR_GPU_LAYERS": "2",
            "GENERATOR_TEMPERATURE": "0.1",
            "GENERATOR_TOP_P": "0.9",
            "GENERATOR_MAX_TOKENS": "90",
            "GENERATOR_USE_MMAP": "false",
            "GENERATOR_USE_MLOCK": "true",
        })
        self.assertEqual(Path(settings.model_path), MODEL_DIR / "custom.gguf")
        self.assertEqual(
            (settings.context_size, settings.threads, settings.batch_size, settings.gpu_layers),
            (4096, 3, 128, 2),
        )
        self.assertEqual((settings.temperature, settings.top_p, settings.max_tokens), (0.1, 0.9, 90))
        self.assertFalse(settings.use_mmap)
        self.assertTrue(settings.use_mlock)

    def test_loader_caches_single_model_by_runtime_configuration(self) -> None:
        fake_one, fake_two = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(generator, "Llama", side_effect=(fake_one, fake_two)) as constructor:
            one = Path(directory) / "model.gguf"
            one.touch()
            generator.GENERATOR_CACHE.clear()
            self.assertIs(generator._get_generator(str(one), 4096, 8, 128, 0, True, False), fake_one)
            self.assertIs(generator._get_generator(str(one), 8192, 12, 512, 0, True, False), fake_two)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(constructor.call_args_list[0].kwargs["model_path"], str(one))

    def test_model_choice_does_not_change_prompt_or_policy(self) -> None:
        package = VerifiedEvidencePackage(
            question="Explain the policy.",
            original_language="english",
            requested_entity=None,
            requested_field="policy",
            support_status="supported",
            evidence=(VerifiedEvidenceItem("doc.pdf", 1, "Policy", None, "Policy evidence.", "x", 1.0),),
        )
        blocks = ["[EVIDENCE 1]\nText: Policy evidence."]
        before = generator._messages(package, blocks)
        with patch.object(generator, "_generate", return_value="Grounded answer.") as generate:
            generator.generate_answer(package.question, evidence_package=package)
        after = generator._messages(package, blocks)
        self.assertEqual(before, after)

    def test_missing_model_error_names_path_and_configuration_variable(self) -> None:
        missing = str(MODEL_DIR / "missing.gguf")
        with self.assertRaisesRegex(FileNotFoundError, "GENERATOR_MODEL_PATH"):
            generator._get_generator(missing, 4096, 8, 128, 0)

    def test_missing_split_shard_is_reported_before_llama_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "model-00001-of-00002.gguf"
            first.touch()
            with self.assertRaisesRegex(FileNotFoundError, "00002"):
                generator._get_generator(str(first), 4096, 8, 128, 0)

    def test_full_32gb_profile_is_configuration_only(self) -> None:
        settings = load_generator_settings({"GENERATOR_PROFILE": "FULL_32GB"})
        self.assertEqual((settings.context_size, settings.threads, settings.batch_size), (8192, 12, 512))
        self.assertIn("7b-instruct-q4_k_m-00001-of-00002", settings.model_path)


if __name__ == "__main__":
    unittest.main()
