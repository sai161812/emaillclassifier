import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "preddict.py"


class ModelPathTests(unittest.TestCase):
    def test_artifacts_resolve_from_module_not_working_directory(self):
        spec = importlib.util.spec_from_file_location("spam_predictor", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        model = Mock()
        model.predict.return_value = [1]
        vectorizer = Mock()
        cleaner = SimpleNamespace(clean_text=lambda text: text.lower(), process_text=lambda text: text)
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch.dict(sys.modules, {"txtcleaner": cleaner}), patch("builtins.open", mock_open()) as opened, patch("pickle.load", side_effect=[model, vectorizer]):
                    spec.loader.exec_module(module)
                    self.assertEqual(module.predict_spam("Example"), "SPAM")
                    self.assertEqual([call.args[0] for call in opened.call_args_list], [MODULE_PATH.parent/"model/spam_classifier.pkl", MODULE_PATH.parent/"model/vectorizer.pkl"])
        finally:
            os.chdir(original)
