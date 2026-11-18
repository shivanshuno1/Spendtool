import json
import os
import sys
import unittest
from unittest.mock import patch
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import llm


class GeminiApiTests(unittest.TestCase):
    def test_call_uses_gemini_request_and_parses_response(self):
        response = {
            "candidates": [{"content": {"parts": [{"text": '[{"id":0,"c":1,"conf":0.9}]'}]}}],
            "usageMetadata": {"promptTokenCount": 17, "candidatesTokenCount": 8},
        }
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data)
            return BytesIO(json.dumps(response).encode())

        with patch("src.llm.urllib.request.urlopen", side_effect=fake_urlopen):
            text, usage = llm._call("gemini-2.5-flash", "system", "user", "test-key")

        self.assertIn("models/gemini-2.5-flash:generateContent", captured["url"])
        self.assertEqual(captured["headers"]["X-goog-api-key"], "test-key")
        self.assertEqual(captured["body"]["systemInstruction"]["parts"][0]["text"], "system")
        self.assertEqual(captured["body"]["contents"][0]["parts"][0]["text"], "user")
        self.assertEqual(text, '[{"id":0,"c":1,"conf":0.9}]')
        self.assertEqual(usage, {"input_tokens": 17, "output_tokens": 8})

    def test_available_requires_gemini_key_and_enabled_config(self):
        cfg = {"classify": {"llm": {"enabled": True}}}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertTrue(llm.available(cfg))
            self.assertFalse(llm.available({"classify": {"llm": {"enabled": False}}}))


if __name__ == "__main__":
    unittest.main()
