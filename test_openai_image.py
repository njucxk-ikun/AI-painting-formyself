"""OpenAI图像客户端的本地测试。"""
import base64
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from api_clients import OpenAIImageClient, ImageGenerationError
from PIL import Image


_png = BytesIO()
Image.new("RGB", (24, 24), "orange").save(_png, format="PNG")
PNG_BYTES = _png.getvalue()


class FakeImages:
    def __init__(self):
        self.generate_args = None
        self.edit_args = None
        self.generate_calls = 0
        self.edit_calls = 0

    @staticmethod
    def _response(count):
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=encoded, url=None) for _ in range(count)]
        )

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.generate_args = kwargs
        return self._response(kwargs["n"])

    def edit(self, **kwargs):
        self.edit_calls += 1
        self.edit_args = kwargs
        return self._response(kwargs["n"])


class OpenAIImageClientTests(unittest.TestCase):
    def make_client(self, output_dir):
        fake_images = FakeImages()
        fake_client = SimpleNamespace(images=fake_images)
        client = OpenAIImageClient(
            api_key="test-key",
            model="test-image-model",
            quality="medium",
            output_dir=output_dir,
            client=fake_client,
        )
        return client, fake_images

    def test_generates_and_saves_two_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client, fake_images = self.make_client(temp_dir)
            paths = client.generate_images("一只纸雕猫", num_images=2, aspect_ratio="16:9")

            self.assertEqual(len(paths), 2)
            self.assertEqual(fake_images.generate_args["size"], "3840x2160")
            self.assertEqual(fake_images.generate_args["quality"], "medium")
            self.assertEqual(fake_images.generate_args["n"], 1)
            self.assertEqual(fake_images.generate_calls, 2)
            self.assertTrue(all(Path(path).read_bytes() == PNG_BYTES for path in paths))

    def test_reference_images_use_edit_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.png"
            reference.write_bytes(PNG_BYTES)
            client, fake_images = self.make_client(temp_dir)

            client.generate_images(
                "保留角色造型并改为水彩风",
                num_images=1,
                reference_images=[(str(reference), "角色参考")],
            )

            self.assertIsNotNone(fake_images.edit_args)
            self.assertEqual(fake_images.edit_args["input_fidelity"], "high")
            self.assertIn("角色参考", fake_images.edit_args["prompt"])

    def test_missing_api_key_has_clear_error(self):
        client = OpenAIImageClient(api_key=None)
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            client.generate_images("测试")

    def test_aspect_ratio_mapping(self):
        self.assertEqual(OpenAIImageClient.size_for_aspect_ratio("1:1"), "2048x2048")
        self.assertEqual(OpenAIImageClient.size_for_aspect_ratio("9:16"), "2160x3840")
        self.assertEqual(OpenAIImageClient.size_for_aspect_ratio("4:3"), "3200x2400")

    def test_partial_failure_preserves_first_image_and_attempt_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client, fake = self.make_client(temp_dir)
            original = fake.generate
            def fail_second(**kwargs):
                if fake.generate_calls:
                    raise RuntimeError("synthetic service failure")
                return original(**kwargs)
            fake.generate = fail_second
            with self.assertRaises(ImageGenerationError) as caught:
                client.generate_images("橘猫")
            self.assertEqual(caught.exception.api_calls, 2)
            self.assertEqual(len(caught.exception.saved_paths), 1)
            self.assertTrue(Path(caught.exception.saved_paths[0]).exists())

    def test_non_image_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client, fake = self.make_client(temp_dir)
            fake.generate = lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"not-image").decode(), url=None)])
            with self.assertRaises(ImageGenerationError) as caught:
                client.generate_images("test", num_images=1)
            self.assertEqual(caught.exception.saved_paths, [])


if __name__ == "__main__":
    unittest.main()
