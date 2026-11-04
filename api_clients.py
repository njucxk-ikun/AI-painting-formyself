"""
API客户端封装
"""
import base64
from contextlib import ExitStack
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Optional
from pathlib import Path
import httpx
from openai import OpenAI
from PIL import Image


from deepseek_client import DeepSeekClient


class ImageGenerationError(RuntimeError):
    """A generation may have a successful first image worth preserving."""
    def __init__(self, message, saved_paths, api_calls):
        super().__init__(message)
        self.saved_paths = saved_paths
        self.api_calls = api_calls


class OpenAIImageClient:
    """OpenAI Images API客户端。"""

    SIZE_BY_ASPECT_RATIO = {
        "1:1": "2048x2048",
        "16:9": "3840x2160",
        "4:3": "3200x2400",
        "9:16": "2160x3840",
        "3:4": "2400x3200",
    }

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        model: str = "gpt-image-2",
        quality: str = "high",
        response_format: str = "url",
        use_env_proxy: bool = True,
        output_dir: str = "workspace/temp_outputs",
        client=None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.client = client or (
            OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(trust_env=use_env_proxy, timeout=600),
                max_retries=0,
            )
            if api_key
            else None
        )
        self.model = model
        self.quality = quality
        self.response_format = response_format
        self.use_env_proxy = use_env_proxy
        self.output_dir = Path(output_dir)

    @classmethod
    def size_for_aspect_ratio(cls, aspect_ratio: str) -> str:
        """将产品画幅映射为OpenAI Images API支持的输出尺寸。"""
        return cls.SIZE_BY_ASPECT_RATIO.get(aspect_ratio, "1024x1024")

    def generate_images(
        self,
        prompt: str,
        num_images: int = 2,
        aspect_ratio: str = "1:1",
        resolution: Optional[str] = None,
        reference_images: List[Tuple[str, str]] = None,
    ) -> List[str]:
        """生成图片；提供参考图时自动改用Images编辑接口。"""
        if not self.client:
            raise ValueError("未配置OPENAI_API_KEY，请先在.env中填写OpenAI API密钥")
        if not prompt.strip():
            raise ValueError("图像提示词不能为空")
        if num_images < 1:
            raise ValueError("生成数量必须大于0")

        request_args = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": resolution or self.size_for_aspect_ratio(aspect_ratio),
            "quality": self.quality,
            "output_format": "png",
            "response_format": self.response_format,
        }

        saved_paths = []
        api_calls = 0
        try:
            if reference_images:
                role_notes = "；".join(
                    f"参考图{i + 1}用途：{role}"
                    for i, (_, role) in enumerate(reference_images)
                )
                request_args["prompt"] = f"{prompt}\n\n参考图说明：{role_notes}"
                request_args["input_fidelity"] = "high"

            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for index in range(1, num_images + 1):
                api_calls += 1
                if reference_images:
                    with ExitStack() as stack:
                        edit_args = dict(request_args)
                        edit_args["image"] = [
                            stack.enter_context(open(image_path, "rb"))
                            for image_path, _ in reference_images
                        ]
                        response = self.client.images.edit(**edit_args)
                else:
                    response = self.client.images.generate(**request_args)

                if not response.data:
                    continue
                image = response.data[0]
                image_base64 = getattr(image, "b64_json", None)
                image_url = getattr(image, "url", None)
                if image_base64:
                    image_bytes = base64.b64decode(image_base64)
                elif image_url:
                    download_response = httpx.get(
                        image_url,
                        headers={"Accept": "image/*", "User-Agent": "Mozilla/5.0"},
                        timeout=180,
                        follow_redirects=True,
                        trust_env=self.use_env_proxy,
                    )
                    download_response.raise_for_status()
                    image_bytes = download_response.content
                else:
                    continue
                with Image.open(BytesIO(image_bytes)) as decoded:
                    decoded.verify()
                filepath = self.output_dir / f"openai_{timestamp}_{index}.png"
                filepath.write_bytes(image_bytes)
                saved_paths.append(str(filepath))

            if len(saved_paths) != num_images:
                raise RuntimeError(
                    f"图像服务返回了{len(saved_paths)}张有效图片，预期为{num_images}张"
                )
            return saved_paths
        except Exception as e:
            raise ImageGenerationError(
                f"图像生成未完成（{type(e).__name__}），已保留{len(saved_paths)}张图片，可重试。",
                saved_paths, api_calls,
            ) from None

    def get_usage_info(self) -> dict:
        """返回当前配置；API本身不提供余额查询。"""
        return {
            "success": bool(self.api_key),
            "message": "当前使用xcode.best兼容图像服务，用量请在供应商控制台查看",
            "console_url": "https://xcode.best",
            "base_url": self.base_url,
            "model": self.model,
            "quality": self.quality,
            "response_format": self.response_format,
        }
