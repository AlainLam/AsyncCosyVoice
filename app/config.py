import os
from functools import lru_cache
from typing import Literal

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


MAX_UPLOAD_SIZE_MB = 30
MIN_REF_AUDIO_DURATION_SEC = 3.0
MAX_REF_AUDIO_DURATION_SEC = 30.0


PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT_DIR = PROJECT_ROOT_DIR / "CosyVoice"
MATCHA_TTS_ROOT_DIR = COSYVOICE_ROOT_DIR / "third_party" / "Matcha-TTS"
_DEFAULT_MODEL_DIR = COSYVOICE_ROOT_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
_DEFAULT_TTSFRD_DIR = COSYVOICE_ROOT_DIR / "pretrained_models" / "CosyVoice-ttsfrd"


class AppSettings(
    BaseSettings,
    cli_parse_args=True,
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    """Runtime configuration for the app server."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: Literal[
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    ] = Field(default="info")

    model_dir: str = Field(
        default=str(_DEFAULT_MODEL_DIR), description="Model directory path"
    )
    ttsfrd_dir: str = Field(
        default=str(_DEFAULT_TTSFRD_DIR), description="TTSFRD model directory path"
    )
    fp16: bool = Field(
        default=False,
        description="Enable float16 autocast/TRT precision for CosyVoice flow and hift modules."
        "CosyVoice officially recommends maintaining fp32 usage on CosyVoice3. "
        "Please evaluate whether this feature must be enabled based on your requirements and test results.",
    )
    trt_concurrent: int = Field(
        default=1,
        ge=1,
        description="Number of concurrent inference requests to allow when using TensorRT. Set to `1` to disable concurrency and avoid potential issues with non-thread-safe TensorRT engine contexts.",
    )
    initial_token_hop_len: int = Field(
        default=15,
        ge=8,
        le=25,
        description="Number of LLM tokens to accumulate before synthesising "
        "the first streaming audio chunk. The second chunk then realigns "
        "to the 25-token flow grid.",
    )

    # vLLM EngineArgs passed via app layer
    vllm_gpu_memory_utilization: float = Field(default=0.20, gt=0.0, le=1.0)
    vllm_block_size: int | None = Field(default=None, ge=1)
    vllm_swap_space: float | None = Field(default=None, ge=0.0)
    vllm_enforce_eager: bool = Field(default=False)
    vllm_dtype: str = Field(default="bfloat16")
    vllm_kv_cache_dtype: str | None = Field(default=None)
    vllm_disable_log_stats: bool = Field(default=False)
    vllm_tensor_parallel_size: int | None = Field(default=None, ge=1)
    vllm_max_model_len: int = Field(default=16384, ge=1)

    # vLLM runtime envs configured at app layer
    vllm_logging_level: str | None = Field(default=None)
    vllm_worker_multiproc_method: str | None = Field(default=None)
    vllm_use_triton_flash_attn: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="COSYVOICE_",
        extra="ignore",
    )

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def ttsfrd_path(self) -> Path:
        ttsfrd_path = Path(self.ttsfrd_dir)
        if not ttsfrd_path.is_absolute():
            ttsfrd_path = self.model_path.parent / ttsfrd_path
        return ttsfrd_path

    @property
    def ttsfrd_resource_dir(self) -> Path:
        return self.ttsfrd_path / "resource"

    def build_vllm_kwargs(self) -> dict[str, object]:
        vllm_kwargs = {
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "block_size": self.vllm_block_size,
            "swap_space": self.vllm_swap_space,
            "enforce_eager": self.vllm_enforce_eager,
            "dtype": self.vllm_dtype,
            "kv_cache_dtype": self.vllm_kv_cache_dtype,
            "disable_log_stats": self.vllm_disable_log_stats,
            "tensor_parallel_size": self.vllm_tensor_parallel_size,
            "max_model_len": self.vllm_max_model_len,
        }
        return {key: value for key, value in vllm_kwargs.items() if value is not None}

    def apply_vllm_env(self) -> None:
        vllm_env = {
            "VLLM_LOGGING_LEVEL": self.vllm_logging_level,
            "VLLM_WORKER_MULTIPROC_METHOD": self.vllm_worker_multiproc_method,
            "VLLM_USE_TRITON_FLASH_ATTN": self.vllm_use_triton_flash_attn,
        }
        for key, value in vllm_env.items():
            if value is None:
                continue
            os.environ[key] = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Load settings from env vars and CLI args.

    Priority: CLI args override environment variables.
    """
    return AppSettings()
