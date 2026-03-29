# Copyright (c) 2026 Alain Lam
# SPDX-License-Identifier: Apache-2.0
#
# Derived in part from FunAudioLLM/CosyVoice (Apache-2.0).
# Upstream references:
# - cosyvoice/cli/cosyvoice.py
# Modified for async model loading and orchestration.

"""Top-level async CosyVoice facade for loading and TTS orchestration."""

import logging
import os
import time
from collections.abc import AsyncGenerator
from os import PathLike
from typing import Any, BinaryIO

import torch
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.cli.model import CosyVoice2Model, CosyVoice3Model
from cosyvoice.utils.class_utils import get_model_type

from app.cosyvoice.frontend import CosyVoiceFrontend
from app.cosyvoice.llm import AsyncLLM
from app.cosyvoice.model import AsyncModel
from app.cosyvoice.types import ModelVersion, TTSMode

_logger = logging.getLogger(__name__)

_V3_SILENT_TOKENS = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

_SPEECH_TOKENIZER = {
    ModelVersion.V2: "speech_tokenizer_v2.onnx",
    ModelVersion.V3: "speech_tokenizer_v3.onnx",
}

_YAML_FILENAMES = {
    ModelVersion.V2: "cosyvoice2.yaml",
    ModelVersion.V3: "cosyvoice3.yaml",
}


class AsyncCosyVoice:
    """Application-level facade for loading models and synthesizing speech."""

    def __init__(
        self,
        *,
        model: AsyncModel,
        frontend: CosyVoiceFrontend,
        sample_rate: int,
        model_dir: str,
        version: ModelVersion,
        device: torch.device,
    ) -> None:
        self.model = model
        self.frontend = frontend
        self.sample_rate = sample_rate
        self.model_dir = model_dir
        self.version = version
        self.device = device

    @classmethod
    def load(
        cls,
        model_dir: str,
        *,
        load_trt: bool = False,
        fp16: bool = False,
        trt_concurrent: int = 1,
        vllm_kwargs: dict[str, object] | None = None,
        initial_token_hop_len: int = 15,
        ttsfrd_resource_dir: str | None = None,
    ) -> "AsyncCosyVoice":
        """Load model weights and return a ready-to-use async facade."""
        version = _detect_version(model_dir)
        yaml_path = os.path.join(model_dir, _YAML_FILENAMES[version])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _logger.info("Loading %s from %s", version.name, model_dir)
        with open(yaml_path, "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
                },
            )

        expected_type = (
            CosyVoice3Model if version == ModelVersion.V3 else CosyVoice2Model
        )
        detected_type = get_model_type(configs)
        if detected_type != expected_type:
            raise TypeError(
                f"Expected {expected_type.__name__} but config yields "
                f"{detected_type.__name__} in {model_dir}"
            )

        if not torch.cuda.is_available():
            if fp16 or load_trt:
                _logger.warning("No CUDA device — disabling fp16 and TRT")
            fp16 = False
            load_trt = False

        frontend = CosyVoiceFrontend(
            model_version=version,
            device=device,
            get_tokenizer=configs["get_tokenizer"],
            feat_extractor=configs["feat_extractor"],
            campplus_model=os.path.join(model_dir, "campplus.onnx"),
            speech_tokenizer_model=os.path.join(model_dir, _SPEECH_TOKENIZER[version]),
            spk_info_path=os.path.join(model_dir, "spk2info.pt"),
            allowed_special=configs["allowed_special"],
            ttsfrd_resource_dir=ttsfrd_resource_dir,
        )
        sample_rate: int = configs["sample_rate"]

        llm_module: torch.nn.Module = configs["llm"]
        flow_module: torch.nn.Module = configs["flow"]
        hift_module: torch.nn.Module = configs["hift"]

        _load_weights(llm_module, flow_module, hift_module, model_dir, device)

        if load_trt:
            _load_trt(flow_module, model_dir, fp16, trt_concurrent, version)

        llm = AsyncLLM(llm_module=llm_module, version=version, device=device)
        llm.load_vllm(os.path.join(model_dir, "vllm"), **(vllm_kwargs or {}))

        model = AsyncModel(
            llm=llm,
            flow=flow_module,
            hift=hift_module,
            version=version,
            device=device,
            fp16=fp16,
            initial_token_hop_len=initial_token_hop_len,
            silent_tokens=_V3_SILENT_TOKENS if version == ModelVersion.V3 else [],
        )

        del configs
        _logger.info(
            "AsyncCosyVoice loaded: version=%s, sample_rate=%d, initial_token_hop_len=%d",
            version.name,
            sample_rate,
            initial_token_hop_len,
        )

        return cls(
            model=model,
            frontend=frontend,
            sample_rate=sample_rate,
            model_dir=model_dir,
            version=version,
            device=device,
        )

    def list_available_spks(self) -> list[str]:
        return list(self.frontend.spk2info.keys())

    def register_speaker(
        self,
        spk_id: str,
        ref_wav: str | PathLike[str] | BinaryIO,
        ref_text: str = "",
    ) -> None:
        if not spk_id:
            raise ValueError("spk_id must be a non-empty string")
        self.frontend.register_speaker(spk_id, ref_wav, ref_text)

    def save_spkinfo(self, path: str | None = None) -> None:
        self.frontend.save_speaker(path)

    async def synthesize(
        self,
        *,
        text: str,
        voice_id: str,
        mode: TTSMode = TTSMode.ZERO_SHOT,
        instruction: str | None = None,
        stream: bool = True,
        speed: float = 1.0,
        text_frontend: bool = True,
        split_sentences: bool = True,
    ) -> AsyncGenerator[dict[str, torch.Tensor], None]:
        segments = self.frontend.text_normalize(
            text,
            split=split_sentences,
            text_frontend=text_frontend,
        )
        for segment in segments:
            segment_text = str(segment)

            model_input = self.frontend.build_tts_input(
                spk_id=voice_id,
                text=segment_text,
                mode=mode,
                instruction=instruction,
            )

            start = time.time()
            _logger.info("synthesis text %s", segment_text)
            async for output in self.model.tts(
                model_input=model_input,
                stream=stream,
                speed=speed,
            ):
                speech_len = output["tts_speech"].shape[1] / self.sample_rate
                elapsed = time.time() - start
                _logger.info(
                    "yield speech len %.3f, rtf %.3f, latency %.3f sec",
                    speech_len,
                    elapsed / speech_len if speech_len > 0 else 0.0,
                    elapsed,
                )
                yield output
                start = time.time()

    async def synthesize_vc(
        self,
        *,
        source_wav: str | PathLike[str] | BinaryIO,
        prompt_wav: str | PathLike[str] | BinaryIO,
        stream: bool = True,
        speed: float = 1.0,
    ) -> AsyncGenerator[dict[str, torch.Tensor], None]:
        """Voice conversion: re-synthesize ``source_wav`` in the timbre of ``prompt_wav``."""
        model_input = self.frontend.build_vc_model_input(source_wav, prompt_wav)
        start = time.time()
        async for output in self.model.tts(
            model_input=model_input,
            stream=stream,
            speed=speed,
        ):
            speech_len = output["tts_speech"].shape[1] / self.sample_rate
            elapsed = time.time() - start
            _logger.info(
                "vc yield speech len %.3f, rtf %.3f, latency %.3f sec",
                speech_len,
                elapsed / speech_len if speech_len > 0 else 0.0,
                elapsed,
            )
            yield output
            start = time.time()


def _detect_version(model_dir: str) -> ModelVersion:
    if os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
        return ModelVersion.V3
    if os.path.exists(os.path.join(model_dir, "cosyvoice2.yaml")):
        return ModelVersion.V2
    raise TypeError(
        f"No cosyvoice2.yaml or cosyvoice3.yaml found in {model_dir}. "
        "CosyVoice V1 is not supported."
    )


def _load_weights(
    llm: torch.nn.Module,
    flow: torch.nn.Module,
    hift: torch.nn.Module,
    model_dir: str,
    device: torch.device,
) -> None:
    llm.load_state_dict(
        torch.load(
            os.path.join(model_dir, "llm.pt"),
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )
    llm.to(device).eval()

    flow.load_state_dict(
        torch.load(
            os.path.join(model_dir, "flow.pt"),
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )
    flow.to(device).eval()

    hift_state_dict = {
        key.replace("generator.", ""): value
        for key, value in torch.load(
            os.path.join(model_dir, "hift.pt"),
            map_location=device,
            weights_only=True,
        ).items()
    }
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.to(device).eval()

    _logger.info("Model weights loaded from %s", model_dir)


def _load_trt(
    flow: Any,
    model_dir: str,
    fp16: bool,
    trt_concurrent: int,
    version: ModelVersion,
) -> None:
    from cosyvoice.utils.common import TrtContextWrapper
    from cosyvoice.utils.file_utils import convert_onnx_to_trt

    precision = "fp16" if fp16 else "fp32"
    plan_path = os.path.join(
        model_dir,
        f"flow.decoder.estimator.{precision}.mygpu.plan",
    )
    onnx_path = os.path.join(model_dir, "flow.decoder.estimator.fp32.onnx")

    if version == ModelVersion.V3 and fp16:
        _logger.warning(
            "DiT TensorRT fp16 engine may have performance issues — use with caution"
        )

    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        trt_kwargs = {
            "min_shape": [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)],
            "opt_shape": [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)],
            "max_shape": [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)],
            "input_names": ["x", "mask", "mu", "cond"],
        }
        convert_onnx_to_trt(plan_path, trt_kwargs, onnx_path, fp16)

    del flow.decoder.estimator

    import tensorrt as _trt  # type: ignore[import-untyped]

    trt: Any = _trt
    device = next(flow.parameters()).device
    with open(plan_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TRT engine from {plan_path}")

    flow.decoder.estimator = TrtContextWrapper(  # type: ignore[assignment]
        engine,
        trt_concurrent=trt_concurrent,
        device=device,
    )
    _logger.info("TensorRT engine loaded: %s", plan_path)


__all__ = ["AsyncCosyVoice", "TTSMode"]
