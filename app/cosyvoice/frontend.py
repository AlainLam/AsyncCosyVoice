# Copyright (c) 2026 Alain Lam
# SPDX-License-Identifier: Apache-2.0
#
# Derived in part from FunAudioLLM/CosyVoice (Apache-2.0).
# Upstream references:
# - cosyvoice/cli/frontend.py
# Modified for CosyVoice3 prompt normalization and cached speaker data.

import logging
import json
import importlib
import os
import re
from functools import partial
from os import PathLike
from typing import BinaryIO, Callable

import inflect
import numpy as np
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi
import whisper
from torch import Tensor

from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    is_only_punctuation,
    remove_bracket,
    replace_blank,
    replace_corner_mark,
    spell_out_number,
    split_paragraph,
)

from app.cosyvoice.types import (
    FrontendEngine,
    ModelInput,
    ModelVCInput,
    ModelVersion,
    SpeakerInfo,
    TTSMode,
)

_CV3_SYS_INST = "You are a helpful assistant."
_EOP = "<|endofprompt|>"

_logger = logging.getLogger(__name__)


class CosyVoiceFrontend:
    def __init__(
        self,
        model_version: ModelVersion,
        device: torch.device,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk_info_path: str = "",
        allowed_special: str = "all",
        ttsfrd_resource_dir: str | None = None,
    ):
        self.model_version = model_version
        self.device = device
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.spk_info_path = spk_info_path

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1

        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model,
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available()
                    else "CPUExecutionProvider"
                )
            ],
        )

        if os.path.exists(spk_info_path):
            self.spk2info: dict[str, SpeakerInfo] = torch.load(
                spk_info_path,
                map_location=self.device,
                weights_only=True,
            )
        else:
            self.spk2info = {}

        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        self.text_frontend: FrontendEngine = FrontendEngine.NONE
        self._prepare_frontend_engine(ttsfrd_resource_dir)

    def _prepare_frontend_engine(self, ttsfrd_resource_dir: str | None = None) -> None:
        if ttsfrd_resource_dir is not None:
            try:
                ttsfrd = importlib.import_module("ttsfrd")

                self.frd = ttsfrd.TtsFrontendEngine()
                assert (
                    self.frd.initialize(ttsfrd_resource_dir) is True
                ), "failed to initialize ttsfrd resource"
                self.frd.set_lang_type("pinyinvg")
                self.text_frontend = FrontendEngine.TTSFRD
                _logger.info("use ttsfrd frontend")
                return
            except Exception:
                _logger.warning(
                    "ttsfrd frontend is not available, falling back to other frontends if possible."
                )

        try:
            from wetext import Normalizer as EnNormalizer
            from wetext import Normalizer as ZhNormalizer

            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.text_frontend = FrontendEngine.WETEXT
            _logger.info("use wetext frontend")
        except Exception:
            self.text_frontend = FrontendEngine.NONE
            _logger.info("no frontend is avaliable")

    def _extract_text_token(self, text: str) -> tuple[Tensor, Tensor]:
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32, device=self.device)
        text_token_len = torch.tensor(
            [text_token.shape[1]], dtype=torch.int32, device=self.device
        )
        return text_token, text_token_len

    def _empty_text_token(self) -> tuple[Tensor, Tensor]:
        text_token = torch.zeros((1, 0), dtype=torch.int32, device=self.device)
        text_token_len = torch.tensor([0], dtype=torch.int32, device=self.device)
        return text_token, text_token_len

    def _build_prompt(
        self,
        *,
        mode: TTSMode,
        ref_text: str = "",
        instruction: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        instruction = (instruction or "").removesuffix(_EOP).strip()

        if mode == TTSMode.ZERO_SHOT:
            if self.model_version == ModelVersion.V2:
                return self._extract_text_token(ref_text)
            if self.model_version == ModelVersion.V3:
                return self._extract_text_token(f"{_CV3_SYS_INST}{_EOP}{ref_text}")
            raise ValueError(f"Unsupported CosyVoice version: {self.model_version}")

        if mode == TTSMode.INSTRUCT:
            if self.model_version == ModelVersion.V2:
                return self._extract_text_token(f"{instruction}{_EOP}")
            if self.model_version == ModelVersion.V3:
                return self._extract_text_token(f"{_CV3_SYS_INST} {instruction}{_EOP}")
            raise ValueError(f"Unsupported CosyVoice version: {self.model_version}")

        if mode == TTSMode.CROSS_LINGUAL:
            if self.model_version == ModelVersion.V2:
                return self._empty_text_token()
            if self.model_version == ModelVersion.V3:
                return self._extract_text_token(f"{_CV3_SYS_INST}{_EOP}")
            raise ValueError(f"Unsupported CosyVoice version: {self.model_version}")

        raise ValueError(f"Unsupported TTS mode: {mode}")

    def load_speech(
        self, file: str | PathLike[str] | BinaryIO, resample: int
    ) -> Tensor:
        # Seek to the beginning before each load so that BinaryIO objects can be
        # safely passed to load_speech multiple times (e.g., loading at 16k and 24k).
        if hasattr(file, "seekable") and file.seekable():  # type: ignore[attr-defined]
            file.seek(0)  # type: ignore[attr-defined]
        return load_wav(file, resample)

    def _extract_speech_token(self, speech_16k: Tensor) -> tuple[Tensor, Tensor]:
        assert (
            speech_16k.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech_16k, n_mels=128)
        speech_token = np.asarray(
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    .name: feat.detach()
                    .cpu()
                    .numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0],
            dtype=np.int32,
        ).reshape(1, -1)
        speech_token_tensor = torch.from_numpy(speech_token).to(
            device=self.device, dtype=torch.int32
        )
        speech_token_len = torch.tensor(
            [speech_token_tensor.shape[1]], dtype=torch.int32, device=self.device
        )
        return speech_token_tensor, speech_token_len

    def _extract_spk_embedding(self, speech_16k: Tensor) -> Tensor:
        feat = kaldi.fbank(
            speech_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = np.asarray(
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0],
            dtype=np.float32,
        ).reshape(1, -1)
        return torch.from_numpy(embedding).to(device=self.device, dtype=torch.float32)

    def _extract_speech_feat(self, speech_24k: Tensor) -> tuple[Tensor, Tensor]:
        speech_feat = self.feat_extractor(speech_24k).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.to(device=self.device).unsqueeze(dim=0)
        speech_feat_len = torch.tensor(
            [speech_feat.shape[1]], dtype=torch.int32, device=self.device
        )
        return speech_feat, speech_feat_len

    def text_normalize(
        self,
        text: str,
        split: bool = True,
        text_frontend: bool = True,
    ) -> list[str] | str:
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text

        text = text.strip()
        if self.text_frontend == FrontendEngine.TTSFRD:
            texts = [
                i["text"]
                for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]
            ]
            text = "".join(texts)
        else:
            if contains_chinese(text):
                if self.text_frontend == FrontendEngine.WETEXT:
                    text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
                texts = list(
                    split_paragraph(
                        text,
                        partial(
                            self.tokenizer.encode, allowed_special=self.allowed_special
                        ),
                        "zh",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
            else:
                if self.text_frontend == FrontendEngine.WETEXT:
                    text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(
                    split_paragraph(
                        text,
                        partial(
                            self.tokenizer.encode, allowed_special=self.allowed_special
                        ),
                        "en",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )

        texts = [item for item in texts if not is_only_punctuation(item)]
        return texts if split is True else text

    def register_speaker(
        self,
        spk_id: str,
        ref_wav: str | PathLike[str] | BinaryIO,
        ref_text: str,
    ) -> None:
        speech_16k = self.load_speech(ref_wav, 16000)
        speech_24k = self.load_speech(ref_wav, 24000)

        speech_token, speech_token_len = self._extract_speech_token(speech_16k)
        embedding = self._extract_spk_embedding(speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(speech_24k)
        prompt_text, prompt_text_len = self._build_prompt(
            mode=TTSMode.ZERO_SHOT,
            ref_text=ref_text,
        )

        if self.model_version in (ModelVersion.V2, ModelVersion.V3):
            token_len = min(speech_feat.shape[1] // 2, speech_token.shape[1])
            speech_feat = speech_feat[:, : 2 * token_len]
            speech_feat_len[:] = 2 * token_len
            speech_token = speech_token[:, :token_len]
            speech_token_len[:] = token_len

        self.spk2info[spk_id] = SpeakerInfo(
            ref_text=ref_text,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            llm_prompt_speech_token=speech_token,
            llm_prompt_speech_token_len=speech_token_len,
            flow_prompt_speech_token=speech_token,
            flow_prompt_speech_token_len=speech_token_len,
            prompt_speech_feat=speech_feat,
            prompt_speech_feat_len=speech_feat_len,
            llm_embedding=embedding,
            flow_embedding=embedding,
        )

    def save_speaker(self, path: str | None = None) -> None:
        save_path = path or self.spk_info_path
        if not save_path:
            raise ValueError(
                "No save path provided. Pass an explicit `path` or set `spk_info_path` in the constructor."
            )
        torch.save(self.spk2info, save_path)

    def build_tts_input(
        self,
        spk_id: str,
        text: str,
        mode: TTSMode = TTSMode.ZERO_SHOT,
        instruction: str | None = None,
    ) -> ModelInput:
        instruction = instruction.strip() if instruction is not None else None
        if not instruction:
            instruction = None

        if mode == TTSMode.INSTRUCT:
            if instruction is None:
                raise ValueError("mode='instruct' requires a non-empty instruction")
        elif instruction is not None:
            raise ValueError(f"mode='{mode.value}' does not accept instruction")

        if spk_id not in self.spk2info:
            raise KeyError(f"Speaker '{spk_id}' is not registered")

        text_token, text_token_len = self._extract_text_token(text)
        speaker = self.spk2info[spk_id]
        if "ref_text" not in speaker:
            raise ValueError(
                f"Voice '{spk_id}' was registered via the legacy API (no ref_text). "
                "Please re-register using register_speaker()."
            )

        prompt_text = speaker["prompt_text"]
        prompt_text_len = speaker["prompt_text_len"]
        llm_prompt_speech_token = speaker["llm_prompt_speech_token"]
        llm_prompt_speech_token_len = speaker["llm_prompt_speech_token_len"]

        if mode != TTSMode.ZERO_SHOT:
            prompt_text, prompt_text_len = self._build_prompt(
                mode=mode,
                instruction=instruction,
            )
            llm_prompt_speech_token = llm_prompt_speech_token.new_zeros((1, 0))
            llm_prompt_speech_token_len = llm_prompt_speech_token_len.new_tensor([0])

        return ModelInput(
            type="tts",
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            llm_prompt_speech_token=llm_prompt_speech_token,
            llm_prompt_speech_token_len=llm_prompt_speech_token_len,
            flow_prompt_speech_token=speaker["flow_prompt_speech_token"],
            flow_prompt_speech_token_len=speaker["flow_prompt_speech_token_len"],
            prompt_speech_feat=speaker["prompt_speech_feat"],
            prompt_speech_feat_len=speaker["prompt_speech_feat_len"],
            llm_embedding=speaker["llm_embedding"],
            flow_embedding=speaker["flow_embedding"],
            text=text_token,
            text_len=text_token_len,
        )

    def build_vc_model_input(
        self,
        source_speech: str | PathLike[str] | BinaryIO,
        prompt_speech: str | PathLike[str] | BinaryIO,
    ) -> ModelVCInput:
        source_speech_16k = self.load_speech(source_speech, 16000)
        prompt_speech_16k = self.load_speech(prompt_speech, 16000)
        prompt_speech_24k = self.load_speech(prompt_speech, 24000)

        source_speech_token, source_speech_token_len = self._extract_speech_token(
            source_speech_16k
        )
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            prompt_speech_16k
        )
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(
            prompt_speech_24k
        )
        flow_embedding = self._extract_spk_embedding(prompt_speech_16k)

        return ModelVCInput(
            type="vc",
            source_speech_token=source_speech_token,
            source_speech_token_len=source_speech_token_len,
            flow_prompt_speech_token=prompt_speech_token,
            flow_prompt_speech_token_len=prompt_speech_token_len,
            prompt_speech_feat=prompt_speech_feat,
            prompt_speech_feat_len=prompt_speech_feat_len,
            flow_embedding=flow_embedding,
        )
