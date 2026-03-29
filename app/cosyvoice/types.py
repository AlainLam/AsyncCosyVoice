# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alain Lam

"""Strongly-typed data structures for the async CosyVoice inference engine."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Literal, TypedDict
from typing_extensions import NotRequired

import torch


def _empty_token_buffer() -> torch.Tensor:
    return torch.empty((1, 0), dtype=torch.int32)


class ModelVersion(IntEnum):
    V2 = 2
    V3 = 3


class FrontendEngine(str, Enum):
    TTSFRD = "ttsfrd"
    WETEXT = "wetext"
    NONE = "none"


class TTSMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    INSTRUCT = "instruct"
    CROSS_LINGUAL = "cross_lingual"


# ---------------------------------------------------------------------------
# HiFT vocoder streaming caches
# ---------------------------------------------------------------------------


@dataclass
class HiFTCacheV2:
    """Streaming cache for HiFTGenerator (CosyVoice2).

    Carries mel, source, and speech tail segments for fade-in/out
    crossfade between consecutive streaming chunks.
    """

    mel: torch.Tensor
    """[1, 80, mel_cache_len] — tail mel frames for overlap."""

    source: torch.Tensor
    """[1, 1, source_cache_len] — source excitation cache."""

    speech: torch.Tensor
    """[1, source_cache_len] — speech tail for fade crossfade."""


@dataclass
class HiFTCacheV3:
    """Streaming cache for CausalHiFTGenerator (CosyVoice3).

    Accumulates the full mel and tracks how many speech samples have
    already been yielded so only the new portion is returned each time.
    """

    mel: torch.Tensor
    """[1, 80, T_accumulated] — full accumulated mel up to current chunk."""

    speech_offset: int
    """Number of speech samples already yielded."""


# ---------------------------------------------------------------------------
# Per-request session state
# ---------------------------------------------------------------------------


@dataclass
class TTSSession:
    """Per-request mutable state for an async TTS inference session.

    Replaces the global ``tts_speech_token_dict``, ``llm_end_dict``,
    ``hift_cache_dict`` dicts from the original CosyVoice2/3Model.
    """

    request_id: str
    """Unique identifier for this inference session."""

    tokens: list[int] = field(default_factory=list)
    """Accumulated speech tokens produced by the LLM."""

    token_buffer: torch.Tensor = field(default_factory=_empty_token_buffer)
    """Cached int32 prefix tensor used to avoid rebuilding the full token history."""

    done: bool = False
    """Set to ``True`` when the LLM generator has finished."""

    error: BaseException | None = None
    """If the LLM producer hits an error, it is stored here."""

    token_event: asyncio.Event = field(default_factory=asyncio.Event)
    """Signalled whenever new tokens are appended or ``done`` is set."""

    hift_cache: HiFTCacheV2 | HiFTCacheV3 | None = None
    """Vocoder streaming cache — version-specific."""


# ---------------------------------------------------------------------------
# Speaker info
# ---------------------------------------------------------------------------


class SpeakerInfo(TypedDict):
    """
    Metadata about a speaker, added some fields that are useful for engineering optimization.
    """

    ref_text: NotRequired[str]
    """Plain reference transcription (no system-prompt prefix)."""

    prompt_text: torch.Tensor
    """[1, N_text_tokens] int32 — pre-tokenised prompt text."""

    prompt_text_len: torch.Tensor
    """[1] int32."""

    llm_prompt_speech_token: torch.Tensor
    """[1, N_speech_tokens] int32 — reference speech tokens for LLM."""

    llm_prompt_speech_token_len: torch.Tensor
    """[1] int32."""

    flow_prompt_speech_token: torch.Tensor
    """[1, N_speech_tokens] int32 — reference speech tokens for flow model."""

    flow_prompt_speech_token_len: torch.Tensor
    """[1] int32."""

    prompt_speech_feat: torch.Tensor
    """[1, T_mel, 80] float — 24 kHz mel-spectrogram features."""

    prompt_speech_feat_len: torch.Tensor
    """[1] int32."""

    llm_embedding: torch.Tensor
    """[1, 192] float — speaker embedding for LLM conditioning."""

    flow_embedding: torch.Tensor
    """[1, 192] float — speaker embedding for flow model conditioning."""


# ----------------------------------------------------------------------------
# Model input types
# ----------------------------------------------------------------------------


class ModelInput(TypedDict):
    """Typed payload accepted by ``CosyVoiceModel.tts()`` for TTS modes."""

    type: Literal["tts"]
    """Discriminant for TTS vs VC mode."""

    text: torch.Tensor
    """[1, N_text_tokens] int32 — synthesized text tokens."""

    text_len: torch.Tensor
    """[1] int32."""

    flow_embedding: torch.Tensor
    """[1, 192] float — speaker embedding for flow conditioning."""

    llm_embedding: torch.Tensor
    """[1, 192] float — speaker embedding for LLM conditioning."""

    prompt_text: torch.Tensor
    """[1, N_text_tokens] int32 — prompt / instruction text tokens."""

    prompt_text_len: torch.Tensor
    """[1] int32."""

    llm_prompt_speech_token: torch.Tensor
    """[1, N_speech_tokens] int32 — reference speech tokens for the LLM."""

    llm_prompt_speech_token_len: torch.Tensor
    """[1] int32."""

    flow_prompt_speech_token: torch.Tensor
    """[1, N_speech_tokens] int32 — reference speech tokens for the flow model."""

    flow_prompt_speech_token_len: torch.Tensor
    """[1] int32."""

    prompt_speech_feat: torch.Tensor
    """[1, T_mel, 80] float — prompt mel features."""

    prompt_speech_feat_len: torch.Tensor
    """[1] int32."""


class ModelVCInput(TypedDict):
    """Typed payload for the voice-conversion subset of ``tts()`` inputs."""

    type: Literal["vc"]
    """Discriminant for VC vs TTS mode."""

    source_speech_token: torch.Tensor
    """[1, N_speech_tokens] int32 — source speech tokens for VC."""

    source_speech_token_len: torch.Tensor
    """[1] int32."""

    flow_prompt_speech_token: torch.Tensor
    """[1, N_speech_tokens] int32 — prompt speech tokens for flow conditioning."""

    flow_prompt_speech_token_len: torch.Tensor
    """[1] int32."""

    prompt_speech_feat: torch.Tensor
    """[1, T_mel, 80] float — prompt mel features."""

    prompt_speech_feat_len: torch.Tensor
    """[1] int32."""

    flow_embedding: torch.Tensor
    """[1, 192] float — prompt speaker embedding for flow conditioning."""
