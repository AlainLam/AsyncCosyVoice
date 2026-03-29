# Copyright (c) 2026 Alain Lam
# SPDX-License-Identifier: Apache-2.0
#
# Derived in part from FunAudioLLM/CosyVoice (Apache-2.0).
# Upstream references:
# - cosyvoice/cli/model.py
# - cosyvoice/utils/common.py
# Modified for typed session state and async streaming integration.

"""Token-to-waveform conversion and PCM helpers.

Reimplements ``CosyVoice2Model.token2wav`` / ``CosyVoice3Model.token2wav``
with explicit typed parameters instead of global dict lookups.
"""

from typing import Any

import torch
from torch.nn import functional as F

from app.cosyvoice.types import HiFTCacheV2, HiFTCacheV3, TTSSession


def _fade_in_out_device(
    fade_in_tensor: torch.Tensor,
    fade_out_tensor: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Apply overlap-add crossfade without leaving the current device.

    Review: cosyvoice.utils.common -> fade_in_out
    """
    overlap_len = min(
        window.shape[0] // 2,
        fade_in_tensor.shape[-1],
        fade_out_tensor.shape[-1],
    )
    if overlap_len == 0:
        return fade_in_tensor

    if window.device != fade_in_tensor.device or window.dtype != fade_in_tensor.dtype:
        window = window.to(device=fade_in_tensor.device, dtype=fade_in_tensor.dtype)

    blended = fade_in_tensor.clone()
    blended[..., :overlap_len] = (
        blended[..., :overlap_len] * window[:overlap_len]
        + fade_out_tensor[..., -overlap_len:] * window[-overlap_len:]
    )
    return blended


# ---------------------------------------------------------------------------
# CosyVoice2 token → waveform
# ---------------------------------------------------------------------------


def token2wav_v2(
    token: torch.Tensor,
    prompt_token: torch.Tensor,
    prompt_token_len: torch.Tensor,
    prompt_feat: torch.Tensor,
    prompt_feat_len: torch.Tensor,
    embedding: torch.Tensor,
    token_offset: int,
    session: TTSSession,
    flow: Any,
    hift: Any,
    fp16: bool,
    *,
    mel_cache_len: int,
    source_cache_len: int,
    speech_window: torch.Tensor,
    token_mel_ratio: int,
    streaming: bool = False,
    finalize: bool = False,
    speed: float = 1.0,
) -> torch.Tensor:
    """Convert speech tokens to audio waveform (CosyVoice2 path).

    Mirrors ``CosyVoice2Model.token2wav`` but operates on a typed
    ``TTSSession`` instead of global dicts.

    Returns
    -------
    torch.Tensor  [1, N_samples]
    """
    device = next(flow.parameters()).device
    token = token.to(device=device, dtype=torch.int32)

    with torch.cuda.amp.autocast(fp16):
        tts_mel, _ = flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=device),
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
        )

    tts_mel = tts_mel[:, :, token_offset * token_mel_ratio :]

    # Append hift cache
    cache = session.hift_cache
    if isinstance(cache, HiFTCacheV2):
        hift_cache_mel = cache.mel
        hift_cache_source = cache.source
        tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
    else:
        hift_cache_source = torch.zeros(1, 1, 0, dtype=tts_mel.dtype, device=device)

    if not finalize:
        tts_speech, tts_source = hift.inference(
            speech_feat=tts_mel, cache_source=hift_cache_source
        )
        if isinstance(cache, HiFTCacheV2):
            tts_speech = _fade_in_out_device(tts_speech, cache.speech, speech_window)
        session.hift_cache = HiFTCacheV2(
            mel=tts_mel[:, :, -mel_cache_len:],
            source=tts_source[:, :, -source_cache_len:],
            speech=tts_speech[:, -source_cache_len:],
        )
        tts_speech = tts_speech[:, :-source_cache_len]
    else:
        if speed != 1.0:
            assert (
                session.hift_cache is None
            ), "speed change only supported in non-stream inference mode"
            tts_mel = F.interpolate(
                tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
            )
        tts_speech, tts_source = hift.inference(
            speech_feat=tts_mel, cache_source=hift_cache_source
        )
        if isinstance(cache, HiFTCacheV2):
            tts_speech = _fade_in_out_device(tts_speech, cache.speech, speech_window)

    return tts_speech


# ---------------------------------------------------------------------------
# CosyVoice3 token → waveform
# ---------------------------------------------------------------------------


def token2wav_v3(
    token: torch.Tensor,
    prompt_token: torch.Tensor,
    prompt_token_len: torch.Tensor,
    prompt_feat: torch.Tensor,
    prompt_feat_len: torch.Tensor,
    embedding: torch.Tensor,
    token_offset: int,
    session: TTSSession,
    flow: Any,
    hift: Any,
    fp16: bool,
    *,
    token_mel_ratio: int,
    streaming: bool = False,
    finalize: bool = False,
    speed: float = 1.0,
) -> torch.Tensor:
    """Convert speech tokens to audio waveform (CosyVoice3 path).

    Mirrors ``CosyVoice3Model.token2wav`` — uses ``CausalHiFTGenerator``
    which tracks a cumulative ``speech_offset``.

    Returns
    -------
    torch.Tensor  [1, N_samples]
    """
    device = next(flow.parameters()).device
    token = token.to(device=device, dtype=torch.int32)

    with torch.cuda.amp.autocast(fp16):
        tts_mel, _ = flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=device),
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
        )

        tts_mel = tts_mel[:, :, token_offset * token_mel_ratio :]

        cache = session.hift_cache
        if isinstance(cache, HiFTCacheV3):
            tts_mel = torch.concat([cache.mel, tts_mel], dim=2)
            cache.mel = tts_mel
        else:
            session.hift_cache = HiFTCacheV3(mel=tts_mel, speech_offset=0)
            cache = session.hift_cache

        if speed != 1.0:
            assert (
                token_offset == 0 and finalize is True
            ), "speed change only supported in non-stream inference mode"
            tts_mel = F.interpolate(
                tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
            )

        tts_speech, _ = hift.inference(speech_feat=tts_mel, finalize=finalize)
        tts_speech = tts_speech[:, cache.speech_offset :]
        cache.speech_offset += tts_speech.shape[1]

    return tts_speech


# ---------------------------------------------------------------------------
# PCM conversion
# ---------------------------------------------------------------------------


def tensor_to_pcm_s16le(tensor: torch.Tensor | None) -> bytes | None:
    """Convert a waveform tensor to 16-bit little-endian PCM bytes."""
    if tensor is None:
        return None
    pcm = (
        tensor.detach()
        .squeeze(0)
        .clamp(-1.0, 1.0)
        .mul(32767)
        .round()
        .clamp(-32768, 32767)
        .to(torch.int16)
        .cpu()
    )
    return pcm.numpy().tobytes()
