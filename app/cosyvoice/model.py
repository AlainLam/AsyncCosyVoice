# Copyright (c) 2026 Alain Lam
# SPDX-License-Identifier: Apache-2.0
#
# Derived in part from FunAudioLLM/CosyVoice (Apache-2.0).
# Upstream references:
# - cosyvoice/cli/model.py
# Modified for per-request session state and async streaming flow.

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import cast

import numpy as np
import torch

from app.cosyvoice.audio import token2wav_v2, token2wav_v3
from app.cosyvoice.llm import AsyncLLM
from app.cosyvoice.types import ModelInput, ModelVCInput, ModelVersion, TTSSession

_MAX_SILENT_TOKENS = 5


_logger = logging.getLogger(__name__)


class AsyncModel:
    """Fully asynchronous TTS model orchestrator.

    Parameters
    ----------
    llm : AsyncLLM
        Async vLLM-backed token generator.
    flow : torch.nn.Module
        Flow matching model.
    hift : torch.nn.Module
        HiFT vocoder.
    version : ModelVersion
        V2 or V3 — controls token2wav variant and cache types.
    device : torch.device
        Target device used by the model components.
    fp16 : bool
        Whether to run flow/hift under ``torch.cuda.amp.autocast``.
    initial_token_hop_len : int
        Number of tokens to accumulate before synthesising the first
        streaming chunk. The second chunk then realigns to the 25-token
        flow grid before continuing with normal streaming hops.

        Keep the default when you want the most stable first chunk.
        Use 15 when you want a lower TTFT and can accept some first-chunk risk.
    silent_tokens : list[int]
        Token IDs considered "silence" — runs of more than 5 consecutive
        are suppressed. Empty for V2, FSQ tokens for V3.
    """

    def __init__(
        self,
        llm: AsyncLLM,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
        version: ModelVersion,
        device: torch.device,
        fp16: bool = False,
        initial_token_hop_len: int = 25,
        silent_tokens: list[int] = [],
    ) -> None:
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.version = version
        self.device = device
        self.fp16 = fp16
        self.initial_token_hop_len = initial_token_hop_len
        self.silent_tokens: set[int] = set(silent_tokens)

        self.token_hop_len: int = 25
        self.token_max_hop_len: int = 4 * self.token_hop_len
        self.stream_scale_factor: int = 2

        if not (1 <= initial_token_hop_len <= self.token_hop_len):
            raise ValueError(
                f"initial_token_hop_len must be between 1 and {self.token_hop_len}"
            )

        if self.initial_token_hop_len < self.token_hop_len:
            _logger.warning(
                f"initial_token_hop_len {initial_token_hop_len} is less than official hop length {self.token_hop_len}, "
                "this may cause some quality drop on the first chunk, please adjust according to the actual needs."
            )

        self.pre_lookahead_len: int = getattr(flow, "pre_lookahead_len", 3)
        self.token_mel_ratio: int = getattr(flow, "token_mel_ratio", 2)

        if version == ModelVersion.V2:
            self.mel_cache_len: int = 8
            self.source_cache_len: int = self.mel_cache_len * 480
            self.speech_window: torch.Tensor = torch.from_numpy(
                np.hamming(2 * self.source_cache_len)
            ).to(device=self.device, dtype=torch.float32)
        else:
            self.mel_cache_len = 0
            self.source_cache_len = 0
            self.speech_window = torch.empty(0, device=self.device, dtype=torch.float32)

    async def tts(
        self,
        *,
        model_input: ModelInput | ModelVCInput,
        stream: bool,
        speed: float = 1.0,
    ) -> AsyncGenerator[dict[str, torch.Tensor], None]:
        """Main TTS entry point."""
        session = TTSSession(request_id=str(uuid.uuid4()))

        infer_task: asyncio.Task[None]
        prompt_token = model_input["flow_prompt_speech_token"]
        prompt_token_len = model_input["flow_prompt_speech_token_len"]
        prompt_feat = model_input["prompt_speech_feat"]
        prompt_feat_len = model_input["prompt_speech_feat_len"]
        embedding = model_input["flow_embedding"]
        if model_input["type"] == "tts":
            tts_input = cast(ModelInput, model_input)
            infer_task = asyncio.create_task(self._llm_producer(session, tts_input))
        else:
            vc_input = cast(ModelVCInput, model_input)
            infer_task = asyncio.create_task(self._vc_producer(session, vc_input))
        try:
            if stream:
                async for chunk in self._streaming(
                    session=session,
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                ):
                    yield chunk
            else:
                chunk = await self._non_streaming(
                    session=session,
                    infer_task=infer_task,
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    speed=speed,
                )
                yield chunk
        finally:
            if not infer_task.done():
                infer_task.cancel()
            try:
                await infer_task
            except asyncio.CancelledError:
                pass

    async def _streaming(
        self,
        *,
        session: TTSSession,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
    ) -> AsyncGenerator[dict[str, torch.Tensor], None]:
        token_offset = 0
        emitted_chunks = 0
        next_hop_len = self.initial_token_hop_len
        prompt_len = int(prompt_token_len.item())

        while True:
            await session.token_event.wait()
            session.token_event.clear()

            if session.error is not None:
                raise session.error

            current_hop_len = next_hop_len
            available = len(session.tokens) - token_offset

            while available >= current_hop_len + self.pre_lookahead_len:
                end_idx = token_offset + current_hop_len + self.pre_lookahead_len
                token_prefix = self._token_prefix(session, end_idx)
                speech = await asyncio.to_thread(
                    self._token2wav,
                    token=token_prefix,
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    token_offset=token_offset,
                    session=session,
                    streaming=True,
                    finalize=False,
                )
                yield {"tts_speech": speech.cpu()}

                token_offset += current_hop_len
                emitted_chunks += 1

                if emitted_chunks == 1:
                    next_hop_len = self._calculate_realign_hop_len(
                        prompt_token_len=prompt_len,
                        token_offset=token_offset,
                    )
                elif emitted_chunks == 2:
                    next_hop_len = min(
                        self.token_max_hop_len,
                        self.token_hop_len * self.stream_scale_factor,
                    )
                else:
                    next_hop_len = min(
                        self.token_max_hop_len,
                        next_hop_len * self.stream_scale_factor,
                    )

                current_hop_len = next_hop_len
                available = len(session.tokens) - token_offset

            if session.done and available < current_hop_len + self.pre_lookahead_len:
                break

        if session.error is not None:
            raise session.error

        if session.tokens:
            final_tokens = self._token_prefix(session)
            speech = await asyncio.to_thread(
                self._token2wav,
                token=final_tokens,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                token_offset=token_offset,
                session=session,
                streaming=False,
                finalize=True,
            )
            yield {"tts_speech": speech.cpu()}

    async def _non_streaming(
        self,
        *,
        session: TTSSession,
        infer_task: asyncio.Task[None],
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        speed: float,
    ) -> dict[str, torch.Tensor]:
        await infer_task
        if session.error is not None:
            raise session.error

        tokens = self._token_prefix(session)
        speech = await asyncio.to_thread(
            self._token2wav,
            token=tokens,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            token_offset=0,
            session=session,
            streaming=False,
            finalize=True,
            speed=speed,
        )
        return {"tts_speech": speech.cpu()}

    async def _llm_producer(self, session: TTSSession, model_input: ModelInput) -> None:
        text = model_input["text"]

        silent_run = 0
        try:
            async for token_batch in self.llm.generate_tokens(
                text=text,
                text_len=model_input["text_len"],
                prompt_text=model_input["prompt_text"],
                prompt_text_len=model_input["prompt_text_len"],
                prompt_speech_token=model_input["llm_prompt_speech_token"],
                prompt_speech_token_len=model_input["llm_prompt_speech_token_len"],
                request_id=session.request_id,
            ):
                accepted: list[int] = []
                for token_id in token_batch:
                    if token_id in self.silent_tokens:
                        silent_run += 1
                        if silent_run > _MAX_SILENT_TOKENS:
                            continue
                    else:
                        silent_run = 0
                    accepted.append(token_id)

                if accepted:
                    session.tokens.extend(accepted)
                    session.token_event.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.exception("LLM producer failed for request %s", session.request_id)
            session.error = exc
        finally:
            session.done = True
            session.token_event.set()

    async def _vc_producer(
        self, session: TTSSession, model_input: ModelVCInput
    ) -> None:
        token = model_input["source_speech_token"].to(device="cpu", dtype=torch.int32)
        session.token_buffer = token
        session.tokens = token.squeeze(0).tolist()
        session.done = True
        session.token_event.set()

    def _token_prefix(
        self, session: TTSSession, end_idx: int | None = None
    ) -> torch.Tensor:
        if end_idx is None:
            end_idx = len(session.tokens)

        cached_len = session.token_buffer.shape[1]
        if end_idx > cached_len:
            suffix = torch.tensor(
                [session.tokens[cached_len:end_idx]],
                dtype=torch.int32,
            )
            if cached_len == 0:
                session.token_buffer = suffix
            else:
                session.token_buffer = torch.cat([session.token_buffer, suffix], dim=1)

        return session.token_buffer[:, :end_idx]

    def _calculate_realign_hop_len(
        self,
        *,
        prompt_token_len: int,
        token_offset: int,
    ) -> int:
        remainder = (prompt_token_len + token_offset) % self.token_hop_len
        correction = (-remainder) % self.token_hop_len
        return self.token_hop_len + correction

    def _token2wav(
        self,
        *,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        token_offset: int,
        session: TTSSession,
        streaming: bool = False,
        finalize: bool = False,
        speed: float = 1.0,
    ) -> torch.Tensor:
        if self.version == ModelVersion.V2:
            return token2wav_v2(
                token=token,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                token_offset=token_offset,
                session=session,
                flow=self.flow,
                hift=self.hift,
                fp16=self.fp16,
                mel_cache_len=self.mel_cache_len,
                source_cache_len=self.source_cache_len,
                speech_window=self.speech_window,
                token_mel_ratio=self.token_mel_ratio,
                streaming=streaming,
                finalize=finalize,
                speed=speed,
            )

        return token2wav_v3(
            token=token,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            token_offset=token_offset,
            session=session,
            flow=self.flow,
            hift=self.hift,
            fp16=self.fp16,
            token_mel_ratio=self.token_mel_ratio,
            streaming=streaming,
            finalize=finalize,
            speed=speed,
        )
