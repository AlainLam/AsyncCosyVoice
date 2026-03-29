# Copyright (c) 2026 Alain Lam
# SPDX-License-Identifier: Apache-2.0
#
# Derived in part from FunAudioLLM/CosyVoice (Apache-2.0).
# Upstream references:
# - cosyvoice/llm/llm.py
# - cosyvoice/cli/model.py
# Modified for AsyncLLMEngine-based inference.

"""Async vLLM inference wrapper for CosyVoice speech-token generation.

This module preserves the prompt construction used by the official
``Qwen2LM`` / ``CosyVoice3LM`` inference path and delegates decoding to
``AsyncLLMEngine``.

References:
- ``cosyvoice/llm/llm.py``: ``Qwen2LM.inference()`` and vLLM decoding behavior
- ``cosyvoice/cli/model.py``: ``CosyVoice2Model.load_vllm()``
"""

import asyncio
import logging
import uuid as _uuid
from collections.abc import AsyncGenerator
from typing import Any

import torch

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import EmbedsPrompt
from vllm.sampling_params import RequestOutputKind

from cosyvoice.utils.file_utils import export_cosyvoice2_vllm

from app.cosyvoice.types import ModelVersion


_logger = logging.getLogger(__name__)


class AsyncLLM:
    """Async speech-token generator backed by ``AsyncLLMEngine``.

    Parameters
    ----------
    llm_module:
        The original ``Qwen2LM`` (V2) or ``CosyVoice3LM`` (V3) module.
        We access its embedding tables and token metadata but do NOT
        run its ``inference()`` / ``inference_wrapper()`` methods.
    version:
        Model version — controls which embedding table is used for the
        ``<sos>`` and ``<task_id>`` special tokens.
    device:
        Target CUDA / CPU device.
    """

    def __init__(
        self,
        llm_module: torch.nn.Module,
        version: ModelVersion,
        device: torch.device,
    ) -> None:
        self._mod: Any = llm_module
        self._version = version
        self._device = device
        self.vllm_engine: AsyncLLMEngine | None = None

        self._llm_input_size: int = int(llm_module.llm_input_size)  # type: ignore[arg-type]
        self._stop_token_ids: list[int] = [int(x) for x in llm_module.stop_token_ids]  # type: ignore[union-attr]
        # vLLM may include a stop token in the final DELTA output.
        # Filter it out before passing IDs to the downstream speech-token path.
        self._stop_token_ids_set: frozenset[int] = frozenset(self._stop_token_ids)
        self._sos: int = int(llm_module.sos)  # type: ignore[arg-type]
        self._task_id: int = int(llm_module.task_id)  # type: ignore[arg-type]

    def load_vllm(self, model_dir: str, **vllm_kwargs: Any) -> None:
        """Export weights and initialize ``AsyncLLMEngine``."""
        export_cosyvoice2_vllm(self._mod, model_dir, self._device)

        engine_kwargs: dict[str, Any] = {
            "model": model_dir,
            "skip_tokenizer_init": True,
            "enable_prompt_embeds": True,
            # Override via vllm_kwargs={"gpu_memory_utilization": x} if needed.
            "gpu_memory_utilization": 0.2,
        }
        engine_kwargs.update(vllm_kwargs)
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Free the original transformer layers — they are now redundant.
        del self._mod.llm.model.model.layers
        _logger.info("AsyncLLMEngine loaded from %s", model_dir)

    @torch.inference_mode()
    def prepare_lm_input(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> tuple[torch.Tensor, int, int]:
        """Build the prompt embedding and output-length bounds for decoding.

        Mirrors official ``Qwen2LM`` / ``CosyVoice3LM`` inference.

        Returns
        -------
        lm_input : Tensor [1, seq_len, hidden]
            The full LLM prompt embedding.
        min_len : int
            Minimum number of speech tokens to generate.
        max_len : int
            Maximum number of speech tokens to generate.
        """
        mod = self._mod

        # Concatenate prompt text and synthesis text before token embedding.
        text = torch.concat([prompt_text, text], dim=1)
        text_len = text_len + prompt_text_len
        text_emb = mod.llm.model.model.embed_tokens(text)

        # CosyVoice3 expects ``<|endofprompt|>`` to appear in the combined text.
        if self._version == ModelVersion.V3:
            assert 151646 in text, (
                "<|endofprompt|> not detected in CosyVoice3 text or prompt_text, "
                "check your input!"
            )

        # V2 and V3 use different embedding tables for special tokens.
        if self._version == ModelVersion.V3:
            sos_emb = mod.speech_embedding.weight[self._sos].reshape(1, 1, -1)
            task_id_emb = mod.speech_embedding.weight[self._task_id].reshape(1, 1, -1)
        elif self._version == ModelVersion.V2:
            sos_emb = mod.llm_embedding.weight[self._sos].reshape(1, 1, -1)
            task_id_emb = mod.llm_embedding.weight[self._task_id].reshape(1, 1, -1)
        else:
            raise ValueError(f"Unsupported model version: {self._version}")

        # Embed reference speech tokens when they are provided.
        if prompt_speech_token_len.item() != 0:
            prompt_speech_token_emb = mod.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1,
                0,
                self._llm_input_size,
                dtype=text_emb.dtype,
                device=text_emb.device,
            )

        lm_input = torch.concat(
            [sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # Decode-length bounds follow the official inference ratios.
        min_len = int((text_len - prompt_text_len).item() * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len).item() * max_token_text_ratio)

        return lm_input, min_len, max_len

    async def generate_tokens(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        request_id: str | None = None,
    ) -> AsyncGenerator[list[int], None]:
        """Stream speech-token ID batches from ``AsyncLLMEngine``.

        Parameters
        ----------
        text, text_len, prompt_text, prompt_text_len, prompt_speech_token,
        prompt_speech_token_len:
            Same semantics as official ``Qwen2LM`` / ``CosyVoice3LM`` inference.
        request_id:
            Optional unique string for the vLLM request. A UUID is
            generated if not provided.

        Yields
        ------
        list[int]
            New speech token IDs from one DELTA update.
        """
        assert self.vllm_engine is not None, "call load_vllm() first"

        if request_id is None:
            request_id = str(_uuid.uuid4())

        lm_input, min_len, max_len = self.prepare_lm_input(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
        )
        prompt = EmbedsPrompt(
            prompt_embeds=lm_input.squeeze(0).to(torch.bfloat16),
        )

        sampling_params = SamplingParams(
            top_k=sampling,
            stop_token_ids=self._stop_token_ids,
            min_tokens=min_len,
            max_tokens=max_len,
            output_kind=RequestOutputKind.DELTA,
        )

        try:
            async for request_output in self.vllm_engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                if not request_output.outputs:
                    continue
                token_ids = [
                    token_id
                    for token_id in request_output.outputs[0].token_ids
                    if token_id not in self._stop_token_ids_set
                ]
                if token_ids:
                    yield token_ids
        finally:
            try:
                await self.vllm_engine.abort(request_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.warning(
                    "Failed to abort vLLM request %s during cleanup",
                    request_id,
                    exc_info=True,
                )
