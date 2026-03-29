import asyncio
import logging
import os
import tempfile
import torch

from contextlib import suppress
from typing import AsyncGenerator


from app.cosyvoice import AsyncCosyVoice, TTSMode
from app.cosyvoice.audio import tensor_to_pcm_s16le
from app.config import (
    AppSettings,
    PROJECT_ROOT_DIR,
    MIN_REF_AUDIO_DURATION_SEC,
    MAX_REF_AUDIO_DURATION_SEC,
)
from app.ffmpeg import (
    AudioMetadata,
    AsyncFfmpegProcess,
    AudioFormat,
    AudioTranscodingError,
    audio_file_to_wav,
    build_ffmpeg_cmd,
    pcm_bytes_transcoding,
    probe_audio_file,
)

_logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base exception class for service errors."""


class ServiceNotInitializedError(ServiceError):
    """Raised when the service is not properly initialized."""


class ServiceVoiceAlreadyExistsError(ServiceError):
    """Raised when attempting to register a voice with an ID that already exists."""


class ServiceInvalidVoiceIdError(ServiceError):
    """Raised when the provided voice ID is invalid."""


class ServiceVoiceNotFoundError(ServiceError):
    """Raised when the requested voice ID does not exist."""


class ServiceReferenceAudioTooShortError(ServiceError):
    """Raised when reference audio duration is below the minimum."""


class ServiceReferenceAudioTooLongError(ServiceError):
    """Raised when reference audio duration exceeds the maximum."""


class ServiceReferenceAudioInvalidChannelsError(ServiceError):
    """Raised when reference audio has no usable channels."""


class ServiceReferenceAudioSampleRateTooLowError(ServiceError):
    """Raised when reference audio sample rate is below the minimum."""


class CosyVoiceService:

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.cosyvoice: AsyncCosyVoice | None = None
        self._speaker_registry_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the service, including loading the model and registering default voices."""
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

        self.settings.apply_vllm_env()
        model_dir = str(self.settings.model_path.resolve())
        vllm_kwargs = self.settings.build_vllm_kwargs()

        self.cosyvoice = AsyncCosyVoice.load(
            model_dir=model_dir,
            load_trt=True,
            fp16=self.settings.fp16,
            trt_concurrent=self.settings.trt_concurrent,
            vllm_kwargs=vllm_kwargs,
            initial_token_hop_len=self.settings.initial_token_hop_len,
            ttsfrd_resource_dir=str(self.settings.ttsfrd_resource_dir.resolve()),
        )

        await self._register_default_voices()

    def ensure_cosyvoice(self) -> AsyncCosyVoice:
        if self.cosyvoice is None:
            raise ServiceNotInitializedError()
        return self.cosyvoice

    def list_voices(self) -> list[str]:
        """Return all registered voice identifiers."""
        cosyvoice = self.ensure_cosyvoice()
        return cosyvoice.list_available_spks()

    def voice_exists(self, voice_id: str) -> bool:
        """Check if a voice_id is already registered."""
        return voice_id in self.list_voices()

    def ensure_voice_exists(self, voice_id: str) -> None:
        """Validate voice ID format and require the voice to be registered."""
        self.validate_voice_id_format(voice_id)
        if not self.voice_exists(voice_id):
            raise ServiceVoiceNotFoundError(f"Voice ID '{voice_id}' not found")

    def validate_voice_id_format(self, voice_id: str) -> None:
        """Validate voice ID format only (no existence check)."""
        if not voice_id:
            raise ServiceInvalidVoiceIdError("Voice ID cannot be empty")
        if not all(c.isalnum() or c in "-_" for c in voice_id):
            raise ServiceInvalidVoiceIdError(
                "Voice ID can only contain alphanumeric characters, hyphens, and underscores"
            )
        if len(voice_id) > 64:
            raise ServiceInvalidVoiceIdError(
                "Voice ID must be at most 64 characters long"
            )

    async def validate_voice_audio(self, audio_path: str) -> AudioMetadata:
        """Probe reference audio and enforce the service quality floor."""
        metadata = await probe_audio_file(audio_path)
        duration_sec = metadata.duration
        if duration_sec < MIN_REF_AUDIO_DURATION_SEC:
            raise ServiceReferenceAudioTooShortError(
                f"Reference audio is too short: {duration_sec:.2f} seconds. Minimum required is {MIN_REF_AUDIO_DURATION_SEC} seconds."
            )
        if duration_sec > MAX_REF_AUDIO_DURATION_SEC:
            raise ServiceReferenceAudioTooLongError(
                f"Reference audio is too long: {duration_sec:.2f} seconds. Maximum allowed is {MAX_REF_AUDIO_DURATION_SEC} seconds."
            )
        if metadata.channels < 1:
            raise ServiceReferenceAudioInvalidChannelsError(
                "Reference audio must have at least one channel.",
            )
        if metadata.sample_rate < 16000:
            raise ServiceReferenceAudioSampleRateTooLowError(
                "Reference audio sample rate must be at least 16 kHz.",
            )
        return metadata

    async def register_voice(
        self, voice_id: str, ref_audio: str, ref_text: str
    ) -> None:
        """Normalize reference audio and register a new speaker profile."""
        cosyvoice = self.ensure_cosyvoice()

        self.validate_voice_id_format(voice_id)
        # Speaker registration mutates a shared in-memory registry. Keep the
        # duplicate-ID check and the final write in one critical section.
        async with self._speaker_registry_lock:
            if self.voice_exists(voice_id):
                raise ServiceVoiceAlreadyExistsError(f"Voice ID '{voice_id}' already exists")

            metadata = await self.validate_voice_audio(ref_audio)

            wav_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_wav:
                    wav_path = tmp_wav.name

                await audio_file_to_wav(metadata, wav_path, 16000)

                await asyncio.to_thread(
                    cosyvoice.register_speaker,
                    voice_id,
                    wav_path,
                    ref_text,
                )

            finally:
                if wav_path is not None:
                    try:
                        os.unlink(wav_path)
                    except FileNotFoundError:
                        pass

    async def delete_voice(self, voice_id: str) -> None:
        """Delete a registered speaker profile from the in-memory registry."""
        cosyvoice = self.ensure_cosyvoice()

        async with self._speaker_registry_lock:
            self.ensure_voice_exists(voice_id)
            cosyvoice.frontend.spk2info.pop(voice_id, None)

    async def _register_default_voices(self) -> None:
        """Load bundled voices from ``assets/<voice_id>.wav`` + ``.txt`` pairs."""
        assets_dir = PROJECT_ROOT_DIR / "assets"
        if not assets_dir.is_dir():
            _logger.warning(
                "Assets directory not found at %s, skipping default voice registration.",
                assets_dir,
            )
            return
        for file in assets_dir.iterdir():
            if file.suffix.lower() == ".wav":
                voice_id = file.stem
                if self.voice_exists(voice_id):
                    _logger.info(
                        "Default voice '%s' already registered; skipping.",
                        voice_id,
                    )
                    continue
                ref_audio = str(file)
                ref_text_file = assets_dir / f"{voice_id}.txt"
                if not ref_text_file.is_file():
                    _logger.warning(
                        "Reference text file not found for '%s' at %s, skipping.",
                        voice_id,
                        ref_text_file,
                    )
                    continue
                ref_text = ref_text_file.read_text(encoding="utf-8").strip()
                try:
                    await self.register_voice(voice_id, ref_audio, ref_text)
                    _logger.info("Registered default voice '%s' from assets.", voice_id)
                except Exception as e:
                    _logger.error(
                        "Failed to register default voice '%s': %s",
                        voice_id,
                        e,
                        exc_info=True,
                    )

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        response_format: AudioFormat = "pcm",
        instruct_text: str = "",
        speed: float = 1.0,
        text_frontend: bool = True,
        split_sentences: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        """Generate encoded audio chunks for streaming transports."""
        cosyvoice = self.ensure_cosyvoice()
        self.ensure_voice_exists(voice_id)

        mode = TTSMode.INSTRUCT if instruct_text else TTSMode.ZERO_SHOT
        gen = cosyvoice.synthesize(
            text=text,
            voice_id=voice_id,
            mode=mode,
            instruction=instruct_text or None,
            stream=True,
            speed=speed,
            text_frontend=text_frontend,
            split_sentences=split_sentences,
        )

        if response_format == "pcm":
            async for output in gen:
                pcm_data = tensor_to_pcm_s16le(output.get("tts_speech"))
                if pcm_data:
                    yield pcm_data
            return

        ffmpeg_cmd = build_ffmpeg_cmd(response_format, cosyvoice.sample_rate)
        async with AsyncFfmpegProcess(ffmpeg_cmd) as ffmpeg:

            async def pump() -> None:
                try:
                    # Feed raw PCM into ffmpeg while the caller consumes the
                    # encoded stream from ``output_stream``.
                    async for output in gen:
                        pcm_chunk = tensor_to_pcm_s16le(output.get("tts_speech"))
                        if pcm_chunk:
                            await ffmpeg.feed(pcm_chunk)
                finally:
                    exit_code = await ffmpeg.shutdown()
                    if exit_code not in (0, -1):
                        raise AudioTranscodingError(
                            f"ffmpeg exited with code {exit_code} while streaming audio"
                        )

            feeder_task = asyncio.create_task(pump())

            try:
                async for transcoded_chunk in ffmpeg.output_stream():
                    yield transcoded_chunk
                await feeder_task
            except asyncio.CancelledError:
                if not feeder_task.done():
                    feeder_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await feeder_task
                raise
            finally:
                if not feeder_task.done():
                    feeder_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await feeder_task

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        response_format: AudioFormat = "pcm",
        instruct_text: str = "",
        speed: float = 1.0,
        text_frontend: bool = True,
        split_sentences: bool = True,
    ) -> bytes:
        """Generate the full encoded audio payload in memory."""
        cosyvoice = self.ensure_cosyvoice()
        self.ensure_voice_exists(voice_id)

        mode = TTSMode.INSTRUCT if instruct_text else TTSMode.ZERO_SHOT
        pcm_chunks: list[bytes] = []
        async for output in cosyvoice.synthesize(
            text=text,
            voice_id=voice_id,
            mode=mode,
            instruction=instruct_text or None,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
            split_sentences=split_sentences,
        ):
            pcm_data = tensor_to_pcm_s16le(output.get("tts_speech"))
            if pcm_data:
                pcm_chunks.append(pcm_data)

        pcm_bytes = b"".join(pcm_chunks)
        if response_format == "pcm":
            return pcm_bytes

        return await pcm_bytes_transcoding(
            pcm_bytes,
            cosyvoice.sample_rate,
            response_format,
        )

    def cleanup(self) -> None:
        """Best-effort cleanup for torch.distributed process groups."""
        dist = getattr(torch, "distributed", None)
        if dist is None:
            return

        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
                _logger.info("Destroyed torch.distributed process group")
        except Exception:
            _logger.warning(
                "Failed to destroy torch.distributed process group cleanly",
                exc_info=True,
            )
