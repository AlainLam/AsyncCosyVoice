"""Async FFmpeg/FFprobe helpers for audio probing and transcoding.

This module provides:
- Metadata probing via ``ffprobe``.
- Streaming PCM transcoding via a managed async ``ffmpeg`` subprocess.
- File-to-file WAV normalization.
"""

import asyncio
import json
import logging
import os
import shutil

from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Optional

_logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 4096
_DEFAULT_QUEUE_SIZE = 100
_DEFAULT_PROBE_TIMEOUT = 10.0
_TERMINATE_WAIT_TIMEOUT = 2.0

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

_ENCODER_OPTIONS: dict[AudioFormat, list[str]] = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"],
    "opus": ["-c:a", "libopus", "-b:a", "64k"],
    "aac": ["-c:a", "aac", "-b:a", "128k"],
    "flac": ["-c:a", "flac"],
    "wav": ["-c:a", "pcm_s16le"],
    "pcm": ["-c:a", "pcm_s16le"],
}

_PIPE_FORMAT_OPTIONS: dict[AudioFormat, list[str]] = {
    "mp3": ["-f", "mp3", "pipe:1"],
    "opus": ["-f", "opus", "pipe:1"],
    "aac": ["-f", "adts", "pipe:1"],
    "flac": ["-f", "flac", "pipe:1"],
    "wav": ["-f", "wav", "pipe:1"],
    "pcm": ["-f", "s16le", "pipe:1"],
}


class AudioError(Exception):
    """Base class for all audio-processing errors in this module."""

    pass


class AudioProbeError(AudioError):
    """Custom exception to indicate ffprobe encountered an error."""

    pass


class AudioProbeUnavailableError(AudioProbeError):
    """Custom exception to indicate ffprobe is not available."""

    pass


class AudioProbeTimeoutError(AudioProbeError):
    """Custom exception to indicate ffprobe timed out."""

    pass


class AudioTranscodingError(AudioError):
    """Custom exception to indicate ffmpeg transcoding failed."""

    pass


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata for an audio file."""

    path: str
    codec_name: str
    sample_rate: int
    channels: int
    format_names: list[str]
    duration: float


class AsyncFfmpegProcess:
    """
    Asynchronous wrapper around ffmpeg process using a producer-consumer
    pattern to handle real-time audio transcoding without blocking.
    """

    def __init__(
        self, command_args: list[str], queue_size: int = _DEFAULT_QUEUE_SIZE
    ) -> None:
        """
        Initialize ffmpeg process wrapper.

        :param command_args: Complete list of ffmpeg arguments.
        :param queue_size: Max number of chunks to buffer before applying backpressure.
        """
        if not command_args:
            raise ValueError("command_args cannot be empty")
        if queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")

        self._args = command_args
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._output_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(
            maxsize=queue_size
        )
        self._read_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Launch the ffmpeg subprocess and start the stdout reader task."""
        if self._proc and self._proc.returncode is None:
            raise RuntimeError("FFmpeg process already started")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AudioProbeUnavailableError("ffmpeg executable not found") from e

        # Start background task to drain stdout to avoid buffer bloat
        self._read_task = asyncio.create_task(self._read_stdout())

    async def _read_stdout(self) -> None:
        """Internal loop to read data from ffmpeg stdout and put into queue."""
        try:
            while self._proc and self._proc.stdout:
                chunk = await self._proc.stdout.read(_DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                await self._output_queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.debug("FFmpeg stdout reader stopped unexpectedly", exc_info=True)
        finally:
            # Signal EOF (None) to the consumer of output_stream
            await self._output_queue.put(None)

    async def feed(self, data: bytes) -> None:
        """
        Feed raw audio data to ffmpeg stdin.
        Applies backpressure via drain().
        """
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("FFmpeg process not started or already closed.")
        if not data:
            return

        try:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError):
            # Ffmpeg might have exited early due to error or finished processing
            _logger.debug("FFmpeg stdin closed while feeding data", exc_info=True)

    async def output_stream(self) -> AsyncGenerator[bytes, None]:
        """Yields transcoded data chunks as they become available."""
        while True:
            chunk = await self._output_queue.get()
            if chunk is None:
                break
            yield chunk

    async def close(self) -> None:
        """
        Gracefully close stdin only.

        This mirrors asyncio subprocess behavior where callers explicitly choose
        whether/when to await process termination via ``wait()``.
        """
        if self._proc and self._proc.stdin:
            if not self._proc.stdin.is_closing():
                try:
                    self._proc.stdin.close()
                    await self._proc.stdin.wait_closed()
                except Exception:
                    _logger.debug("Failed to close FFmpeg stdin cleanly", exc_info=True)

    async def shutdown(
        self, timeout: float = _TERMINATE_WAIT_TIMEOUT, graceful: bool = True
    ) -> int:
        """
        Reap process resources with bounded wait.

        If ``graceful`` is True, close stdin and wait for natural exit first.
        Then escalate to SIGTERM and SIGKILL if still running.
        """
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")

        if graceful:
            await self.close()
            exit_code = await self.wait(timeout=timeout)
            if exit_code != -1:
                if self._read_task:
                    try:
                        await self._read_task
                    except asyncio.CancelledError:
                        pass
                return exit_code

        await self.terminate()
        exit_code = await self.wait(timeout=timeout)
        if exit_code == -1 and self._proc and self._proc.returncode is None:
            await self.kill()
            exit_code = await self.wait(timeout=timeout)

        if self._read_task:
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        return exit_code

    async def terminate(self) -> None:
        """Sends SIGTERM to ffmpeg. Does NOT wait for exit."""
        if self._proc:
            try:
                self._proc.terminate()
            except ProcessLookupError:
                pass

    async def kill(self) -> None:
        """Sends SIGKILL to ffmpeg. Does NOT wait for exit."""
        if self._proc:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    async def poll(self) -> Optional[int]:
        """Check the return code of the ffmpeg process."""
        if self._proc:
            return self._proc.returncode
        return None

    async def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for ffmpeg process to finish and return exit code."""
        if not self._proc:
            return -1
        try:
            return await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            _logger.debug("Timed out while waiting for FFmpeg process exit")
            return -1
        except Exception:
            _logger.exception("Unexpected error while waiting for FFmpeg process")
            return -1

    async def __aenter__(self) -> "AsyncFfmpegProcess":
        """Enter async context manager and start process."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager.
        Ensures resources are reaped without risking indefinite hangs.
        """
        await self.shutdown(timeout=_TERMINATE_WAIT_TIMEOUT, graceful=False)


def _encoder_options(output_format: AudioFormat) -> list[str]:
    options = _ENCODER_OPTIONS.get(output_format)
    if options is None:
        raise ValueError(f"Unsupported format: {output_format}")
    return options


def _file_format_options(output_format: AudioFormat) -> list[str]:
    if output_format == "pcm":
        return ["-f", "s16le"]
    return []


def build_ffmpeg_cmd(format: AudioFormat, sample_rate: int) -> list[str]:
    """
    Builds the ffmpeg command optimized for streaming pipes.
    -nostdin: Prevents ffmpeg from grabbing the terminal.
    -v error: Keeps stderr clean for purely data-driven logging.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    cmd: list[str] = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-vn",
        "-sn",
        "-dn",
    ]

    cmd.extend(_encoder_options(format))
    cmd.extend(_PIPE_FORMAT_OPTIONS[format])
    return cmd


async def pcm_bytes_transcoding(
    data: bytes,
    sample_rate: int,
    output_format: AudioFormat,
) -> bytes:
    """Convert raw PCM (s16le, mono) bytes to the requested audio format.

    Uses a single ``communicate()`` call – the right tool when the complete
    PCM buffer is already in memory (non-streaming / batch path). When the
    target format is ``pcm``, the input bytes are returned directly.

    :param data: Raw PCM audio bytes (signed 16-bit little-endian, mono).
    :param sample_rate: Sample rate of the input PCM data in Hz
                        (e.g. 16000 for CosyVoice1, 24000 for CosyVoice2/3).
    :param output_format: Target :data:`AudioFormat`.
    :returns: Audio bytes in the requested format.
    :raises ValueError: If ``output_format`` is unsupported or ``sample_rate`` is invalid.
    :raises AudioProbeUnavailableError: If the ``ffmpeg`` executable is not found.
    :raises AudioTranscodingError: If ffmpeg exits with a non-zero return code.
    """
    if output_format == "pcm" or not data:
        return data

    cmd = build_ffmpeg_cmd(output_format, sample_rate)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise AudioProbeUnavailableError("ffmpeg executable not found") from e

    stdout, stderr = await proc.communicate(input=data)

    if proc.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace").strip()
        raise AudioTranscodingError(
            f"ffmpeg exited with code {proc.returncode}: {error_msg}"
        )

    return stdout


async def audio_file_to_wav(
    input_metadata: AudioMetadata,
    output_path: str,
    output_sample_rate: Literal[16000, 24000],
) -> bool:
    """Normalize an audio file to mono WAV encoded as ``pcm_s16le``.

    This helper is intentionally explicit: it always invokes ffmpeg and writes
    a WAV file using 16-bit PCM, mono, and a caller-selected sample rate.

    :param input_metadata: Metadata for the input file (from ``probe_audio_file``).
    :param output_path: Destination path for the normalized WAV file.
    :param output_sample_rate: Output sample rate in Hz. Must be ``16000`` or ``24000``.
    :returns: ``True`` when transcoding succeeds.
    :raises AudioProbeError: If the input file does not exist.
    :raises AudioProbeUnavailableError: If the ``ffmpeg`` executable is not found.
    :raises AudioTranscodingError: If ffmpeg exits with a non-zero return code.
    """
    if not os.path.isfile(input_metadata.path):
        raise AudioProbeError(f"Audio file does not exist: {input_metadata.path}")

    if (
        "wav" in input_metadata.format_names
        and input_metadata.codec_name == "pcm_s16le"
        and input_metadata.channels == 1
        and input_metadata.sample_rate == output_sample_rate
    ):
        shutil.copyfile(input_metadata.path, output_path)
        return True

    cmd: list[str] = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        input_metadata.path,
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(output_sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "-y",
        output_path,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise AudioProbeUnavailableError("ffmpeg executable not found") from e

    _, stderr = await proc.communicate()
    exit_code = proc.returncode

    if exit_code != 0:
        error_msg = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise AudioTranscodingError(
            f"ffmpeg exited with code {exit_code} while transcoding {input_metadata.path!r}: {error_msg}"
        )

    return True


async def probe_audio_file(
    input_path: str, timeout: Optional[float] = None
) -> AudioMetadata:
    """Use ffprobe to extract structured audio metadata only.

    :param input_path: Path to the audio file to probe.
    :param timeout: Optional timeout in seconds for the ffprobe command.
    :return: AudioMetadata object containing extracted information.
    :raises AudioProbeUnavailableError: If ffprobe is not installed or not found.
    :raises AudioProbeTimeoutError: If ffprobe execution exceeds the specified timeout.
    :raises AudioProbeError: For any other errors during ffprobe execution or output parsing.
    """
    if timeout is None:
        timeout = _DEFAULT_PROBE_TIMEOUT

    if not input_path:
        raise ValueError("input_path cannot be empty")
    if timeout <= 0:
        raise ValueError("timeout must be a positive number")
    if not os.path.isfile(input_path):
        raise AudioProbeError(f"Audio file does not exist: {input_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=format_name,duration",
        "-of",
        "json",
        input_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        _logger.error("ffprobe executable not found")
        raise AudioProbeUnavailableError("ffprobe executable not found") from e

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        _logger.error("ffprobe timed out after %s seconds", timeout)
        raise AudioProbeTimeoutError(
            f"ffprobe timed out after {timeout} seconds"
        ) from e

    if proc.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace").strip()
        _logger.error("ffprobe error: %s", error_msg)
        raise AudioProbeError(f"ffprobe error: {error_msg}")

    try:
        info = json.loads(stdout.decode("utf-8", errors="replace"))
        streams = info.get("streams") or []
        if not streams:
            raise ValueError("No audio stream found")

        stream = streams[0]
        fmt = info.get("format") or {}
        format_names = [name for name in fmt.get("format_name", "").split(",") if name]

        return AudioMetadata(
            path=input_path,
            codec_name=str(stream.get("codec_name", "")).lower(),
            sample_rate=int(stream.get("sample_rate") or 0),
            channels=int(stream.get("channels") or 0),
            format_names=format_names,
            duration=float(fmt.get("duration") or 0),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        _logger.error(f"Failed to parse ffprobe output: {e}")
        raise AudioProbeError(f"Failed to parse ffprobe output: {e}") from e
