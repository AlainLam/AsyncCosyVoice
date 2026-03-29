"""HTTP transport for the CosyVoice service."""

import base64
import logging
import os
import tempfile

import aiofiles

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status as HttpCode,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import MAX_UPLOAD_SIZE_MB, load_settings
from app.ffmpeg import (
    AudioFormat,
    AudioProbeError,
    AudioProbeTimeoutError,
    AudioProbeUnavailableError,
    AudioTranscodingError,
)
from app.schemas import (
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechRequest,
    VoiceRegistrationResponse,
)
from app.service import (
    CosyVoiceService,
    ServiceReferenceAudioInvalidChannelsError,
    ServiceReferenceAudioSampleRateTooLowError,
    ServiceReferenceAudioTooLongError,
    ServiceReferenceAudioTooShortError,
    ServiceError,
    ServiceInvalidVoiceIdError,
    ServiceNotInitializedError,
    ServiceVoiceAlreadyExistsError,
    ServiceVoiceNotFoundError,
)

_logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Transport-layer error with a public-facing status, code, and message."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        self.status_code = status_code
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _get_media_type(format: AudioFormat) -> str:
    type_mapping = {
        "pcm": "application/octet-stream",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
    }
    return type_mapping.get(format, "application/octet-stream")


def _validate_audio_upload(file: UploadFile) -> None:
    """Validate the HTTP upload envelope before passing it to the service."""
    if not file.filename:
        raise ApiError(
            status_code=HttpCode.HTTP_400_BAD_REQUEST,
            code="audio_upload_missing_filename",
            detail="Audio file must have a filename.",
        )

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise ApiError(
            status_code=HttpCode.HTTP_400_BAD_REQUEST,
            code="audio_upload_invalid_content_type",
            detail="Uploaded file is not an audio file.",
        )

    file_size = file.size
    if file_size is None:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)

    if file_size == 0:
        raise ApiError(
            status_code=HttpCode.HTTP_400_BAD_REQUEST,
            code="audio_upload_empty",
            detail="Audio file cannot be empty.",
        )
    if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ApiError(
            status_code=HttpCode.HTTP_400_BAD_REQUEST,
            code="audio_upload_too_large",
            detail=f"Audio file size exceeds the maximum limit of {MAX_UPLOAD_SIZE_MB} MB.",
        )


async def _persist_upload(file: UploadFile) -> str:
    """Persist the uploaded audio to a temporary file for ffprobe/ffmpeg."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        async with aiofiles.open(temp_path, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out_file.write(chunk)
        return temp_path
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _build_sse_event(
    event_name: str,
    payload: SpeechAudioDeltaEvent | SpeechAudioDoneEvent,
) -> str:
    return f"event: {event_name}\ndata: {payload.model_dump_json()}\n\n"


def _error_response(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": detail},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("Starting CosyVoice HTTP service")
    settings = load_settings()
    service = CosyVoiceService(settings)
    app.state.settings = settings
    app.state.service = service

    await service.initialize()

    try:
        yield
    finally:
        _logger.info("Shutting down CosyVoice HTTP service")
        service.cleanup()


app = FastAPI(
    title="CosyVoice TTS Backend Service",
    description="A TTS backend service for CosyVoice models.",
    version="3.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_service(request: Request) -> CosyVoiceService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        _logger.error("CosyVoice service not initialized")
        raise ApiError(
            status_code=HttpCode.HTTP_503_SERVICE_UNAVAILABLE,
            code="service_unavailable",
            detail="Service is temporarily unavailable.",
        )
    return service


@app.exception_handler(ApiError)
def handle_api_error(_: Request, exc: ApiError):
    return _error_response(exc.status_code, exc.code, exc.detail)


@app.exception_handler(ServiceError)
def handle_service_error(_: Request, exc: ServiceError):
    _logger.warning("Service error: %s", exc)
    if isinstance(exc, ServiceNotInitializedError):
        status_code = HttpCode.HTTP_503_SERVICE_UNAVAILABLE
        code = "service_unavailable"
        detail = "Service is temporarily unavailable."
    elif isinstance(exc, ServiceVoiceAlreadyExistsError):
        status_code = HttpCode.HTTP_409_CONFLICT
        code = "voice_already_exists"
        detail = str(exc)
    elif isinstance(exc, ServiceVoiceNotFoundError):
        status_code = HttpCode.HTTP_404_NOT_FOUND
        code = "voice_not_found"
        detail = str(exc)
    elif isinstance(exc, ServiceInvalidVoiceIdError):
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        code = "voice_id_invalid"
        detail = str(exc)
    elif isinstance(exc, ServiceReferenceAudioTooShortError):
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        code = "reference_audio_too_short"
        detail = str(exc)
    elif isinstance(exc, ServiceReferenceAudioTooLongError):
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        code = "reference_audio_too_long"
        detail = str(exc)
    elif isinstance(exc, ServiceReferenceAudioInvalidChannelsError):
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        code = "reference_audio_invalid_channels"
        detail = str(exc)
    elif isinstance(exc, ServiceReferenceAudioSampleRateTooLowError):
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        code = "reference_audio_sample_rate_too_low"
        detail = str(exc)
    else:
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        code = "request_validation_failed"
        detail = str(exc)

    return _error_response(status_code, code, detail)


@app.exception_handler(AudioProbeError)
def handle_audio_probe_error(_: Request, exc: AudioProbeError):
    _logger.error("Audio probe error: %s", exc, exc_info=True)
    if isinstance(exc, AudioProbeTimeoutError):
        status_code = HttpCode.HTTP_503_SERVICE_UNAVAILABLE
        detail = "Audio detection timed out. Please try again later."
        code = "audio_probe_timed_out"
    elif isinstance(exc, AudioProbeUnavailableError):
        status_code = HttpCode.HTTP_503_SERVICE_UNAVAILABLE
        detail = "Audio detection unavailable. Please try again later."
        code = "audio_probe_unavailable"
    else:
        status_code = HttpCode.HTTP_400_BAD_REQUEST
        detail = "Audio detection failed. Please try again later."
        code = "audio_probe_failed"
    return _error_response(status_code, code, detail)


@app.exception_handler(AudioTranscodingError)
def handle_audio_transcoding_error(_: Request, exc: AudioTranscodingError):
    _logger.error("Audio transcoding error: %s", exc, exc_info=True)
    return _error_response(
        HttpCode.HTTP_500_INTERNAL_SERVER_ERROR,
        "audio_processing_failed",
        "Audio processing failed. Please try again later.",
    )


@app.exception_handler(HTTPException)
def handle_http_exception(_: Request, exc: HTTPException):
    if exc.status_code == HttpCode.HTTP_404_NOT_FOUND:
        code = "resource_not_found"
        detail = "Resource not found."
    elif exc.status_code == HttpCode.HTTP_405_METHOD_NOT_ALLOWED:
        code = "method_not_allowed"
        detail = "Method not allowed."
    elif exc.status_code == HttpCode.HTTP_401_UNAUTHORIZED:
        code = "authentication_failed"
        detail = "Authentication failed."
    elif exc.status_code == HttpCode.HTTP_403_FORBIDDEN:
        code = "access_denied"
        detail = "Access denied."
    else:
        code = (
            "request_validation_failed"
            if exc.status_code < 500
            else "internal_error"
        )
        detail = (
            "Request validation failed."
            if exc.status_code < 500
            else "An unexpected error occurred. Please try again later."
        )
    return _error_response(exc.status_code, code, detail)


@app.exception_handler(Exception)
def handle_exception(_: Request, exc: Exception):
    _logger.error("Unhandled exception: %s", exc, exc_info=True)
    return _error_response(
        HttpCode.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred. Please try again later.",
    )


@app.get("/health", summary="Health check endpoint")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/voices/register",
    summary="Register a new voice",
    status_code=HttpCode.HTTP_201_CREATED,
)
async def register_voice(
    voice_id: str = Form(..., description="Unique identifier for the voice to register"),
    ref_audio: UploadFile = File(
        ...,
        description="Reference audio file for voice registration.",
    ),
    ref_text: str = Form(
        ...,
        description="Transcription of the reference audio.",
    ),
    service: CosyVoiceService = Depends(get_service),
) -> VoiceRegistrationResponse:
    service.validate_voice_id_format(voice_id)
    _validate_audio_upload(ref_audio)

    temp_path: str | None = None
    try:
        temp_path = await _persist_upload(ref_audio)
        await service.register_voice(voice_id, temp_path, ref_text)
        return VoiceRegistrationResponse(voice_id=voice_id)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


@app.get("/v1/voices", summary="List registered voices")
async def list_voices(
    service: CosyVoiceService = Depends(get_service),
) -> list[str]:
    return service.list_voices()


@app.delete(
    "/v1/voices/{voice_id}",
    summary="Delete a registered voice",
    status_code=HttpCode.HTTP_204_NO_CONTENT,
)
async def delete_voice(
    voice_id: str,
    service: CosyVoiceService = Depends(get_service),
) -> None:
    await service.delete_voice(voice_id)


@app.post("/v1/audio/speech", summary="Generate speech from text input")
async def generate_speech(
    speech_req: SpeechRequest,
    service: CosyVoiceService = Depends(get_service),
) -> Response:
    voice_id = speech_req.voice
    service.ensure_voice_exists(voice_id)
    
    response_format = speech_req.response_format
    media_type = _get_media_type(response_format)

    if speech_req.disable_stream:
        audio_data = await service.synthesize(
            text=speech_req.input,
            voice_id=voice_id,
            response_format=response_format,
            instruct_text=speech_req.instructions or "",
            speed=speech_req.speed,
        )
        return Response(content=audio_data, media_type=media_type)

    if speech_req.stream_format == "sse":

        async def sse_generator() -> AsyncGenerator[str, None]:
            async for chunk in service.synthesize_stream(
                text=speech_req.input,
                voice_id=voice_id,
                response_format=response_format,
                instruct_text=speech_req.instructions or "",
                speed=speech_req.speed,
            ):
                yield _build_sse_event(
                    "speech.audio.delta",
                    SpeechAudioDeltaEvent(data=base64.b64encode(chunk).decode()),
                )

            yield _build_sse_event("speech.audio.done", SpeechAudioDoneEvent())

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    return StreamingResponse(
        service.synthesize_stream(
            text=speech_req.input,
            voice_id=voice_id,
            response_format=response_format,
            instruct_text=speech_req.instructions or "",
            speed=speech_req.speed,
        ),
        media_type=media_type,
    )
