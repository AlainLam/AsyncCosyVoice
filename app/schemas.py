from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.ffmpeg import AudioFormat


class BaseSchema(BaseModel):
    """Common Pydantic settings for request and response schemas."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
        from_attributes=True,
        arbitrary_types_allowed=False,
        extra="ignore",
    )


class VoiceRegistrationResponse(BaseSchema):
    """Response returned after a voice is registered successfully."""

    voice_id: str = Field(..., description="Registered voice identifier.")


class SpeechRequest(BaseSchema):
    """OpenAI-compatible request body for `/v1/audio/speech`."""

    model_config = ConfigDict(extra="allow")

    input: str = Field(
        ...,
        description="The text to generate audio for. The maximum length is 4096 characters.",
        min_length=1,
        max_length=4096,
    )
    model: Literal["cosyvoice"] = Field(
        default="cosyvoice",
        description="Model identifier. Fixed to `cosyvoice` for this service.",
    )
    voice: str = Field(
        ...,
        description="The voice identifier to use when generating the audio.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Optional style instructions, compatible with the OpenAI audio speech API.",
    )
    response_format: AudioFormat = Field(
        default="mp3",
        description="The audio format to return. Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.",
    )
    speed: float = Field(
        default=1.0,
        description="The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default.",
        ge=0.25,
        le=4.0,
    )
    stream_format: Optional[Literal["sse", "audio"]] = Field(
        default=None,
        description="The streaming output format. Supported formats are `sse` and `audio`.",
    )

    # extra_body fields ---------------------------------------------------------------------
    disable_stream: bool = Field(
        default=False,
        description="Whether to disable streaming output, even if a stream_format is specified. This can be used to force a non-streaming response for clients that do not support streaming.",
    )


class SpeechAudioDeltaEvent(BaseSchema):
    """OpenAI-style SSE payload for an incremental audio chunk."""

    type: Literal["speech.audio.delta"] = Field(
        default="speech.audio.delta",
        description="Event type for a streamed audio chunk.",
    )
    data: str = Field(
        ...,
        description="Base64-encoded audio bytes for the next streamed chunk.",
    )


class SpeechAudioDoneEvent(BaseSchema):
    """OpenAI-style SSE payload indicating the audio stream has finished."""

    type: Literal["speech.audio.done"] = Field(
        default="speech.audio.done",
        description="Event type emitted once audio generation has completed.",
    )
