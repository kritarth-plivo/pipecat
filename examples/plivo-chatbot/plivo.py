#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json

from pydantic import BaseModel

from pipecat.frames.frames import AudioRawFrame, Frame, InputAudioRawFrame, StartInterruptionFrame
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class PlivoFrameSerializer(FrameSerializer):
    class InputParams(BaseModel):
        sample_rate: int = 16000

    def __init__(self, stream_id: str, params: InputParams = InputParams()):
        self._stream_id = stream_id
        self._params = params

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.TEXT

    def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            payload = base64.b64encode(frame.audio).decode("utf-8")
            answer = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": self._params.sample_rate,
                    "payload": payload
                },
            }
            return json.dumps(answer)

        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clearAudio", "streamId": self._stream_id}
            return json.dumps(answer)

        return None

    def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] != "media":
            return None
        else:
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            audio_frame = InputAudioRawFrame(
                audio=payload, num_channels=1, sample_rate=self._params.sample_rate
            )
            return audio_frame
