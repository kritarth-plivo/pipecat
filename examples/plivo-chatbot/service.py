#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import wave
from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from time import sleep
import requests
import json

from loguru import logger

from pipecat.audio.utils import calculate_audio_volume, exp_smoothing
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
    UserImageRequestFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.time import seconds_to_nanoseconds
from pipecat.services.ai_services import *

# from dg import AudioLengthFrame, TextIDFrame

class TextService(AIService):
    def __init__(
        self,
        *,
        # if True, TTSService will push TextFrames and LLMFullResponseEndFrames,
        # otherwise subclass must do it
        push_text_frames: bool = False,
        text_filter: Optional[BaseTextFilter] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._push_text_frames: bool = push_text_frames
        self._settings: Dict[str, Any] = {}
        self._text_filter: Optional[BaseTextFilter] = text_filter
        self._last_text: str = ""
        self._last_length_ms: float = 0.0
        self._last_start_time: float = 0.0

        from phi.agent import Agent
        from phi.model.openai import OpenAIChat
        from phi.tools.duckduckgo import DuckDuckGo
        self.web_agent = Agent(
            name="Web Agent",
            model=OpenAIChat(id="gpt-4o"),
            tools=[DuckDuckGo()],
            instructions=["Answer concisely and always mention sources in a single word."],
            show_tool_calls=True,
            markdown=False,
        )

    def can_generate_metrics(self) -> bool:
        return True

    # Converts the text to text.
    @abstractmethod
    async def run_t3(self, text: str) -> AsyncGenerator[Frame, None]:
        import uuid
        id = str(uuid.uuid4())
        logger.debug(f"Sending Query: [{id}]: [{text}]")

        import time
        if not (self._last_length_ms == 0.0 or self._last_start_time == 0.0):
            # check if time elapsed is more than this value
            elapsed_ms = time.time() * 1000 - self._last_start_time
            print(elapsed_ms)
            if elapsed_ms < self._last_length_ms:
                print(f'TTS was INTERRUPTED for sure {elapsed_ms*100/self_last_length_ms} % was spoken')


        await self.start_ttfb_metrics()

        try:
            # make request to queue execution service
            # yield each text

            s = requests.Session()
            #url = 'https://queue-qa.voice.plivodev.com/execution_service/v1/voice/stt'
            url = 'http://pipecat.plivodev.com:9999/completion/stream/50'
            headers = {"Content-Type": "application/json"}
            data = {"question": text}
            with s.post(url, headers=headers, data=json.dumps(data), stream=True) as resp:
                for line in resp.iter_content(chunk_size=1024):
                    if line:
                        if self._last_start_time == 0.0:
                            self._last_text = text
                            self._last_length_ms = 0.0
                            self._last_start_time = time.time() * 1000
                        await self.stop_ttfb_metrics()
                        print(line.decode('utf-8'), end='', flush=True)
                        # yield TextIDFrame(id, line.decode('utf-8'))
                        yield TextFrame(line.decode('utf-8'))
            print('', flush=True)
            yield LLMFullResponseEndFrame()
            # for delta in self.web_agent.run(text, stream=True):
            #     yield TextFrame(delta.content)

            self._last_text = text
        except Exception as e:
            logger.error(f"{self} exception: {e}")

        await self.stop_ttfb_metrics()

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self._process_text_frame(frame)
        # elif isinstance(frame, AudioLengthFrame):
        #     self._last_length_ms += frame.length_ms
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        if self._text_filter:
            self._text_filter.handle_interruption()
        await self.push_frame(frame, direction)

    async def _process_text_frame(self, frame: TextFrame):
        text: str | None = None
        text = frame.text

        if text:
            await self._push_tts_frames(text)

    async def _push_tts_frames(self, text: str):
        # Don't send only whitespace. This causes problems for some TTS models. But also don't
        # strip all whitespace, as whitespace can influence prosody.
        if not text.strip():
            return

        await self.start_processing_metrics()
        if self._text_filter:
            self._text_filter.reset_interruption()
            text = self._text_filter.filter(text)
        await self.process_generator(self.run_t3(text))
        await self.stop_processing_metrics()
        if self._push_text_frames:
            # We send the original text after the audio. This way, if we are
            # interrupted, the text is not added to the assistant context.
            await self.push_frame(TextFrame(text))
