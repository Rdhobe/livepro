import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import boto3
import sounddevice as sd
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from botocore.config import Config

from config import AWS_REGION, BEDROCK_MODEL_ID


AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCK_MS = 40
AUDIO_BLOCK_SIZE = int(AUDIO_SAMPLE_RATE * (AUDIO_BLOCK_MS / 1000))


@dataclass
class AssistantConfig:
    system_prompt: str = (
        "You are a real-time voice assistant. Keep answers short, clear, and conversational. "
        "Prefer 1-3 short sentences unless the user asks for detail."
    )
    temperature: float = 0.5
    max_gen_len: int = 300
    voice_id: str = "Joanna"


class BedrockLlamaStreamer:
    def __init__(self, region: str, model_id: str):
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 1, "mode": "standard"}),
        )

    def _format_prompt(self, messages: list[dict], system_prompt: str) -> str:
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
        )
        for msg in messages:
            prompt += (
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def stream_response(self, messages: list[dict], system_prompt: str, temperature: float, max_gen_len: int):
        body = {
            "prompt": self._format_prompt(messages, system_prompt),
            "max_gen_len": max_gen_len,
            "temperature": temperature,
        }
        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        for event in response["body"]:
            payload = event.get("chunk", {}).get("bytes")
            if not payload:
                continue
            chunk = json.loads(payload)
            token = chunk.get("generation", "")
            if token:
                yield token


class PollySpeaker:
    def __init__(self, region: str, voice_id: str = "Joanna"):
        self.voice_id = voice_id
        self.client = boto3.client("polly", region_name=region)

    def speak(self, text: str):
        if not text.strip():
            return
        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId=self.voice_id,
            SampleRate=str(AUDIO_SAMPLE_RATE),
            Engine="neural",
        )
        pcm_bytes = response["AudioStream"].read()
        audio = memoryview(pcm_bytes).cast("h")
        sd.play(audio, samplerate=AUDIO_SAMPLE_RATE)
        sd.wait()


class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, on_final_text: Callable[[str], asyncio.Future]):
        super().__init__(output_stream)
        self.on_final_text = on_final_text

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if result.is_partial:
                continue
            for alt in result.alternatives:
                text = alt.transcript.strip()
                if text:
                    await self.on_final_text(text)


class LiveVoiceAssistant:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.messages: list[dict] = []
        self.bedrock = BedrockLlamaStreamer(AWS_REGION, BEDROCK_MODEL_ID)
        self.speaker = PollySpeaker(AWS_REGION, config.voice_id)
        self.transcribe = TranscribeStreamingClient(region=AWS_REGION)
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.stop_event = asyncio.Event()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[audio warning] {status}")
        chunk = bytes(indata)
        try:
            self.audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    async def _write_mic_audio(self, input_stream):
        while not self.stop_event.is_set():
            chunk = await self.audio_queue.get()
            await input_stream.send_audio_event(audio_chunk=chunk)
        await input_stream.end_stream()

    async def _on_user_text(self, text: str):
        if not text:
            return
        print(f"\n🧑 {text}")
        self.messages.append({"role": "user", "content": text})

        print("🤖 ", end="", flush=True)
        assistant_text = ""
        sentence_buffer = ""

        for token in self.bedrock.stream_response(
            self.messages,
            self.config.system_prompt,
            self.config.temperature,
            self.config.max_gen_len,
        ):
            print(token, end="", flush=True)
            assistant_text += token
            sentence_buffer += token

            if any(sentence_buffer.endswith(p) for p in (".", "?", "!", "\n")):
                to_speak = sentence_buffer.strip()
                sentence_buffer = ""
                await asyncio.to_thread(self.speaker.speak, to_speak)

        if sentence_buffer.strip():
            await asyncio.to_thread(self.speaker.speak, sentence_buffer.strip())

        print("\n")
        self.messages.append({"role": "assistant", "content": assistant_text.strip()})

    async def run(self):
        print("Live voice assistant started. Speak naturally (Ctrl+C to exit).")
        stream = await self.transcribe.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=AUDIO_SAMPLE_RATE,
            media_encoding="pcm",
        )

        handler = TranscriptHandler(stream.output_stream, self._on_user_text)

        with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=AUDIO_BLOCK_SIZE,
            dtype="int16",
            channels=AUDIO_CHANNELS,
            callback=self._audio_callback,
        ):
            await asyncio.gather(
                self._write_mic_audio(stream.input_stream),
                handler.handle_events(),
            )


async def main():
    assistant = LiveVoiceAssistant(AssistantConfig())
    try:
        await assistant.run()
    except KeyboardInterrupt:
        assistant.stop_event.set()
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
