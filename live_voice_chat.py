import asyncio
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import boto3
import pyttsx3
import sounddevice as sd
from botocore.config import Config
from vosk import KaldiRecognizer, Model

from config import AWS_REGION, BEDROCK_MODEL_ID, VOSK_MODEL_PATH

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
    speech_rate: int = 180


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


class LocalSpeaker:
    def __init__(self, rate: int = 180):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self._lock = threading.Lock()

    def speak(self, text: str):
        if not text.strip():
            return
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()


class LocalVoskTranscriber:
    def __init__(self, model_path: str):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, AUDIO_SAMPLE_RATE)
        self.recognizer.SetWords(False)
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)

    def audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            print(f"[audio warning] {status}")
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def listen_for_final_text(self, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                data = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    yield text


def validate_environment() -> None:
    if not Path(VOSK_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"VOSK model not found at '{VOSK_MODEL_PATH}'. "
            "Download a model and update VOSK_MODEL_PATH in config.py."
        )


class LiveVoiceAssistant:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.messages: list[dict] = []
        self.bedrock = BedrockLlamaStreamer(AWS_REGION, BEDROCK_MODEL_ID)
        self.speaker = LocalSpeaker(rate=config.speech_rate)
        self.transcriber = LocalVoskTranscriber(VOSK_MODEL_PATH)
        self.stop_event = threading.Event()

    async def _handle_user_text(self, text: str):
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
                segment = sentence_buffer.strip()
                sentence_buffer = ""
                await asyncio.to_thread(self.speaker.speak, segment)

        if sentence_buffer.strip():
            await asyncio.to_thread(self.speaker.speak, sentence_buffer.strip())

        print("\n")
        self.messages.append({"role": "assistant", "content": assistant_text.strip()})

    async def run(self):
        print("Live voice assistant started. Speak naturally (Ctrl+C to exit).")
        transcript_iter = self.transcriber.listen_for_final_text(self.stop_event)

        with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=AUDIO_BLOCK_SIZE,
            dtype="int16",
            channels=AUDIO_CHANNELS,
            callback=self.transcriber.audio_callback,
        ):
            while not self.stop_event.is_set():
                text = await asyncio.to_thread(lambda: next(transcript_iter, None))
                if text:
                    await self._handle_user_text(text)


def main():
    validate_environment()
    assistant = LiveVoiceAssistant(AssistantConfig())
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        assistant.stop_event.set()
        print("\nStopped.")


if __name__ == "__main__":
    main()
