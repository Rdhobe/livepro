# LivePro: Hands-Free Voice Chat with Bedrock (Open-Source STT/TTS)

This project uses a local open-source speech stack and Bedrock for the LLM:
- **Vosk** (offline STT)
- **pyttsx3** (offline TTS via local speech engine)
- **Bedrock Llama 3 streaming** for model responses
# LivePro: Hands-Free Voice Chat with Bedrock

A low-latency voice loop using:
- **Amazon Transcribe Streaming** for speech-to-text (STT)
- **Bedrock Llama 3 streaming** for real-time model output
- **Amazon Polly Neural** for text-to-speech (TTS)

## Quick start

1. Create config:

```bash
cp config.py.example config.py
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download a Vosk model:

```bash
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

4. Run:
3. Run:

```bash
python live_voice_chat.py
```

## Why this matches your request

- No paid STT APIs.
- No paid TTS APIs.
- Keyboard-free loop: speak → model responds in voice.

## AWS requirements (model only)

- Bedrock model access enabled for your `BEDROCK_MODEL_ID`.
- IAM permission:
  - `bedrock:InvokeModelWithResponseStream`

## Notes

- If startup fails with “VOSK model not found”, set `VOSK_MODEL_PATH` in `config.py`.
- Install system voices for better TTS quality (e.g., `espeak-ng` on Linux).
- For higher recognition accuracy, use a larger Vosk model.
## Notes for low latency

- Uses 40ms audio chunks for near real-time transcription.
- Streams model tokens and starts speaking at sentence boundaries.
- Keeps model responses concise through the system prompt.

## AWS requirements

- Bedrock model access enabled for your `BEDROCK_MODEL_ID`.
- IAM permissions for:
  - `bedrock:InvokeModelWithResponseStream`
  - `transcribe:StartStreamTranscriptionWebSocket`
  - `polly:SynthesizeSpeech`

## Next upgrades for production

- Add wake-word detection to avoid accidental turn captures.
- Add barge-in (interrupt TTS when user starts speaking).
- Add VAD endpointing + silence timeout for tighter turn-taking.
- Add conversation persistence and observability.
