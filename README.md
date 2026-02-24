# LivePro: Hands-Free Voice Chat with Bedrock (Open-Source STT/TTS)

This version removes paid cloud speech services and uses a local open-source voice stack:
- **Vosk** (offline STT)
- **pyttsx3** (offline TTS using local speech engine)
- **Bedrock Llama 3 streaming** for model responses

## Quick start

1. Create config:

```bash
cp config.py.example config.py
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download a Vosk model (example):

```bash
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

4. Run:

```bash
python live_voice_chat.py
```

## Why this matches your request

- No paid STT/TTS APIs.
- Speech recognition runs locally via Vosk.
- Speech synthesis runs locally via pyttsx3.
- Keyboard-free loop: speak → model replies in voice.

## AWS requirements (model only)

- Bedrock model access enabled for your `BEDROCK_MODEL_ID`.
- IAM permissions for:
  - `bedrock:InvokeModelWithResponseStream`

## Notes

- For better voice quality, install system voices (e.g., `espeak-ng` on Linux).
- For lower latency and better accuracy, prefer a larger Vosk model on stronger hardware.
- Next upgrade: add VAD + barge-in so assistant stops speaking when you interrupt.
