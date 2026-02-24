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

3. Run:

```bash
python live_voice_chat.py
```

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
