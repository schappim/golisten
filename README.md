# golisten

A self-contained command-line tool for speech-to-text. Give it an MP3, get text
back. By default it runs [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
with Parakeet on your own machine; it also drives
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), or the OpenAI,
ElevenLabs, and Deepgram APIs. Written in Go, no ffmpeg required for MP3 or
WAV — just a single binary.

This is the mirror image of [gospeak](https://github.com/schappim/gospeak):
gospeak turns text into speech, golisten turns speech into text. Same flag
style, same provider names, same `OPENAI_API_KEY` / `ELEVENLABS_API_KEY` /
`DEEPGRAM_API_KEY` environment variables.

```bash
golisten interview.mp3
cat note.mp3 | golisten -p openai | gospeak -v nova
```

## Features

- **Local first** — transcribe.cpp and whisper.cpp run on your machine, with no
  API key and no audio leaving it
- **Parakeet by default** — faster and more accurate than the whisper models
  most people have installed, and it punctuates properly
- **Canary, Whisper, Moonshine, Voxtral, Granite and more** via transcribe.cpp
- **Cloud when you want it** — OpenAI, ElevenLabs, and Deepgram
- **No ffmpeg required** for MP3 and WAV — both are decoded, downmixed, and
  resampled to 16 kHz in pure Go (ffmpeg is used only for other containers)
- **Finds your install** — picks whichever engine you have, and auto-discovers
  binaries and models, including pre-2024 whisper.cpp checkouts that still
  build a binary called `main`
- **Output as txt, SRT, VTT, or JSON**, with timestamps and speaker labels
- **Auto-chunking for long audio** — splits at quiet points and stitches the
  transcript back onto one timeline, so a two-hour recording never hits a
  provider's upload limit
- **Automatic retries** — up to 3 attempts per chunk with exponential backoff
- Reads a file or stdin, writes a file or stdout

## Requirements

- macOS, Linux, or Windows
- For local backends: whisper.cpp or transcribe.cpp, plus a model
- For cloud backends: an API key for your chosen provider
- ffmpeg, only if you feed it something other than MP3 or WAV

## Installation

### Build from Source

```bash
git clone https://github.com/schappim/golisten.git
cd golisten
go build -o golisten .
sudo cp golisten /usr/local/bin/
```

### Get an engine and a model

The default backend is transcribe.cpp running Parakeet:

```bash
git clone https://github.com/handy-computer/transcribe.cpp
cd transcribe.cpp && cmake -B build && cmake --build build -j
cp build/bin/transcribe-cli ~/.cache/golisten/bin/     # or anywhere on PATH

golisten --download parakeet   # into ~/.cache/golisten/models
golisten interview.mp3
```

whisper.cpp is the easier install, and golisten falls back to it automatically:

```bash
brew install whisper-cpp
golisten --download base.en
golisten interview.mp3
```

`--download` names:

| Engine | Names |
|---|---|
| transcribe.cpp | `parakeet` (default), `parakeet-q8`, `parakeet-f16`, `parakeet-v2`, `parakeet-en`, `parakeet-ctc`, `parakeet-rnnt` |
| whisper.cpp | `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`, `large-v3-turbo`, and the quantised variants |

### Choosing the Backend

With no `-p`, golisten uses the first local engine that has both a binary and a
model: **transcribe.cpp**, then **whisper.cpp**, then **parakeet-cli**. Parakeet
has no translation task, so `--translate` starts at whisper.cpp instead.

An explicit `-m` also selects the engine — a `.gguf` name resolves to
transcribe.cpp, a `ggml-*.bin` name to whisper.cpp. `golisten -V` prints what
was chosen:

```
$ golisten -V interview.mp3
Backend: transcribe (auto-selected)
Engine:  /Users/you/.cache/golisten/bin/transcribe-cli
Model:   /Users/you/.cache/golisten/models/parakeet-tdt-0.6b-v3-Q4_K_M.gguf
```

### Configuration

Set an API key only if you want a cloud backend — the same variables gospeak
uses:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export DEEPGRAM_API_KEY="your-deepgram-api-key"
```

Or pass the key directly with `--token`.

## Usage

### Basic Usage (local, no API key)

```bash
# Transcribe a file
golisten interview.mp3

# Pipe audio in
cat interview.mp3 | golisten

# Anything ffmpeg can open works too
golisten lecture.m4a
golisten talk.mp4
```

### Output Formats

```bash
golisten -f txt  interview.mp3     # plain text (default)
golisten -f srt  interview.mp3     # SubRip subtitles
golisten -f vtt  interview.mp3     # WebVTT
golisten -f json interview.mp3     # segments with millisecond timings

# Timestamped plain text, whisper.cpp style
golisten --timestamps interview.mp3

# Save to a file (add -s to also print it)
golisten -f srt -o subs.srt talk.mp4
```

### Choosing a Model

```bash
# By name, matched against the model directories
golisten -m large-v3-turbo interview.mp3

# Or by path
golisten -m ~/models/ggml-medium.en.bin interview.mp3
```

### Model Discovery

With no `-m`, golisten uses the first directory that contains a model the
selected engine can run, and within it the best one — for transcribe.cpp that
means Parakeet ahead of Canary ahead of Whisper, and `Q8_0` ahead of `Q4_K_M`
within a family:

1. The engine's model variable — `$GOLISTEN_TRANSCRIBE_MODEL`,
   `$GOLISTEN_WHISPER_MODEL` or `$GOLISTEN_PARAKEET_MODEL`
2. `~/.cache/golisten/models` — where `--download` puts things
3. `~/Library/Application Support/MacWhisper/models` (macOS, whisper only)
4. `/opt/homebrew/share/whisper-cpp/models`, `/usr/local/share/whisper-cpp/models`
5. The engine's own checkout: `~/transcribe.cpp/models`, `~/whisper.cpp/models`,
   `~/code/…`, `~/src/…`, `~/.cache/whisper.cpp`
6. `./models`

Earlier directories win, so a model you downloaded with `--download` takes
precedence over one you already had elsewhere. `golisten -V` prints the engine
and model actually chosen:

```
$ golisten -V interview.mp3
Backend: transcribe (auto-selected)
Engine:  /Users/you/.cache/golisten/bin/transcribe-cli
Model:   /Users/you/.cache/golisten/models/parakeet-tdt-0.6b-v3-Q4_K_M.gguf
Audio:   42m18s at 16000 Hz mono
```

whisper.cpp's dummy `for-tests-*` fixtures and its VAD models live in the same
directories as real models and are skipped — auto-selecting one produces
confident nonsense.

Binaries are found the same way: the engine's `_BIN` variable, then
`~/.cache/golisten/bin`, then `PATH`, then the usual build locations. A pre-2024
whisper.cpp checkout that still builds `main` is picked up from a whisper.cpp
directory, but never from `PATH`, where that name means nothing.

### Language and Translation

golisten defaults to `--language auto`, so the language is detected. This
differs from whisper.cpp's own default of `en`, which silently transcribes
every language as though it were English.

```bash
golisten -l de interview.mp3          # tell it the language
golisten --translate interview.mp3    # translate the speech into English
```

### Parakeet

transcribe.cpp is the default and gives Parakeet with segment timestamps, so
SRT and VTT work:

```bash
golisten meeting.wav
golisten -p transcribe -m parakeet-tdt-0.6b-v3-Q8_0.gguf -f srt meeting.wav
```

whisper.cpp also ships a `parakeet-cli`. It is easier to get hold of, but emits
plain text only:

```bash
golisten -p parakeet meeting.wav
```

transcribe.cpp runs Parakeet, Canary, Canary-Qwen, Whisper, GigaAM, Moonshine,
Qwen3-ASR, SenseVoice, Voxtral, and Granite Speech from one binary. Models are
GGUF files from [huggingface.co/handy-computer](https://huggingface.co/handy-computer).

```bash
golisten -p transcribe -m canary-1b-flash.gguf --backend metal talk.wav
```

### Cloud Providers

```bash
# OpenAI
golisten -p openai interview.mp3
golisten -p openai -m gpt-4o-transcribe interview.mp3   # higher accuracy, text only

# ElevenLabs
golisten -p elevenlabs interview.mp3
golisten -p elevenlabs --diarize -f json panel.mp3

# Deepgram
golisten -p deepgram interview.mp3
golisten -p deepgram -m nova-3 --diarize -f srt standup.m4a
```

### Speaker Diarization

Supported by `elevenlabs`, `deepgram`, and `transcribe`:

```bash
golisten -p deepgram --diarize -f srt panel.mp3
```

```
1
00:00:00,000 --> 00:00:04,120
speaker_0: Welcome back to the show.

2
00:00:04,120 --> 00:00:07,500
speaker_1: Thanks for having me.
```

### Improving Accuracy

Bias the model toward names and jargon it would otherwise mangle:

```bash
golisten --prompt "Marcus, Little Bird, Raspberry Pi Pico, I2C, MQTT" standup.mp3
```

### Pointing at a Specific Install

Auto-discovery covers the usual locations. Override it when you need to:

```bash
golisten --bin ~/src/whisper.cpp/build/bin/whisper-cli interview.mp3

export GOLISTEN_WHISPER_BIN=~/src/whisper.cpp/build/bin/whisper-cli
export GOLISTEN_WHISPER_MODEL=~/models/ggml-large-v3.bin
```

## Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--provider` | `-p` | Backend to use | `auto` |
| `--model` | `-m` | Model path or name | Auto-discovered |
| `--language` | `-l` | Spoken language | `auto` |
| `--format` | `-f` | `txt`, `srt`, `vtt`, `json` | `txt` |
| `--output` | `-o` | Write the transcript to a file | stdout |
| `--show` | `-s` | Print the transcript even when saving | `false` |
| `--threads` | `-t` | Threads for local engines | Engine default |
| `--verbose` | `-V` | Stream engine progress to stderr | `false` |
| `--timestamps` | | Include timestamps in txt output | `false` |
| `--translate` | | Translate the speech into English | `false` |
| `--diarize` | | Label speakers where supported | `false` |
| `--prompt` | | Bias spelling and style | - |
| `--bin` | | Path to the local engine binary | Auto-discovered |
| `--backend` | | Compute backend for transcribe.cpp | `auto` |
| `--token` | | API key | From env var |
| `--download` | | Download a model and exit | - |
| `--help` | `-h` | Show help | - |

## Backend Comparison

| | transcribe | whisper | parakeet | openai | elevenlabs | deepgram |
|---|---|---|---|---|---|---|
| Auto-select order | 1st | 2nd | 3rd | - | - | - |
| Runs locally | yes | yes | yes | no | no | no |
| Timestamps | yes | yes | no | `whisper-1` only | yes | yes |
| Diarization | yes | no | no | no | yes | yes |
| Translation | model-dependent | yes | no | yes | no | no |
| Default model | best found (parakeet) | best found | best found | `whisper-1` | `scribe_v1` | `nova-3` |
| Binary env var | `GOLISTEN_TRANSCRIBE_BIN` | `GOLISTEN_WHISPER_BIN` | `GOLISTEN_PARAKEET_BIN` | - | - | - |
| Model env var | `GOLISTEN_TRANSCRIBE_MODEL` | `GOLISTEN_WHISPER_MODEL` | `GOLISTEN_PARAKEET_MODEL` | - | - | - |
| API key env var | - | - | - | `OPENAI_API_KEY` | `ELEVENLABS_API_KEY` | `DEEPGRAM_API_KEY` |

## How the Audio Pipeline Works

Every local engine wants 16 kHz mono audio, so golisten always hands them
exactly that:

1. The container is identified from its magic bytes, never its file extension.
2. MP3 is decoded with `go-mp3`; WAV is parsed in-process (8/16/24/32-bit PCM
   and IEEE float, any channel count). Anything else goes to ffmpeg.
3. Multi-channel audio is downmixed to mono.
4. The result is resampled to 16 kHz with a Blackman-windowed sinc filter whose
   passband stops at 94.5% of Nyquist — the same rolloff soxr and librosa use,
   so audio reaches the acoustic model shaped the way its training data was.

Doing this in-process rather than leaning on the engine means golisten works
with whisper.cpp builds that predate its built-in MP3 support, including
checkouts old enough to still produce a binary called `main`.

Cloud backends get the original file bytes untouched — re-encoding would only
make the upload bigger and the audio worse.

## Long Audio

Recordings past a provider's upload limit (25 MB for OpenAI) are decoded, split
into chunks of at most 10 minutes, transcribed separately, and stitched back
onto a single timeline. Splits land at the quietest point in the second half of
each window, so a cut falls in a pause rather than mid-word — the audio-side
equivalent of breaking long text on sentence boundaries.

Each chunk is retried up to 3 times with exponential backoff (1s, 2s, 4s).
Progress is reported on stderr:

```
Audio is 1h58m12s — splitting into 12 chunks
Transcribing chunk 1/12 (10m00s)...
Transcribing chunk 2/12 (9m58s)...
```

Local engines are handed the whole file — whisper.cpp and transcribe.cpp do
their own internal windowing.

## Scripting Examples

### Transcribe a folder of recordings

```bash
for f in recordings/*.mp3; do
  golisten -f srt -o "${f%.mp3}.srt" "$f"
done
```

### Subtitle a video

```bash
golisten -f srt -o talk.srt talk.mp4
ffmpeg -i talk.mp4 -vf subtitles=talk.srt talk-subbed.mp4
```

### Summarise a meeting

```bash
golisten standup.m4a | llm "Summarise this standup as bullet points"
```

### Round-trip through gospeak

```bash
golisten note.mp3 | llm "Reply to this" | gospeak -v nova
```

### Search a podcast for a phrase

```bash
golisten -f json episode.mp3 | jq -r '.segments[] | select(.text | test("kubernetes"; "i")) | "\(.start_ms/1000)s \(.text)"'
```

## Error Handling

Errors go to stderr and the exit code is non-zero:

```
Error: OPENAI_API_KEY environment variable not set and --token not provided
Error: Invalid provider 'nope'. Use one of: auto, whisper, parakeet, transcribe, openai, elevenlabs, deepgram
Error: the parakeet backend returns text without timestamps.
       Use -p transcribe for timestamped parakeet output, or -p whisper.
Error: no local speech-to-text engine found.
Install one:
  transcribe.cpp (parakeet, the default):  build it, then golisten --download parakeet
  whisper.cpp:                             brew install whisper-cpp, then golisten --download base.en
Or use a hosted backend, e.g. golisten -p openai audio.mp3
```

## Help

```bash
golisten --help
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
