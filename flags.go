package main

import (
	"flag"
	"fmt"
	"io"
)

// parseFlags builds the option set. The third return value is true when the
// caller should exit immediately with the returned code — either because
// parsing failed or because --help was handled.
func parseFlags(args []string, stderr io.Writer) (options, int, bool) {
	var opts options

	fs := flag.NewFlagSet("golisten", flag.ContinueOnError)
	fs.SetOutput(stderr)

	fs.StringVar(&opts.provider, "provider", defaultProvider, "Transcription backend")
	fs.StringVar(&opts.provider, "p", defaultProvider, "Transcription backend (shorthand)")
	fs.StringVar(&opts.model, "model", "", "Model path or name")
	fs.StringVar(&opts.model, "m", "", "Model path or name (shorthand)")
	fs.StringVar(&opts.language, "language", defaultLanguage, "Spoken language, or auto")
	fs.StringVar(&opts.language, "l", defaultLanguage, "Spoken language (shorthand)")
	fs.StringVar(&opts.format, "format", defaultFormat, "Output format: txt, srt, vtt, json")
	fs.StringVar(&opts.format, "f", defaultFormat, "Output format (shorthand)")
	fs.StringVar(&opts.output, "output", "", "Write the transcript to this file")
	fs.StringVar(&opts.output, "o", "", "Write the transcript to this file (shorthand)")
	fs.IntVar(&opts.threads, "threads", 0, "Threads for local engines")
	fs.IntVar(&opts.threads, "t", 0, "Threads for local engines (shorthand)")
	fs.BoolVar(&opts.show, "show", false, "Print the transcript even when saving to a file")
	fs.BoolVar(&opts.show, "s", false, "Print the transcript even when saving to a file (shorthand)")
	fs.BoolVar(&opts.verbose, "verbose", false, "Stream engine progress to stderr")
	fs.BoolVar(&opts.verbose, "V", false, "Stream engine progress to stderr (shorthand)")
	fs.StringVar(&opts.binary, "bin", "", "Path to the local engine binary")
	fs.StringVar(&opts.token, "token", "", "API key for the provider")
	fs.StringVar(&opts.prompt, "prompt", "", "Initial prompt to bias spelling and style")
	fs.StringVar(&opts.backend, "backend", "", "Compute backend for transcribe.cpp")
	fs.BoolVar(&opts.translate, "translate", false, "Translate the speech into English")
	fs.BoolVar(&opts.diarize, "diarize", false, "Label speakers where supported")
	fs.BoolVar(&opts.timestamps, "timestamps", false, "Include timestamps in txt output")
	fs.StringVar(&opts.download, "download", "", "Download a whisper model and exit")
	fs.BoolVar(&opts.help, "help", false, "Show help")
	fs.BoolVar(&opts.help, "h", false, "Show help (shorthand)")

	fs.Usage = func() { printUsage(stderr) }

	if err := fs.Parse(args); err != nil {
		return opts, 2, true
	}
	if opts.help {
		fs.Usage()
		return opts, 0, true
	}
	opts.args = fs.Args()
	return opts, 0, false
}

func printUsage(w io.Writer) {
	fmt.Fprintf(w, "golisten - Speech-to-text using local whisper.cpp / transcribe.cpp, or the OpenAI,\n")
	fmt.Fprintf(w, "           ElevenLabs, and Deepgram APIs\n\n")
	fmt.Fprintf(w, "Usage: golisten [options] audio.mp3\n")
	fmt.Fprintf(w, "       cat audio.mp3 | golisten [options]\n\n")
	fmt.Fprintf(w, "Options:\n")
	fmt.Fprintf(w, "  -p, --provider    Backend: auto, transcribe, whisper, parakeet, openai,\n")
	fmt.Fprintf(w, "                    elevenlabs, deepgram (default: auto)\n")
	fmt.Fprintf(w, "  -m, --model       Model path, or a name to match in the model dirs\n")
	fmt.Fprintf(w, "  -l, --language    Spoken language, e.g. en, de (default: auto)\n")
	fmt.Fprintf(w, "  -f, --format      Output format: txt, srt, vtt, json (default: txt)\n")
	fmt.Fprintf(w, "  -o, --output      Write the transcript to this file\n")
	fmt.Fprintf(w, "  -s, --show        Print the transcript even when saving to a file\n")
	fmt.Fprintf(w, "  -t, --threads     Threads for local engines\n")
	fmt.Fprintf(w, "  -V, --verbose     Stream engine progress to stderr\n")
	fmt.Fprintf(w, "      --timestamps  Include timestamps in txt output\n")
	fmt.Fprintf(w, "      --translate   Translate the speech into English\n")
	fmt.Fprintf(w, "      --diarize     Label speakers (elevenlabs, deepgram, transcribe)\n")
	fmt.Fprintf(w, "      --prompt      Initial prompt to bias spelling and style\n")
	fmt.Fprintf(w, "      --bin         Path to the local engine binary\n")
	fmt.Fprintf(w, "      --backend     Compute backend for transcribe.cpp (metal, cuda, ...)\n")
	fmt.Fprintf(w, "      --token       API key (or set the provider's env var)\n")
	fmt.Fprintf(w, "      --download    Download a model and exit (see below)\n")
	fmt.Fprintf(w, "  -h, --help        Show this help message\n\n")

	fmt.Fprintf(w, "Local backends (auto picks the first one installed):\n")
	fmt.Fprintf(w, "  transcribe  transcribe.cpp — parakeet by default, also canary,\n")
	fmt.Fprintf(w, "              whisper, moonshine, voxtral, granite, ...\n")
	fmt.Fprintf(w, "              Models:  *.gguf from huggingface.co/handy-computer\n")
	fmt.Fprintf(w, "              Env: GOLISTEN_TRANSCRIBE_BIN/_MODEL Timestamps: yes\n")
	fmt.Fprintf(w, "  whisper     whisper.cpp (whisper-cli, or a pre-2024 `main`)\n")
	fmt.Fprintf(w, "              Models:  ggml-*.bin   Env: GOLISTEN_WHISPER_BIN/_MODEL\n")
	fmt.Fprintf(w, "              Timestamps: yes. Used first for --translate, which\n")
	fmt.Fprintf(w, "              parakeet cannot do.\n")
	fmt.Fprintf(w, "  parakeet    whisper.cpp's parakeet-cli\n")
	fmt.Fprintf(w, "              Models:  ggml-parakeet-*.bin\n")
	fmt.Fprintf(w, "              Env: GOLISTEN_PARAKEET_BIN/_MODEL   Timestamps: no\n\n")

	fmt.Fprintf(w, "Cloud backends:\n")
	fmt.Fprintf(w, "  openai      Env: OPENAI_API_KEY\n")
	fmt.Fprintf(w, "              Models: whisper-1 (default, timestamped),\n")
	fmt.Fprintf(w, "                      gpt-4o-transcribe, gpt-4o-mini-transcribe (text only)\n")
	fmt.Fprintf(w, "  elevenlabs  Env: ELEVENLABS_API_KEY\n")
	fmt.Fprintf(w, "              Models: scribe_v1 (default), scribe_v2\n")
	fmt.Fprintf(w, "  deepgram    Env: DEEPGRAM_API_KEY\n")
	fmt.Fprintf(w, "              Models: nova-3 (default), nova-2, enhanced, base\n\n")

	fmt.Fprintf(w, "Models (--download saves to ~/.cache/golisten/models):\n")
	fmt.Fprintf(w, "  transcribe.cpp:  parakeet (default), parakeet-q8, parakeet-f16,\n")
	fmt.Fprintf(w, "                   parakeet-v2, parakeet-en, parakeet-ctc, parakeet-rnnt\n")
	fmt.Fprintf(w, "  whisper.cpp:     tiny.en, base.en, small.en, medium.en, large-v3-turbo\n\n")

	fmt.Fprintf(w, "Examples:\n")
	fmt.Fprintf(w, "  golisten interview.mp3\n")
	fmt.Fprintf(w, "  golisten --download parakeet\n")
	fmt.Fprintf(w, "  golisten -m large-v3-turbo -f srt -o subs.srt talk.mp4\n")
	fmt.Fprintf(w, "  golisten -p transcribe -m parakeet-tdt-0.6b-v2-Q4_K_M.gguf meeting.wav\n")
	fmt.Fprintf(w, "  golisten -p deepgram --diarize -f json standup.m4a\n")
	fmt.Fprintf(w, "  cat note.mp3 | golisten -p openai | gospeak -v nova\n")
}
