package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// Cloud provider defaults.
	defaultOpenAIModel     = "whisper-1"
	defaultElevenLabsModel = "scribe_v1"
	defaultDeepgramModel   = "nova-3"

	defaultProvider = "whisper"
	defaultFormat   = "txt"
	defaultLanguage = "auto"

	// Longest slice of audio sent in a single cloud request. At 16 kHz mono
	// 16-bit that is ~19 MB, comfortably inside OpenAI's 25 MB limit while
	// leaving room for the multipart envelope.
	maxChunkDuration = 10 * time.Minute

	// Number of attempts per chunk before giving up.
	maxRetries = 3
)

// Initial backoff between retries (doubled each attempt). Declared as a var
// rather than a const so tests can swap it in for fast-running retry tests.
var initialBackoff = 1 * time.Second

// providers lists every backend, local first.
var providers = []string{"whisper", "parakeet", "transcribe", "openai", "elevenlabs", "deepgram"}

// localProviders are the backends that shell out to a locally-installed engine
// rather than a hosted API.
var localProviders = map[string]bool{
	"whisper":    true,
	"parakeet":   true,
	"transcribe": true,
}

// providerUploadLimit is the largest single request each cloud provider
// accepts. Anything bigger is split into chunks and stitched back together.
var providerUploadLimit = map[string]int{
	"openai":     25 * 1024 * 1024,
	"elevenlabs": 1 << 30,
	"deepgram":   1 << 30,
}

// apiKeyEnv maps each cloud provider to the environment variable holding its
// key — the same names gospeak uses, so one export serves both tools.
var apiKeyEnv = map[string]string{
	"openai":     "OPENAI_API_KEY",
	"elevenlabs": "ELEVENLABS_API_KEY",
	"deepgram":   "DEEPGRAM_API_KEY",
}

func main() {
	var stdin io.Reader = strings.NewReader("")
	if stat, err := os.Stdin.Stat(); err == nil && (stat.Mode()&os.ModeCharDevice) == 0 {
		stdin = os.Stdin
	}
	os.Exit(run(os.Args[1:], stdin, os.Stdout, os.Stderr, os.Getenv))
}

// options is the parsed command line.
type options struct {
	provider   string
	model      string
	language   string
	format     string
	output     string
	binary     string
	token      string
	prompt     string
	backend    string
	threads    int
	translate  bool
	diarize    bool
	timestamps bool
	show       bool
	verbose    bool
	help       bool
	download   string
	args       []string
}

// run is the testable entry point. It returns an exit code instead of calling
// os.Exit so tests can drive the CLI without killing the process. Stdin,
// stdout, stderr, and the env lookup are all injected for the same reason.
func run(args []string, stdin io.Reader, stdout, stderr io.Writer, getenv func(string) string) int {
	opts, code, done := parseFlags(args, stderr)
	if done {
		return code
	}

	if opts.download != "" {
		if err := downloadModel(opts.download, stderr); err != nil {
			fmt.Fprintf(stderr, "Error: %v\n", err)
			return 1
		}
		return 0
	}

	opts.provider = strings.ToLower(opts.provider)
	if !containsString(providers, opts.provider) {
		fmt.Fprintf(stderr, "Error: Invalid provider '%s'. Use one of: %s\n",
			opts.provider, strings.Join(providers, ", "))
		return 1
	}

	opts.format = strings.ToLower(opts.format)
	if !containsString([]string{"txt", "srt", "vtt", "json"}, opts.format) {
		fmt.Fprintf(stderr, "Error: Invalid format '%s'. Use txt, srt, vtt, or json\n", opts.format)
		return 1
	}

	needSegments := formatNeedsSegments(opts.format, opts.timestamps)
	if needSegments && opts.provider == "parakeet" {
		fmt.Fprintln(stderr, "Error: the parakeet backend returns text without timestamps.")
		fmt.Fprintln(stderr, "       Use -p transcribe for timestamped parakeet output, or -p whisper.")
		return 1
	}
	if opts.diarize && !containsString([]string{"elevenlabs", "deepgram", "transcribe"}, opts.provider) {
		fmt.Fprintf(stderr, "Warning: --diarize is not supported by the %s backend, ignoring\n", opts.provider)
		opts.diarize = false
	}

	audioBytes, source, err := readInput(opts.args, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "Error: %v\n", err)
		return 1
	}

	transcript, err := transcribe(opts, audioBytes, needSegments, stderr, getenv)
	if err != nil {
		fmt.Fprintf(stderr, "Error transcribing %s: %v\n", source, err)
		return 1
	}

	rendered, err := render(transcript, opts.format, opts.timestamps)
	if err != nil {
		fmt.Fprintf(stderr, "Error: %v\n", err)
		return 1
	}

	if opts.output != "" {
		if err := os.WriteFile(opts.output, []byte(rendered), 0644); err != nil {
			fmt.Fprintf(stderr, "Error saving file: %v\n", err)
			return 1
		}
		fmt.Fprintf(stderr, "Saved to %s\n", opts.output)
	}
	if opts.output == "" || opts.show {
		fmt.Fprint(stdout, rendered)
	}
	return 0
}

// transcribe dispatches to the selected backend, converting and chunking the
// audio only as much as that backend requires.
func transcribe(opts options, audioBytes []byte, needSegments bool, stderr io.Writer, getenv func(string) string) (Transcript, error) {
	if localProviders[opts.provider] {
		return transcribeLocally(opts, audioBytes, stderr, getenv)
	}
	return transcribeInCloud(opts, audioBytes, needSegments, stderr, getenv)
}

// transcribeLocally normalises the audio to 16 kHz mono WAV and hands it to the
// installed engine. Every supported engine reads that format, including
// whisper.cpp builds old enough to predate its built-in MP3 decoding.
func transcribeLocally(opts options, audioBytes []byte, stderr io.Writer, getenv func(string) string) (Transcript, error) {
	locations := map[string]engineLocations{
		"whisper":    whisperLocations,
		"parakeet":   parakeetLocations,
		"transcribe": transcribeLocations,
	}[opts.provider]

	binary, err := findBinary(locations, opts.binary, getenv)
	if err != nil {
		return Transcript{}, err
	}
	model, err := findModel(locations, opts.model, getenv)
	if err != nil {
		return Transcript{}, err
	}

	audio, err := Decode(audioBytes)
	if err != nil {
		return Transcript{}, err
	}
	if opts.verbose {
		fmt.Fprintf(stderr, "Engine: %s\nModel:  %s\nAudio:  %s at %d Hz mono\n",
			binary, model, formatDuration(audio.Duration()), audio.Rate)
	}

	req := localRequest{
		WAV:       EncodeWAV(audio),
		Binary:    binary,
		Model:     model,
		Threads:   opts.threads,
		Translate: opts.translate,
		Prompt:    opts.prompt,
		Verbose:   opts.verbose,
		Stderr:    stderr,
	}

	var transcript Transcript
	switch opts.provider {
	case "whisper":
		// whisper.cpp assumes English when no language is given, which
		// silently mistranscribes everything else; "auto" is passed through so
		// it detects instead.
		req.Language = opts.language
		transcript, err = transcribeWhisperCpp(req)
	case "parakeet":
		transcript, err = transcribeParakeetCpp(req)
	case "transcribe":
		if opts.language != defaultLanguage {
			req.Language = opts.language
		}
		transcript, err = transcribeTranscribeCpp(req, transcribeCppOptions{
			Diarize: opts.diarize,
			Backend: opts.backend,
		})
	}
	if err != nil {
		return Transcript{}, err
	}
	if transcript.Duration == 0 {
		transcript.Duration = audio.Duration()
	}
	return transcript, nil
}

// transcribeInCloud uploads to a hosted API. Files within the provider's limit
// are sent byte-for-byte as supplied — re-encoding an MP3 would only make the
// upload larger and the audio worse. Anything over the limit is decoded, split
// on quiet points, and stitched back onto one timeline.
func transcribeInCloud(opts options, audioBytes []byte, needSegments bool, stderr io.Writer, getenv func(string) string) (Transcript, error) {
	apiKey := opts.token
	if apiKey == "" {
		apiKey = getenv(apiKeyEnv[opts.provider])
	}
	if apiKey == "" {
		return Transcript{}, fmt.Errorf("%s environment variable not set and --token not provided",
			apiKeyEnv[opts.provider])
	}

	model := opts.model
	if model == "" {
		model = map[string]string{
			"openai":     defaultOpenAIModel,
			"elevenlabs": defaultElevenLabsModel,
			"deepgram":   defaultDeepgramModel,
		}[opts.provider]
	}

	language := opts.language
	if language == defaultLanguage {
		language = "" // every cloud provider auto-detects when unset
	}

	base := cloudRequest{
		APIKey:       apiKey,
		Model:        model,
		Language:     language,
		Prompt:       opts.prompt,
		Translate:    opts.translate,
		Diarize:      opts.diarize,
		NeedSegments: needSegments,
	}

	limit := providerUploadLimit[opts.provider]
	if len(audioBytes) <= limit {
		format := detectFormat(audioBytes)
		req := base
		req.Audio = audioBytes
		req.MIME = mimeForFormat(format)
		req.Ext = fileExtForFormat(format)
		return withRetry("transcribe", stderr, func() (Transcript, error) {
			return callCloud(opts.provider, req)
		})
	}

	audio, err := Decode(audioBytes)
	if err != nil {
		return Transcript{}, err
	}
	chunks := ChunkAudio(audio, int(maxChunkDuration.Seconds())*audio.Rate)
	fmt.Fprintf(stderr, "Audio is %s — splitting into %d chunks\n",
		formatDuration(audio.Duration()), len(chunks))

	parts := make([]Transcript, 0, len(chunks))
	for i, chunk := range chunks {
		fmt.Fprintf(stderr, "Transcribing chunk %d/%d (%s)...\n",
			i+1, len(chunks), formatDuration(chunk.Audio.Duration()))

		req := base
		req.Audio = EncodeWAV(chunk.Audio)
		req.MIME = "audio/wav"
		req.Ext = "wav"

		label := fmt.Sprintf("chunk %d/%d", i+1, len(chunks))
		part, err := withRetry(label, stderr, func() (Transcript, error) {
			return callCloud(opts.provider, req)
		})
		if err != nil {
			return Transcript{}, err
		}
		parts = append(parts, part.shift(chunk.Offset))
	}

	merged := mergeTranscripts(parts)
	merged.Duration = audio.Duration()
	return merged, nil
}

func callCloud(provider string, req cloudRequest) (Transcript, error) {
	switch provider {
	case "openai":
		return transcribeOpenAI(req)
	case "elevenlabs":
		return transcribeElevenLabs(req)
	case "deepgram":
		return transcribeDeepgram(req)
	}
	return Transcript{}, fmt.Errorf("unknown provider %q", provider)
}

// withRetry retries fn up to maxRetries times with exponential backoff. The
// label is used for log messages so the user can see which chunk is retrying.
func withRetry(label string, stderr io.Writer, fn func() (Transcript, error)) (Transcript, error) {
	var lastErr error
	backoff := initialBackoff
	for attempt := 1; attempt <= maxRetries; attempt++ {
		result, err := fn()
		if err == nil {
			return result, nil
		}
		lastErr = err
		if attempt < maxRetries {
			fmt.Fprintf(stderr, "%s: attempt %d/%d failed: %v (retrying in %v)\n",
				label, attempt, maxRetries, err, backoff)
			time.Sleep(backoff)
			backoff *= 2
		}
	}
	return Transcript{}, fmt.Errorf("%s failed after %d attempts: %w", label, maxRetries, lastErr)
}

// readInput loads the audio from the named file, or from stdin when no path was
// given. Returns the bytes and a label for error messages.
func readInput(args []string, stdin io.Reader) ([]byte, string, error) {
	if len(args) > 1 {
		return nil, "", fmt.Errorf("expected a single audio file, got %d", len(args))
	}
	if len(args) == 1 && args[0] != "-" {
		data, err := os.ReadFile(args[0])
		if err != nil {
			return nil, "", fmt.Errorf("failed to read %s: %w", args[0], err)
		}
		if len(data) == 0 {
			return nil, "", fmt.Errorf("%s is empty", args[0])
		}
		return data, args[0], nil
	}

	data, err := io.ReadAll(stdin)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read stdin: %w", err)
	}
	if len(data) == 0 {
		return nil, "", fmt.Errorf("no audio provided (pass a file, or pipe one in)")
	}
	return data, "stdin", nil
}

// modelDownloadBase is the model mirror whisper.cpp's own download script uses,
// as a var so tests can point it at a local server.
var modelDownloadBase = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

// downloadModel fetches a ggml whisper model into the golisten model cache,
// which is first on the auto-discovery path.
func downloadModel(name string, stderr io.Writer) error {
	name = strings.TrimPrefix(strings.TrimSuffix(name, ".bin"), "ggml-")
	if name == "" {
		return fmt.Errorf("no model name given")
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("cannot locate home directory: %w", err)
	}
	dir := filepath.Join(home, ".cache", "golisten", "models")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("cannot create %s: %w", dir, err)
	}

	filename := "ggml-" + name + ".bin"
	dest := filepath.Join(dir, filename)
	if _, err := os.Stat(dest); err == nil {
		fmt.Fprintf(stderr, "Already downloaded: %s\n", dest)
		return nil
	}

	url := modelDownloadBase + "/" + filename
	fmt.Fprintf(stderr, "Downloading %s...\n", url)

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed (%d): no model named %q "+
			"(try tiny.en, base.en, small.en, medium.en, large-v3-turbo)", resp.StatusCode, name)
	}

	// Download to a temporary name so an interrupted transfer can never be
	// mistaken for a complete model by the discovery scan.
	tmp := dest + ".part"
	file, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("cannot write %s: %w", tmp, err)
	}
	written, err := io.Copy(file, resp.Body)
	closeErr := file.Close()
	if err != nil {
		os.Remove(tmp)
		return fmt.Errorf("download failed: %w", err)
	}
	if closeErr != nil {
		os.Remove(tmp)
		return fmt.Errorf("cannot write %s: %w", tmp, closeErr)
	}
	if err := os.Rename(tmp, dest); err != nil {
		return fmt.Errorf("cannot finalise %s: %w", dest, err)
	}

	fmt.Fprintf(stderr, "Saved %s (%.1f MB)\n", dest, float64(written)/(1024*1024))
	return nil
}

// formatDuration renders a length as 1h02m03s / 2m03s / 3.4s, whichever is
// shortest without losing useful precision.
func formatDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	total := int(d.Seconds())
	switch {
	case total >= 3600:
		return fmt.Sprintf("%dh%02dm%02ds", total/3600, (total%3600)/60, total%60)
	case total >= 60:
		return fmt.Sprintf("%dm%02ds", total/60, total%60)
	default:
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
}

func containsString(haystack []string, needle string) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}
