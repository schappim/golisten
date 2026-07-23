package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func emptyEnv(string) string { return "" }

func envMap(m map[string]string) func(string) string {
	return func(key string) string { return m[key] }
}

// runCLI drives run() and returns the exit code plus stdout and stderr.
func runCLI(t *testing.T, args []string, stdin string, getenv func(string) string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := run(args, strings.NewReader(stdin), &stdout, &stderr, getenv)
	return code, stdout.String(), stderr.String()
}

// testWAV builds a short valid 16 kHz mono WAV for CLI tests.
func testWAV() []byte {
	return EncodeWAV(Audio{Samples: make([]float32, 16000), Rate: 16000})
}

func TestMain(m *testing.M) {
	// Keep retry tests fast; the production backoff is a full second.
	initialBackoff = time.Millisecond
	os.Exit(m.Run())
}

func TestRun_HelpReturnsZeroAndPrintsUsage(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"--help"}, "", emptyEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	for _, want := range []string{"golisten", "--provider", "whisper", "transcribe", "deepgram"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("usage is missing %q", want)
		}
	}
}

func TestRun_InvalidFlagReturnsTwo(t *testing.T) {
	code, _, _ := runCLI(t, []string{"--not-a-flag"}, "", emptyEnv)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
}

func TestRun_InvalidProviderReturnsOne(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"-p", "nope", "file.mp3"}, "", emptyEnv)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "Invalid provider") {
		t.Fatalf("stderr = %q", stderr)
	}
}

func TestRun_InvalidFormatReturnsOne(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"-f", "docx", "file.mp3"}, "", emptyEnv)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "Invalid format") {
		t.Fatalf("stderr = %q", stderr)
	}
}

// parakeet-cli has no timing output, so asking it for subtitles must fail with
// a pointer to a backend that can, rather than emit an empty file.
func TestRun_ParakeetWithTimedFormatIsRejected(t *testing.T) {
	for _, args := range [][]string{
		{"-p", "parakeet", "-f", "srt", "file.mp3"},
		{"-p", "parakeet", "-f", "vtt", "file.mp3"},
		{"-p", "parakeet", "--timestamps", "file.mp3"},
	} {
		code, _, stderr := runCLI(t, args, "", emptyEnv)
		if code != 1 {
			t.Fatalf("%v: exit code = %d, want 1", args, code)
		}
		if !strings.Contains(stderr, "transcribe") {
			t.Errorf("%v: the error should point at a timestamped backend, got %q", args, stderr)
		}
	}
}

func TestRun_MissingAPIKeyPerProvider(t *testing.T) {
	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}

	for provider, envVar := range apiKeyEnv {
		t.Run(provider, func(t *testing.T) {
			code, _, stderr := runCLI(t, []string{"-p", provider, audio}, "", emptyEnv)
			if code != 1 {
				t.Fatalf("exit code = %d, want 1", code)
			}
			if !strings.Contains(stderr, envVar) {
				t.Fatalf("the error should name %s, got %q", envVar, stderr)
			}
		})
	}
}

func TestRun_NoInputIsAnError(t *testing.T) {
	code, _, stderr := runCLI(t, []string{}, "", emptyEnv)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "no audio") {
		t.Fatalf("stderr = %q", stderr)
	}
}

func TestRun_TooManyFilesIsAnError(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"a.mp3", "b.mp3"}, "", emptyEnv)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "single audio file") {
		t.Fatalf("stderr = %q", stderr)
	}
}

func TestRun_MissingFileIsAnError(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"/no/such/file.mp3"}, "", emptyEnv)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "failed to read") {
		t.Fatalf("stderr = %q", stderr)
	}
}

func TestRun_EmptyFileIsAnError(t *testing.T) {
	dir := t.TempDir()
	empty := filepath.Join(dir, "empty.mp3")
	if err := os.WriteFile(empty, nil, 0644); err != nil {
		t.Fatal(err)
	}
	code, _, stderr := runCLI(t, []string{empty}, "", emptyEnv)
	if code != 1 || !strings.Contains(stderr, "empty") {
		t.Fatalf("code = %d, stderr = %q", code, stderr)
	}
}

// A --diarize that the backend cannot honour should warn and continue, not
// abort a long transcription run.
func TestRun_UnsupportedDiarizeWarnsAndContinues(t *testing.T) {
	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}
	bin := whisperFixtureBinary(t, dir)

	code, stdout, stderr := runCLI(t, []string{"-p", "whisper", "--diarize", audio}, "",
		envMap(map[string]string{"GOLISTEN_WHISPER_BIN": bin, "GOLISTEN_WHISPER_MODEL": modelFixture(t, dir)}))
	if code != 0 {
		t.Fatalf("exit code = %d, want 0. stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "not supported") {
		t.Errorf("expected a warning, got %q", stderr)
	}
	if !strings.Contains(stdout, "fellow Americans") {
		t.Errorf("stdout = %q", stdout)
	}
}

// whisperFixtureBinary writes a stand-in whisper.cpp that emits the JSON
// sidecar the real binary would.
func whisperFixtureBinary(t *testing.T, dir string) string {
	t.Helper()
	return writeScript(t, dir, "whisper-cli", `
OUT=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then shift; OUT="$1"; fi
  shift
done
cat > "$OUT.json" <<'JSON'
`+whisperJSONFixture+`
JSON
`)
}

func modelFixture(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "ggml-tiny.en.bin")
	if err := os.WriteFile(path, []byte("model"), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRun_LocalWhisperEndToEnd(t *testing.T) {
	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}
	env := envMap(map[string]string{
		"GOLISTEN_WHISPER_BIN":   whisperFixtureBinary(t, dir),
		"GOLISTEN_WHISPER_MODEL": modelFixture(t, dir),
	})

	t.Run("txt to stdout", func(t *testing.T) {
		code, stdout, stderr := runCLI(t, []string{"-p", "whisper", audio}, "", env)
		if code != 0 {
			t.Fatalf("exit code = %d: %s", code, stderr)
		}
		want := "And so my fellow Americans ask what you can do for your country.\n"
		if stdout != want {
			t.Fatalf("stdout = %q, want %q", stdout, want)
		}
	})

	t.Run("srt", func(t *testing.T) {
		code, stdout, _ := runCLI(t, []string{"-p", "whisper", "-f", "srt", audio}, "", env)
		if code != 0 {
			t.Fatalf("exit code = %d", code)
		}
		if !strings.HasPrefix(stdout, "1\n00:00:00,000 --> 00:00:07,960\n") {
			t.Fatalf("stdout = %q", stdout)
		}
	})

	t.Run("audio piped on stdin", func(t *testing.T) {
		code, stdout, stderr := runCLI(t, []string{"-p", "whisper"}, string(testWAV()), env)
		if code != 0 {
			t.Fatalf("exit code = %d: %s", code, stderr)
		}
		if !strings.Contains(stdout, "fellow Americans") {
			t.Fatalf("stdout = %q", stdout)
		}
	})

	t.Run("explicit - reads stdin", func(t *testing.T) {
		code, stdout, _ := runCLI(t, []string{"-p", "whisper", "-"}, string(testWAV()), env)
		if code != 0 {
			t.Fatalf("exit code = %d", code)
		}
		if !strings.Contains(stdout, "fellow Americans") {
			t.Fatalf("stdout = %q", stdout)
		}
	})

	t.Run("output file", func(t *testing.T) {
		out := filepath.Join(dir, "out.txt")
		code, stdout, stderr := runCLI(t, []string{"-p", "whisper", "-o", out, audio}, "", env)
		if code != 0 {
			t.Fatalf("exit code = %d", code)
		}
		if stdout != "" {
			t.Errorf("stdout should be empty when saving to a file, got %q", stdout)
		}
		if !strings.Contains(stderr, "Saved to") {
			t.Errorf("stderr = %q", stderr)
		}
		data, err := os.ReadFile(out)
		if err != nil {
			t.Fatalf("failed to read the output file: %v", err)
		}
		if !strings.Contains(string(data), "fellow Americans") {
			t.Fatalf("file contents = %q", data)
		}
	})

	t.Run("--show also prints when saving", func(t *testing.T) {
		out := filepath.Join(dir, "out2.txt")
		code, stdout, _ := runCLI(t, []string{"-p", "whisper", "-o", out, "-s", audio}, "", env)
		if code != 0 {
			t.Fatalf("exit code = %d", code)
		}
		if !strings.Contains(stdout, "fellow Americans") {
			t.Fatalf("stdout = %q", stdout)
		}
	})

	t.Run("unwritable output path is an error", func(t *testing.T) {
		code, _, stderr := runCLI(t, []string{"-p", "whisper", "-o", "/no/such/dir/out.txt", audio}, "", env)
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
		if !strings.Contains(stderr, "Error saving file") {
			t.Fatalf("stderr = %q", stderr)
		}
	})
}

func TestRun_LocalEngineFailureIsReported(t *testing.T) {
	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}
	bin := writeScript(t, dir, "whisper-cli", "echo 'error: bad model file' >&2\nexit 1\n")

	code, _, stderr := runCLI(t, []string{"-p", "whisper", audio}, "",
		envMap(map[string]string{"GOLISTEN_WHISPER_BIN": bin, "GOLISTEN_WHISPER_MODEL": modelFixture(t, dir)}))
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "bad model file") {
		t.Fatalf("the engine's message should reach the user, got %q", stderr)
	}
}

// The whole point of converting in-process is that the engine receives 16 kHz
// mono WAV no matter what was handed to golisten.
func TestRun_ConvertsInputTo16kMonoWAVForLocalEngines(t *testing.T) {
	dir := t.TempDir()

	// 44.1 kHz stereo input.
	payload := make([]byte, 0, 44100*4)
	for i := 0; i < 44100; i++ {
		payload = append(payload, int16Bytes(1000, -1000)...)
	}
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, buildWAV(wavFormatPCM, 2, 44100, 16, payload), 0644); err != nil {
		t.Fatal(err)
	}

	copyPath := filepath.Join(dir, "seen.wav")
	bin := writeScript(t, dir, "whisper-cli", `
OUT=""
IN=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then shift; OUT="$1"; fi
  if [ "$1" = "-f" ]; then shift; IN="$1"; fi
  shift
done
cp "$IN" `+copyPath+`
echo '{"result":{"language":"en"},"transcription":[]}' > "$OUT.json"
`)

	code, _, stderr := runCLI(t, []string{"-p", "whisper", audio}, "",
		envMap(map[string]string{"GOLISTEN_WHISPER_BIN": bin, "GOLISTEN_WHISPER_MODEL": modelFixture(t, dir)}))
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, stderr)
	}

	seen, err := os.ReadFile(copyPath)
	if err != nil {
		t.Fatalf("the engine did not receive a file: %v", err)
	}
	if rate := binary.LittleEndian.Uint32(seen[24:]); rate != 16000 {
		t.Errorf("engine received %d Hz audio, want 16000", rate)
	}
	if ch := binary.LittleEndian.Uint16(seen[22:]); ch != 1 {
		t.Errorf("engine received %d channels, want mono", ch)
	}
	if bits := binary.LittleEndian.Uint16(seen[34:]); bits != 16 {
		t.Errorf("engine received %d-bit audio, want 16", bits)
	}
}

func TestRun_CloudProviderEndToEnd(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, openAIVerboseFixture)
	defer swapURL(&openAIAPIURL, srv.URL)()

	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := runCLI(t, []string{"-p", "openai", audio}, "",
		envMap(map[string]string{"OPENAI_API_KEY": "sk-from-env"}))
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, stderr)
	}
	if !strings.Contains(stdout, "fellow Americans") {
		t.Fatalf("stdout = %q", stdout)
	}
	if captured.headers.Get("Authorization") != "Bearer sk-from-env" {
		t.Errorf("the API key from the environment was not used: %q",
			captured.headers.Get("Authorization"))
	}
	// A file under the size limit is uploaded byte-for-byte, not re-encoded.
	if !bytes.Equal(captured.file, testWAV()) {
		t.Error("the original bytes should be uploaded unchanged")
	}
}

func TestRun_TokenFlagOverridesEnv(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, `{"text":"ok"}`)
	defer swapURL(&openAIAPIURL, srv.URL)()

	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}

	code, _, _ := runCLI(t, []string{"-p", "openai", "--token", "sk-explicit", audio}, "",
		envMap(map[string]string{"OPENAI_API_KEY": "sk-from-env"}))
	if code != 0 {
		t.Fatalf("exit code = %d", code)
	}
	if captured.headers.Get("Authorization") != "Bearer sk-explicit" {
		t.Errorf("--token should win, got %q", captured.headers.Get("Authorization"))
	}
}

// "auto" is golisten's default and means "let the model detect it" — it must
// not be forwarded to a cloud API as if it were a real language code.
func TestRun_AutoLanguageIsNotSentToCloudProviders(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, `{"text":"ok"}`)
	defer swapURL(&openAIAPIURL, srv.URL)()

	dir := t.TempDir()
	audio := filepath.Join(dir, "a.wav")
	if err := os.WriteFile(audio, testWAV(), 0644); err != nil {
		t.Fatal(err)
	}

	if code, _, _ := runCLI(t, []string{"-p", "openai", audio}, "",
		envMap(map[string]string{"OPENAI_API_KEY": "k"})); code != 0 {
		t.Fatalf("exit code = %d", code)
	}
	if _, ok := captured.fields["language"]; ok {
		t.Errorf("language should be omitted for auto-detection, got %q", captured.fields["language"])
	}
}

// Recordings past the provider's size limit are split, transcribed
// individually, and stitched back onto one timeline.
func TestRun_ChunksOversizedCloudUploads(t *testing.T) {
	var uploads int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		uploads++
		io.WriteString(w, `{"text":"part","language":"en","duration":600,
			"segments":[{"start":0.0,"end":5.0,"text":"part"}]}`)
	}))
	defer srv.Close()
	defer swapURL(&openAIAPIURL, srv.URL)()

	// 25 minutes of audio: over OpenAI's 25 MB limit, so it must be chunked.
	original := providerUploadLimit["openai"]
	providerUploadLimit["openai"] = 1024
	defer func() { providerUploadLimit["openai"] = original }()

	dir := t.TempDir()
	audio := filepath.Join(dir, "long.wav")
	if err := os.WriteFile(audio, EncodeWAV(Audio{Samples: make([]float32, 16000*25*60), Rate: 16000}), 0644); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := runCLI(t, []string{"-p", "openai", "-f", "srt", audio}, "",
		envMap(map[string]string{"OPENAI_API_KEY": "k"}))
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, stderr)
	}
	if uploads < 3 {
		t.Fatalf("expected at least 3 chunk uploads, got %d", uploads)
	}
	if !strings.Contains(stderr, "splitting into") {
		t.Errorf("expected chunk progress on stderr, got %q", stderr)
	}
	// Later chunks must be offset onto the global timeline, not all start at 0.
	if strings.Count(stdout, "00:00:00,000 --> ") > 1 {
		t.Errorf("chunk segments were not shifted onto one timeline:\n%s", stdout)
	}
	// Chunks should use close to the full 10-minute budget rather than
	// splitting early, so the second chunk starts near the 10-minute mark.
	if !strings.Contains(stdout, "00:09:5") && !strings.Contains(stdout, "00:10:0") {
		t.Errorf("expected the second chunk to start near 10 minutes:\n%s", stdout)
	}
	if uploads > 4 {
		t.Errorf("25 minutes split into %d chunks; the splitter is not using its budget", uploads)
	}
}

func TestWithRetry_SucceedsFirstTry(t *testing.T) {
	calls := 0
	var stderr bytes.Buffer
	got, err := withRetry("test", &stderr, func() (Transcript, error) {
		calls++
		return Transcript{Text: "ok"}, nil
	})
	if err != nil {
		t.Fatalf("withRetry: %v", err)
	}
	if calls != 1 {
		t.Fatalf("called %d times, want 1", calls)
	}
	if got.Text != "ok" {
		t.Fatalf("text = %q", got.Text)
	}
	if stderr.Len() != 0 {
		t.Errorf("a first-try success should log nothing, got %q", stderr.String())
	}
}

func TestWithRetry_RecoversAfterTransientFailure(t *testing.T) {
	calls := 0
	var stderr bytes.Buffer
	got, err := withRetry("chunk 1/2", &stderr, func() (Transcript, error) {
		calls++
		if calls < 3 {
			return Transcript{}, errors.New("upstream hiccup")
		}
		return Transcript{Text: "recovered"}, nil
	})
	if err != nil {
		t.Fatalf("withRetry: %v", err)
	}
	if calls != 3 {
		t.Fatalf("called %d times, want 3", calls)
	}
	if got.Text != "recovered" {
		t.Fatalf("text = %q", got.Text)
	}
	if !strings.Contains(stderr.String(), "chunk 1/2") {
		t.Errorf("retries should be logged with their label, got %q", stderr.String())
	}
}

func TestWithRetry_GivesUpAfterMaxAttempts(t *testing.T) {
	calls := 0
	var stderr bytes.Buffer
	_, err := withRetry("test", &stderr, func() (Transcript, error) {
		calls++
		return Transcript{}, errors.New("still broken")
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	if calls != maxRetries {
		t.Fatalf("called %d times, want %d", calls, maxRetries)
	}
	if !strings.Contains(err.Error(), "still broken") {
		t.Fatalf("the underlying error should be wrapped, got: %v", err)
	}
}

func TestReadInput(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.mp3")
	if err := os.WriteFile(path, []byte("AUDIO"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Run("from a file", func(t *testing.T) {
		data, source, err := readInput([]string{path}, strings.NewReader(""))
		if err != nil {
			t.Fatalf("readInput: %v", err)
		}
		if string(data) != "AUDIO" || source != path {
			t.Fatalf("data = %q, source = %q", data, source)
		}
	})

	t.Run("from stdin", func(t *testing.T) {
		data, source, err := readInput(nil, strings.NewReader("PIPED"))
		if err != nil {
			t.Fatalf("readInput: %v", err)
		}
		if string(data) != "PIPED" || source != "stdin" {
			t.Fatalf("data = %q, source = %q", data, source)
		}
	})

	t.Run("empty stdin is an error", func(t *testing.T) {
		if _, _, err := readInput(nil, strings.NewReader("")); err == nil {
			t.Fatal("expected an error")
		}
	})
}

func TestFormatDuration(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{1500 * time.Millisecond, "1.5s"},
		{90 * time.Second, "1m30s"},
		{3725 * time.Second, "1h02m05s"},
		{-time.Second, "0.0s"},
	}
	for _, tc := range cases {
		if got := formatDuration(tc.d); got != tc.want {
			t.Errorf("formatDuration(%v) = %q, want %q", tc.d, got, tc.want)
		}
	}
}

func TestContainsString(t *testing.T) {
	if !containsString([]string{"a", "b"}, "b") {
		t.Error("expected b to be found")
	}
	if containsString([]string{"a"}, "z") {
		t.Error("z should not be found")
	}
	if containsString(nil, "a") {
		t.Error("nothing is in an empty list")
	}
}

func TestDownloadModel(t *testing.T) {
	t.Run("reports an unknown model", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()
		defer swapURL(&modelDownloadBase, srv.URL)()

		var stderr bytes.Buffer
		err := downloadModel("no-such-model", &stderr)
		if err == nil {
			t.Fatal("expected an error")
		}
		if !strings.Contains(err.Error(), "tiny.en") {
			t.Fatalf("the error should suggest valid names, got: %v", err)
		}
	})

	t.Run("rejects an empty name", func(t *testing.T) {
		var stderr bytes.Buffer
		if err := downloadModel("", &stderr); err == nil {
			t.Fatal("expected an error")
		}
	})
}

// downloadModel accepts a bare name, a ggml- prefix, or a full filename, since
// all three appear in whisper.cpp documentation.
func TestDownloadModel_NormalisesTheName(t *testing.T) {
	var requested string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requested = r.URL.Path
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	defer swapURL(&modelDownloadBase, srv.URL)()

	for _, name := range []string{"base.en", "ggml-base.en", "ggml-base.en.bin"} {
		var stderr bytes.Buffer
		_ = downloadModel(name, &stderr)
		if !strings.HasSuffix(requested, "/ggml-base.en.bin") {
			t.Errorf("input %q requested %q, want .../ggml-base.en.bin", name, requested)
		}
	}
}

// A partial download must not be left where the model scanner would find it and
// hand it to an engine as a real model.
func TestDownloadModel_DoesNotLeaveAPartialFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "truncated")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Close mid-transfer.
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				conn.Close()
			}
		}
	}))
	defer srv.Close()
	defer swapURL(&modelDownloadBase, srv.URL)()

	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory")
	}
	dest := filepath.Join(home, ".cache", "golisten", "models", "ggml-golisten-test-model.bin")
	t.Cleanup(func() {
		os.Remove(dest)
		os.Remove(dest + ".part")
	})

	var stderr bytes.Buffer
	if err := downloadModel("golisten-test-model", &stderr); err == nil {
		t.Skip("the server completed the transfer; nothing to assert")
	}
	if _, err := os.Stat(dest); err == nil {
		t.Fatal("a truncated download was left in place as a usable model")
	}
}

func TestParseFlags_Defaults(t *testing.T) {
	var stderr bytes.Buffer
	opts, code, done := parseFlags([]string{"file.mp3"}, &stderr)
	if done {
		t.Fatalf("parsing should not have finished early (code %d)", code)
	}
	if opts.provider != defaultProvider {
		t.Errorf("provider = %q, want %q", opts.provider, defaultProvider)
	}
	if opts.format != defaultFormat {
		t.Errorf("format = %q, want %q", opts.format, defaultFormat)
	}
	if opts.language != defaultLanguage {
		t.Errorf("language = %q, want %q", opts.language, defaultLanguage)
	}
	if len(opts.args) != 1 || opts.args[0] != "file.mp3" {
		t.Errorf("args = %v", opts.args)
	}
}

func TestParseFlags_ShorthandsMatchLongForms(t *testing.T) {
	var stderr bytes.Buffer
	long, _, _ := parseFlags([]string{
		"--provider", "openai", "--model", "m", "--language", "de",
		"--format", "srt", "--output", "o.srt", "--threads", "8",
	}, &stderr)
	short, _, _ := parseFlags([]string{
		"-p", "openai", "-m", "m", "-l", "de", "-f", "srt", "-o", "o.srt", "-t", "8",
	}, &stderr)

	if long.provider != short.provider || long.model != short.model ||
		long.language != short.language || long.format != short.format ||
		long.output != short.output || long.threads != short.threads {
		t.Fatalf("shorthand and long forms disagree:\n%+v\n%+v", long, short)
	}
}

// main() is a thin wrapper around run(); exercise it as a real process so the
// stdin detection and exit-code plumbing are covered.
func TestMainEntryPoint(t *testing.T) {
	if os.Getenv("GOLISTEN_TEST_SUBPROCESS") == "1" {
		main()
		return
	}

	t.Run("no arguments exits non-zero", func(t *testing.T) {
		cmd := exec.Command(os.Args[0], "-test.run=TestMainEntryPoint")
		cmd.Env = append(os.Environ(), "GOLISTEN_TEST_SUBPROCESS=1")
		cmd.Stdin = strings.NewReader("")
		err := cmd.Run()
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) {
			t.Fatalf("expected a non-zero exit, got %v", err)
		}
	})

	t.Run("help exits zero", func(t *testing.T) {
		cmd := exec.Command(os.Args[0], "-test.run=TestMainEntryPoint", "--help")
		cmd.Env = append(os.Environ(), "GOLISTEN_TEST_SUBPROCESS=1")
		var out bytes.Buffer
		cmd.Stderr = &out
		if err := cmd.Run(); err != nil {
			t.Fatalf("--help should exit 0, got %v", err)
		}
		if !strings.Contains(out.String(), "golisten") {
			t.Fatalf("usage was not printed: %q", out.String())
		}
	})
}

func TestResolveDownload(t *testing.T) {
	cases := []struct {
		input    string
		wantFile string
		wantHost string
	}{
		// transcribe.cpp GGUF weights
		{"parakeet", "parakeet-tdt-0.6b-v3-Q4_K_M.gguf", "handy-computer"},
		{"Parakeet", "parakeet-tdt-0.6b-v3-Q4_K_M.gguf", "handy-computer"},
		{"parakeet-q8", "parakeet-tdt-0.6b-v3-Q8_0.gguf", "handy-computer"},
		{"parakeet-v2", "parakeet-tdt-0.6b-v2-Q4_K_M.gguf", "handy-computer"},
		{"parakeet-tdt-0.6b-v3-F16.gguf", "parakeet-tdt-0.6b-v3-F16.gguf", "handy-computer"},
		// whisper.cpp ggml weights, however the name is spelled
		{"base.en", "ggml-base.en.bin", "ggerganov"},
		{"ggml-base.en", "ggml-base.en.bin", "ggerganov"},
		{"ggml-base.en.bin", "ggml-base.en.bin", "ggerganov"},
		{"large-v3-turbo", "ggml-large-v3-turbo.bin", "ggerganov"},
	}
	for _, tc := range cases {
		file, url := resolveDownload(tc.input)
		if file != tc.wantFile {
			t.Errorf("resolveDownload(%q) file = %q, want %q", tc.input, file, tc.wantFile)
		}
		if !strings.Contains(url, tc.wantHost) {
			t.Errorf("resolveDownload(%q) url = %q, want it to point at %s", tc.input, url, tc.wantHost)
		}
		if !strings.HasSuffix(url, file) {
			t.Errorf("resolveDownload(%q) url %q does not end in the filename", tc.input, url)
		}
	}
}

func TestRankTranscribeModel(t *testing.T) {
	better := func(a, b string) {
		t.Helper()
		if rankTranscribeModel(a) >= rankTranscribeModel(b) {
			t.Errorf("%q should rank ahead of %q", a, b)
		}
	}
	// Parakeet is the default family for general speech.
	better("parakeet-tdt-0.6b-v3-Q4_K_M.gguf", "canary-1b-flash-Q4_K_M.gguf")
	better("canary-1b-flash-Q4_K_M.gguf", "whisper-large-v3-Q4_K_M.gguf")
	// Quantisation only breaks ties inside a family.
	better("parakeet-tdt-0.6b-v3-Q8_0.gguf", "parakeet-tdt-0.6b-v3-Q4_K_M.gguf")
	better("parakeet-tdt-0.6b-v3-Q4_K_M.gguf", "parakeet-tdt-0.6b-v3-F32.gguf")
	// A better family beats a better quantisation.
	better("parakeet-tdt-0.6b-v3-F32.gguf", "whisper-large-v3-Q8_0.gguf")
}

// fakeEngine builds an engineLocations whose binary and model both resolve, or
// deliberately do not, without touching the real filesystem.
func fakeEngine(t *testing.T, dir, name string, installed bool) engineLocations {
	t.Helper()
	loc := engineLocations{
		binEnv:     "FAKE_BIN_" + strings.ToUpper(name),
		modelEnv:   "FAKE_MODEL_" + strings.ToUpper(name),
		binNames:   []string{"definitely-not-installed-" + name},
		modelDirs:  []string{filepath.Join(dir, name)},
		modelMatch: func(string) bool { return true },
		install:    "install " + name,
	}
	if !installed {
		return loc
	}
	sub := filepath.Join(dir, name)
	if err := os.MkdirAll(sub, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "model.bin"), []byte("m"), 0644); err != nil {
		t.Fatal(err)
	}
	loc.binPaths = []string{writeScript(t, sub, "engine", "exit 0\n")}
	return loc
}

func TestResolveAutoProvider(t *testing.T) {
	dir := t.TempDir()

	t.Run("prefers transcribe.cpp when it is installed", func(t *testing.T) {
		locs := map[string]engineLocations{
			"transcribe": fakeEngine(t, dir, "transcribe", true),
			"whisper":    fakeEngine(t, dir, "whisper", true),
			"parakeet":   fakeEngine(t, dir, "parakeet", true),
		}
		got, err := resolveAutoProvider(autoProviderOrder(false), locs, "", "", emptyEnv)
		if err != nil {
			t.Fatalf("resolveAutoProvider: %v", err)
		}
		if got != "transcribe" {
			t.Fatalf("provider = %q, want transcribe", got)
		}
	})

	t.Run("falls back to whisper when transcribe.cpp is absent", func(t *testing.T) {
		locs := map[string]engineLocations{
			"transcribe": fakeEngine(t, dir, "missing-transcribe", false),
			"whisper":    fakeEngine(t, dir, "whisper", true),
			"parakeet":   fakeEngine(t, dir, "missing-parakeet", false),
		}
		got, err := resolveAutoProvider(autoProviderOrder(false), locs, "", "", emptyEnv)
		if err != nil {
			t.Fatalf("resolveAutoProvider: %v", err)
		}
		if got != "whisper" {
			t.Fatalf("provider = %q, want whisper", got)
		}
	})

	// A model with a binary but no weights is not usable, and picking it would
	// only produce an error further down.
	t.Run("skips an engine with a binary but no model", func(t *testing.T) {
		half := fakeEngine(t, dir, "halfinstalled", true)
		half.modelDirs = []string{filepath.Join(dir, "nowhere")}
		locs := map[string]engineLocations{
			"transcribe": half,
			"whisper":    fakeEngine(t, dir, "whisper", true),
		}
		got, err := resolveAutoProvider(autoProviderOrder(false), locs, "", "", emptyEnv)
		if err != nil {
			t.Fatalf("resolveAutoProvider: %v", err)
		}
		if got != "whisper" {
			t.Fatalf("provider = %q, want whisper", got)
		}
	})

	// Parakeet has no translate task, so --translate has to start at whisper
	// rather than pick an engine that will refuse the request.
	t.Run("translation prefers whisper", func(t *testing.T) {
		locs := map[string]engineLocations{
			"transcribe": fakeEngine(t, dir, "transcribe", true),
			"whisper":    fakeEngine(t, dir, "whisper", true),
		}
		got, err := resolveAutoProvider(autoProviderOrder(true), locs, "", "", emptyEnv)
		if err != nil {
			t.Fatalf("resolveAutoProvider: %v", err)
		}
		if got != "whisper" {
			t.Fatalf("provider = %q, want whisper when translating", got)
		}
	})

	t.Run("reports how to install when nothing is found", func(t *testing.T) {
		locs := map[string]engineLocations{
			"transcribe": fakeEngine(t, dir, "none-t", false),
			"whisper":    fakeEngine(t, dir, "none-w", false),
			"parakeet":   fakeEngine(t, dir, "none-p", false),
		}
		_, err := resolveAutoProvider(autoProviderOrder(false), locs, "", "", emptyEnv)
		if err == nil {
			t.Fatal("expected an error")
		}
		for _, want := range []string{"--download parakeet", "whisper-cpp", "-p openai"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the error should mention %q, got: %v", want, err)
			}
		}
	})
}

func TestAutoProviderOrder(t *testing.T) {
	if got := autoProviderOrder(false)[0]; got != "transcribe" {
		t.Errorf("default order starts with %q, want transcribe", got)
	}
	if got := autoProviderOrder(true)[0]; got != "whisper" {
		t.Errorf("translate order starts with %q, want whisper", got)
	}
}

func TestCheckProviderSupportsFormat(t *testing.T) {
	if err := checkProviderSupportsFormat("parakeet", true); err == nil {
		t.Error("parakeet cannot produce timed output and should be rejected")
	}
	if err := checkProviderSupportsFormat("parakeet", false); err != nil {
		t.Errorf("plain text is fine for parakeet, got: %v", err)
	}
	for _, p := range []string{"whisper", "transcribe", "openai", "auto"} {
		if err := checkProviderSupportsFormat(p, true); err != nil {
			t.Errorf("%s should accept timed output, got: %v", p, err)
		}
	}
}

// Argument and input errors are more immediate than "no engine installed", so
// they must not be masked by engine discovery.
func TestRun_InputErrorsPrecedeEngineDiscovery(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"/no/such/file.mp3"}, "", emptyEnv)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "failed to read") {
		t.Fatalf("expected the file error first, got %q", stderr)
	}
}
