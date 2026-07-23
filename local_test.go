package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// writeScript drops an executable shell script into dir and returns its path.
// Using a real script keeps the subprocess plumbing — argument passing, exit
// codes, output files, stderr capture — under test rather than mocked away.
func writeScript(t *testing.T, dir, name, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixtures are POSIX-only")
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0755); err != nil {
		t.Fatalf("failed to write %s: %v", name, err)
	}
	return path
}

// captureArgsScript writes a script that records its argv, one per line, and
// then runs extra shell code.
func captureArgsScript(t *testing.T, dir, name, argsFile, extra string) string {
	t.Helper()
	return writeScript(t, dir, name, `
for a in "$@"; do echo "$a" >> `+argsFile+`; done
`+extra+`
`)
}

// findFlagValue returns the argument following flag in a recorded argv dump.
func findFlagValue(t *testing.T, argsFile, flag string) (string, bool) {
	t.Helper()
	data, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("failed to read recorded args: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	for i, l := range lines {
		if l == flag && i+1 < len(lines) {
			return lines[i+1], true
		}
	}
	return "", false
}

func hasFlag(t *testing.T, argsFile, flag string) bool {
	t.Helper()
	data, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("failed to read recorded args: %v", err)
	}
	for _, l := range strings.Split(string(data), "\n") {
		if l == flag {
			return true
		}
	}
	return false
}

const whisperJSONFixture = `{
  "systeminfo": "test",
  "model": {"type": "tiny"},
  "result": {"language": "en"},
  "transcription": [
    {"timestamps": {"from": "00:00:00,000", "to": "00:00:07,960"},
     "offsets": {"from": 0, "to": 7960},
     "text": " And so my fellow Americans"},
    {"timestamps": {"from": "00:00:07,960", "to": "00:00:10,760"},
     "offsets": {"from": 7960, "to": 10760},
     "text": " ask what you can do for your country."}
  ]
}`

func TestTranscribeWhisperCpp_ParsesJSONSidecar(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	// Mimic whisper.cpp: write <out>.json next to the -of base path.
	bin := captureArgsScript(t, dir, "whisper-cli", argsFile, `
OUT=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then shift; OUT="$1"; fi
  shift
done
cat > "$OUT.json" <<'JSON'
`+whisperJSONFixture+`
JSON
`)

	got, err := transcribeWhisperCpp(localRequest{
		WAV:      EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}),
		Binary:   bin,
		Model:    "/models/ggml-tiny.en.bin",
		Language: "auto",
		Threads:  4,
	})
	if err != nil {
		t.Fatalf("transcribeWhisperCpp: %v", err)
	}

	if got.Language != "en" {
		t.Errorf("language = %q, want en", got.Language)
	}
	if len(got.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(got.Segments))
	}
	if got.Segments[0].Start != 0 || got.Segments[0].End != 7960*time.Millisecond {
		t.Errorf("unexpected first segment timings: %+v", got.Segments[0])
	}
	if got.Segments[1].Start != 7960*time.Millisecond {
		t.Errorf("unexpected second segment start: %v", got.Segments[1].Start)
	}
	if got.Text != "And so my fellow Americans ask what you can do for your country." {
		t.Errorf("text = %q", got.Text)
	}

	if v, ok := findFlagValue(t, argsFile, "-m"); !ok || v != "/models/ggml-tiny.en.bin" {
		t.Errorf("model flag = %q", v)
	}
	if v, ok := findFlagValue(t, argsFile, "-l"); !ok || v != "auto" {
		t.Errorf("language flag = %q, want auto", v)
	}
	if v, ok := findFlagValue(t, argsFile, "-t"); !ok || v != "4" {
		t.Errorf("threads flag = %q, want 4", v)
	}
	if !hasFlag(t, argsFile, "-oj") {
		t.Error("-oj was not passed, so there would be no JSON to read")
	}
}

// Every flag sent to whisper.cpp has to exist in the 2023-era `main` binary
// too, otherwise old checkouts fail with "unknown argument".
func TestTranscribeWhisperCpp_UsesOnlyLongLivedFlags(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	bin := captureArgsScript(t, dir, "whisper-cli", argsFile, `
OUT=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then shift; OUT="$1"; fi
  shift
done
echo '{"result":{"language":"en"},"transcription":[]}' > "$OUT.json"
`)

	_, err := transcribeWhisperCpp(localRequest{
		WAV:    EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}),
		Binary: bin, Model: "m", Language: "en", Threads: 2,
		Translate: true, Prompt: "hello",
	})
	if err != nil {
		t.Fatalf("transcribeWhisperCpp: %v", err)
	}

	// Flags present in whisper.cpp since 2022.
	allowed := map[string]bool{
		"-m": true, "-f": true, "-oj": true, "-of": true,
		"-l": true, "-t": true, "-tr": true, "--prompt": true,
	}
	data, _ := os.ReadFile(argsFile)
	for _, arg := range strings.Split(strings.TrimRight(string(data), "\n"), "\n") {
		if strings.HasPrefix(arg, "-") && !allowed[arg] {
			t.Errorf("flag %q is not available in older whisper.cpp builds", arg)
		}
	}
}

func TestTranscribeWhisperCpp_ReportsEngineFailure(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "whisper-cli", "echo 'error: failed to load model' >&2\nexit 1\n")

	_, err := transcribeWhisperCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "failed to load model") {
		t.Fatalf("the engine's own message should be quoted, got: %v", err)
	}
}

// A zero exit code with no output file must not be reported as an empty
// success — that would silently produce an empty transcript.
func TestTranscribeWhisperCpp_MissingJSONIsAnError(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "whisper-cli", "exit 0\n")

	_, err := transcribeWhisperCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	})
	if err == nil {
		t.Fatal("expected an error when no JSON was written")
	}
}

func TestTranscribeWhisperCpp_MalformedJSONIsAnError(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "whisper-cli", `
OUT=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then shift; OUT="$1"; fi
  shift
done
echo 'not json at all' > "$OUT.json"
`)
	_, err := transcribeWhisperCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	})
	if err == nil {
		t.Fatal("expected an error for malformed JSON")
	}
}

func TestTranscribeParakeetCpp_ReadsTextOutput(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	bin := captureArgsScript(t, dir, "parakeet-cli", argsFile, `
OUT=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then shift; OUT="$1"; fi
  shift
done
printf 'first line\nsecond line\n\n' > "$OUT.txt"
`)

	got, err := transcribeParakeetCpp(localRequest{
		WAV:    EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}),
		Binary: bin, Model: "/models/ggml-parakeet.bin", Threads: 8,
	})
	if err != nil {
		t.Fatalf("transcribeParakeetCpp: %v", err)
	}
	if got.Text != "first line second line" {
		t.Fatalf("text = %q", got.Text)
	}
	if len(got.Segments) != 0 {
		t.Fatalf("parakeet-cli has no timings; expected no segments, got %d", len(got.Segments))
	}
	if !hasFlag(t, argsFile, "-otxt") {
		t.Error("-otxt was not passed")
	}
}

func TestTranscribeParakeetCpp_MissingTextIsAnError(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "parakeet-cli", "exit 0\n")
	_, err := transcribeParakeetCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	})
	if err == nil {
		t.Fatal("expected an error when no text file was written")
	}
}

const transcribeJSONLFixture = `{"type":"batch_header","load_ms":120.5}
{"file":"/tmp/audio.wav","text":"hello world","segments":[{"t0_ms":0,"t1_ms":1000,"text":"hello"},{"t0_ms":1000,"t1_ms":2000,"speaker_id":2,"text":"world"}],"mel_ms":1.0,"encode_ms":2.0,"decode_ms":3.0}`

func TestTranscribeTranscribeCpp_ParsesJSONL(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	bin := captureArgsScript(t, dir, "transcribe-cli", argsFile, `
cat <<'JSONL'
`+transcribeJSONLFixture+`
JSONL
`)

	got, err := transcribeTranscribeCpp(localRequest{
		WAV:    EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}),
		Binary: bin, Model: "/models/parakeet.gguf", Language: "en", Threads: 6,
	}, transcribeCppOptions{Diarize: true, Backend: "metal"})
	if err != nil {
		t.Fatalf("transcribeTranscribeCpp: %v", err)
	}

	if got.Text != "hello world" {
		t.Errorf("text = %q", got.Text)
	}
	if len(got.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(got.Segments))
	}
	if got.Segments[1].Start != time.Second || got.Segments[1].End != 2*time.Second {
		t.Errorf("unexpected segment timings: %+v", got.Segments[1])
	}
	if got.Segments[1].Speaker != "speaker_2" {
		t.Errorf("speaker = %q, want speaker_2", got.Segments[1].Speaker)
	}
	if got.Segments[0].Speaker != "" {
		t.Errorf("a segment with no speaker id should stay unlabelled, got %q", got.Segments[0].Speaker)
	}

	// Batch mode is the only machine-readable path, so both flags must be there.
	if !hasFlag(t, argsFile, "--batch-jsonl") {
		t.Error("--batch-jsonl was not passed")
	}
	if !hasFlag(t, argsFile, "--diarize") {
		t.Error("--diarize was not passed")
	}
	if v, ok := findFlagValue(t, argsFile, "--backend"); !ok || v != "metal" {
		t.Errorf("backend flag = %q, want metal", v)
	}
	if v, ok := findFlagValue(t, argsFile, "--threads"); !ok || v != "6" {
		t.Errorf("threads flag = %q, want 6", v)
	}

	// The batch list must name a real file containing the WAV path.
	listPath, ok := findFlagValue(t, argsFile, "--batch")
	if !ok {
		t.Fatal("--batch was not passed")
	}
	if !strings.HasSuffix(listPath, "batch.list") {
		t.Errorf("unexpected batch list path %q", listPath)
	}
}

func TestTranscribeTranscribeCpp_SurfacesRecordError(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "transcribe-cli",
		`echo '{"file":"a.wav","text":"","error":"unsupported language"}'`+"\n")

	_, err := transcribeTranscribeCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	}, transcribeCppOptions{})
	if err == nil || !strings.Contains(err.Error(), "unsupported language") {
		t.Fatalf("expected the record's error to surface, got: %v", err)
	}
}

func TestTranscribeTranscribeCpp_FallsBackToSegmentText(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "transcribe-cli",
		`echo '{"file":"a.wav","text":"","segments":[{"t0_ms":0,"t1_ms":500,"text":"only here"}]}'`+"\n")

	got, err := transcribeTranscribeCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	}, transcribeCppOptions{})
	if err != nil {
		t.Fatalf("transcribeTranscribeCpp: %v", err)
	}
	if got.Text != "only here" {
		t.Fatalf("text = %q, want the segment text", got.Text)
	}
}

func TestParseTranscribeJSONL(t *testing.T) {
	t.Run("skips the batch header", func(t *testing.T) {
		rec, err := parseTranscribeJSONL(transcribeJSONLFixture)
		if err != nil {
			t.Fatalf("parseTranscribeJSONL: %v", err)
		}
		if rec.Text != "hello world" {
			t.Fatalf("text = %q", rec.Text)
		}
	})

	t.Run("ignores non-JSON noise", func(t *testing.T) {
		in := "loading model...\nusing metal\n" + transcribeJSONLFixture
		rec, err := parseTranscribeJSONL(in)
		if err != nil {
			t.Fatalf("parseTranscribeJSONL: %v", err)
		}
		if rec.Text != "hello world" {
			t.Fatalf("text = %q", rec.Text)
		}
	})

	t.Run("errors when there is no record", func(t *testing.T) {
		if _, err := parseTranscribeJSONL(`{"type":"batch_header","load_ms":1}`); err == nil {
			t.Fatal("expected an error when only a header was emitted")
		}
	})

	t.Run("errors on empty output", func(t *testing.T) {
		if _, err := parseTranscribeJSONL(""); err == nil {
			t.Fatal("expected an error for empty output")
		}
	})

	// A long recording's transcript easily exceeds bufio.Scanner's 64 KB
	// default, which would otherwise truncate the result without an error.
	t.Run("handles very long lines", func(t *testing.T) {
		long := strings.Repeat("word ", 60000)
		line := `{"file":"a.wav","text":"` + long + `"}`
		rec, err := parseTranscribeJSONL(line)
		if err != nil {
			t.Fatalf("parseTranscribeJSONL: %v", err)
		}
		if len(rec.Text) != len(long) {
			t.Fatalf("text was truncated: got %d bytes, want %d", len(rec.Text), len(long))
		}
	})
}

func TestTailLines(t *testing.T) {
	in := "one\n\ntwo\nthree\nfour\n"
	if got := tailLines(in, 2); got != "three\nfour" {
		t.Fatalf("tailLines = %q", got)
	}
	if got := tailLines("only", 5); got != "only" {
		t.Fatalf("tailLines = %q", got)
	}
}

func TestFindBinary(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "engine", "exit 0\n")
	loc := engineLocations{binEnv: "TEST_BIN", binNames: []string{"definitely-not-a-real-binary-xyz"}}

	t.Run("explicit path wins", func(t *testing.T) {
		got, err := findBinary(loc, bin, emptyEnv)
		if err != nil || got != bin {
			t.Fatalf("findBinary = %q, %v", got, err)
		}
	})

	t.Run("env var is used next", func(t *testing.T) {
		got, err := findBinary(loc, "", envMap(map[string]string{"TEST_BIN": bin}))
		if err != nil || got != bin {
			t.Fatalf("findBinary = %q, %v", got, err)
		}
	})

	t.Run("a bad env var is reported, not ignored", func(t *testing.T) {
		_, err := findBinary(loc, "", envMap(map[string]string{"TEST_BIN": "/no/such/binary"}))
		if err == nil || !strings.Contains(err.Error(), "TEST_BIN") {
			t.Fatalf("expected the env var to be named in the error, got: %v", err)
		}
	})

	t.Run("a bad explicit path is an error", func(t *testing.T) {
		if _, err := findBinary(loc, "/no/such/binary", emptyEnv); err == nil {
			t.Fatal("expected an error")
		}
	})

	t.Run("a directory is not an executable", func(t *testing.T) {
		if _, err := findBinary(loc, dir, emptyEnv); err == nil {
			t.Fatal("a directory should not be accepted as a binary")
		}
	})

	t.Run("nothing found gives installation guidance", func(t *testing.T) {
		empty := engineLocations{
			binEnv: "TEST_BIN", binNames: []string{"definitely-not-a-real-binary-xyz"},
			install: "brew install something",
		}
		_, err := findBinary(empty, "", emptyEnv)
		if err == nil || !strings.Contains(err.Error(), "brew install something") {
			t.Fatalf("expected install guidance, got: %v", err)
		}
	})
}

func TestFindModel(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"ggml-tiny.en.bin", "ggml-large-v3.bin", "for-tests-ggml-base.bin", "ggml-silero-vad.bin"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	loc := engineLocations{
		modelEnv:   "TEST_MODEL",
		modelDirs:  []string{dir},
		modelMatch: isWhisperModel,
		install:    "golisten --download base.en",
	}

	t.Run("picks the highest-ranked model", func(t *testing.T) {
		got, err := findModel(loc, "", emptyEnv)
		if err != nil {
			t.Fatalf("findModel: %v", err)
		}
		if filepath.Base(got) != "ggml-large-v3.bin" {
			t.Fatalf("model = %q, want ggml-large-v3.bin", filepath.Base(got))
		}
	})

	t.Run("a bare name selects that model", func(t *testing.T) {
		got, err := findModel(loc, "tiny.en", emptyEnv)
		if err != nil {
			t.Fatalf("findModel: %v", err)
		}
		if filepath.Base(got) != "ggml-tiny.en.bin" {
			t.Fatalf("model = %q, want ggml-tiny.en.bin", filepath.Base(got))
		}
	})

	t.Run("an explicit path is used as given", func(t *testing.T) {
		path := filepath.Join(dir, "ggml-tiny.en.bin")
		got, err := findModel(loc, path, emptyEnv)
		if err != nil || got != path {
			t.Fatalf("findModel = %q, %v", got, err)
		}
	})

	t.Run("env var is honoured", func(t *testing.T) {
		path := filepath.Join(dir, "ggml-tiny.en.bin")
		got, err := findModel(loc, "", envMap(map[string]string{"TEST_MODEL": path}))
		if err != nil || got != path {
			t.Fatalf("findModel = %q, %v", got, err)
		}
	})

	t.Run("a bad env var is reported", func(t *testing.T) {
		_, err := findModel(loc, "", envMap(map[string]string{"TEST_MODEL": "/no/such/model.bin"}))
		if err == nil || !strings.Contains(err.Error(), "TEST_MODEL") {
			t.Fatalf("expected the env var to be named, got: %v", err)
		}
	})

	t.Run("an unknown name is an error", func(t *testing.T) {
		if _, err := findModel(loc, "no-such-model", emptyEnv); err == nil {
			t.Fatal("expected an error")
		}
	})

	t.Run("nothing found gives installation guidance", func(t *testing.T) {
		empty := engineLocations{
			modelEnv: "TEST_MODEL", modelDirs: []string{filepath.Join(dir, "nope")},
			modelMatch: isWhisperModel, install: "golisten --download base.en",
		}
		_, err := findModel(empty, "", emptyEnv)
		if err == nil || !strings.Contains(err.Error(), "--download") {
			t.Fatalf("expected install guidance, got: %v", err)
		}
	})
}

// whisper.cpp ships dummy fixtures and VAD models in the same directory as real
// models. Auto-selecting one of those produces confident nonsense, so they must
// never match.
func TestIsWhisperModel(t *testing.T) {
	cases := map[string]bool{
		"ggml-tiny.en.bin":                  true,
		"ggml-large-v3-turbo.bin":           true,
		"ggml-medium-q5_0.bin":              true,
		"for-tests-ggml-base.bin":           false,
		"ggml-silero-v6.2.0-ggml.bin":       false,
		"ggml-parakeet-tdt-0.6b-v3.bin":     false,
		"whisper.cpp":                       false,
		"ggml-tiny.en.mlmodelc":             false,
		"parakeet-tdt-0.6b-v2-F32.gguf":     false,
		"ggml-base.en-encoder.mlmodelc.zip": false,
	}
	for name, want := range cases {
		if got := isWhisperModel(name); got != want {
			t.Errorf("isWhisperModel(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestIsParakeetModel(t *testing.T) {
	if !isParakeetModel("ggml-parakeet-tdt-0.6b-v3.bin") {
		t.Error("expected a parakeet ggml model to match")
	}
	if isParakeetModel("ggml-tiny.en.bin") {
		t.Error("a whisper model should not match the parakeet pattern")
	}
}

func TestIsGGUFModel(t *testing.T) {
	if !isGGUFModel("parakeet-tdt-0.6b-v2-Q4_K_M.gguf") {
		t.Error("expected a .gguf file to match")
	}
	if isGGUFModel("ggml-tiny.en.bin") {
		t.Error("a .bin file should not match")
	}
}

func TestRankWhisperModel(t *testing.T) {
	if rankWhisperModel("ggml-large-v3.bin") >= rankWhisperModel("ggml-tiny.en.bin") {
		t.Error("large should rank ahead of tiny")
	}
	if rankWhisperModel("ggml-medium.bin") >= rankWhisperModel("ggml-base.bin") {
		t.Error("medium should rank ahead of base")
	}
	if rankWhisperModel("something-else.bin") != len(whisperModelRank) {
		t.Error("an unrecognised name should rank last")
	}
}

func TestExpandHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory available")
	}
	if got := expandHome("~/models"); got != filepath.Join(home, "models") {
		t.Errorf("expandHome = %q", got)
	}
	if got := expandHome("/absolute/path"); got != "/absolute/path" {
		t.Errorf("expandHome should leave absolute paths alone, got %q", got)
	}
}

func TestIsExecutableFile(t *testing.T) {
	dir := t.TempDir()
	script := writeScript(t, dir, "runme", "exit 0\n")
	plain := filepath.Join(dir, "plain.txt")
	if err := os.WriteFile(plain, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	if !isExecutableFile(script) {
		t.Error("an executable script should be recognised")
	}
	if isExecutableFile(plain) {
		t.Error("a non-executable file should be rejected")
	}
	if isExecutableFile(dir) {
		t.Error("a directory should be rejected")
	}
	if isExecutableFile(filepath.Join(dir, "missing")) {
		t.Error("a missing file should be rejected")
	}
}

// The scratch directory must not survive the call, or long transcription
// sessions would leak a copy of every input file into the temp directory.
func TestWithTempDir_CleansUp(t *testing.T) {
	var captured string
	err := withTempDir([]byte("audio"), func(dir, wavPath string) error {
		captured = dir
		if _, err := os.Stat(wavPath); err != nil {
			t.Fatalf("the WAV should exist inside the callback: %v", err)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("withTempDir: %v", err)
	}
	if _, err := os.Stat(captured); !os.IsNotExist(err) {
		t.Fatalf("temp dir %s was not removed", captured)
	}
}

func TestWithTempDir_PropagatesError(t *testing.T) {
	sentinel := os.ErrInvalid
	err := withTempDir([]byte("audio"), func(string, string) error { return sentinel })
	if err != sentinel {
		t.Fatalf("expected the callback's error, got %v", err)
	}
}

// transcribe.cpp exits non-zero when it rejects a request but still writes a
// JSONL record naming the reason. Without preferring that record the user sees
// only "exit status 1" followed by a screenful of backend logging.
func TestTranscribeTranscribeCpp_PrefersRecordErrorOverExitStatus(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "transcribe-cli", `
echo '{"type":"batch_header","load_ms":1}'
echo '{"file":"a.wav","text":"","error":"unsupported task"}'
echo 'ggml_metal_init: allocating' >&2
echo 'ggml_metal_free: deallocating' >&2
exit 1
`)

	_, err := transcribeTranscribeCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
		Translate: true,
	}, transcribeCppOptions{})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "unsupported task") {
		t.Fatalf("the record's reason should be reported, got: %v", err)
	}
	if strings.Contains(err.Error(), "ggml_metal") {
		t.Fatalf("backend logging should not drown the message, got: %v", err)
	}
}

// A non-zero exit with no JSONL record still has to report the engine's stderr.
func TestTranscribeTranscribeCpp_FallsBackToExitStatus(t *testing.T) {
	dir := t.TempDir()
	bin := writeScript(t, dir, "transcribe-cli", "echo 'error: cannot open model' >&2\nexit 1\n")

	_, err := transcribeTranscribeCpp(localRequest{
		WAV: EncodeWAV(Audio{Samples: make([]float32, 16), Rate: 16000}), Binary: bin, Model: "m",
	}, transcribeCppOptions{})
	if err == nil || !strings.Contains(err.Error(), "cannot open model") {
		t.Fatalf("expected the engine's stderr, got: %v", err)
	}
}
