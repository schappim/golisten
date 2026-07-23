package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// localRequest is everything a locally-installed engine needs for one run. The
// WAV has already been normalised to 16 kHz mono by the caller.
type localRequest struct {
	WAV       []byte
	Binary    string
	Model     string
	Language  string
	Threads   int
	Translate bool
	Prompt    string
	Verbose   bool
	Stderr    io.Writer
}

// execCommand is wired through a package var so tests can intercept the
// subprocess without a real engine installed.
var execCommand = exec.Command

// runEngine executes a local engine binary in workDir. Child stderr is streamed
// through when verbose, and always captured so failures can quote it.
func runEngine(req localRequest, workDir string, args ...string) (stdout string, err error) {
	cmd := execCommand(req.Binary, args...)
	cmd.Dir = workDir

	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	if req.Verbose && req.Stderr != nil {
		cmd.Stderr = io.MultiWriter(&errBuf, req.Stderr)
	} else {
		cmd.Stderr = &errBuf
	}

	if runErr := cmd.Run(); runErr != nil {
		return out.String(), fmt.Errorf("%s failed: %w\n%s",
			filepath.Base(req.Binary), runErr, tailLines(errBuf.String(), 12))
	}
	return out.String(), nil
}

// tailLines returns the last n non-empty lines, so an engine's error is visible
// without replaying its entire startup banner.
func tailLines(s string, n int) string {
	lines := []string{}
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			lines = append(lines, l)
		}
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// withTempDir runs fn against a fresh scratch directory holding the input WAV,
// then cleans up. Engines write their output files next to it.
func withTempDir(wav []byte, fn func(dir, wavPath string) error) error {
	dir, err := os.MkdirTemp("", "golisten-")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	wavPath := filepath.Join(dir, "audio.wav")
	if err := os.WriteFile(wavPath, wav, 0600); err != nil {
		return fmt.Errorf("failed to write temp audio: %w", err)
	}
	return fn(dir, wavPath)
}

// ---------------------------------------------------------------------------
// whisper.cpp
// ---------------------------------------------------------------------------

// whisperJSON is the schema whisper.cpp writes for -oj. It has been stable
// since 2022, and only the fields used here are declared.
type whisperJSON struct {
	Result struct {
		Language string `json:"language"`
	} `json:"result"`
	Transcription []struct {
		Offsets struct {
			From int64 `json:"from"`
			To   int64 `json:"to"`
		} `json:"offsets"`
		Text string `json:"text"`
	} `json:"transcription"`
}

// transcribeWhisperCpp shells out to whisper.cpp and reads its JSON sidecar
// rather than scraping stdout, so the result survives changes to the console
// output across whisper.cpp versions.
//
// The flag set is deliberately conservative — every flag used here exists in
// both the modern whisper-cli and the older `main` binary, so an ancient local
// checkout still works.
func transcribeWhisperCpp(req localRequest) (Transcript, error) {
	var result Transcript
	err := withTempDir(req.WAV, func(dir, wavPath string) error {
		base := filepath.Join(dir, "out")
		args := []string{
			"-m", req.Model,
			"-f", wavPath,
			"-oj",
			"-of", base,
		}
		if req.Language != "" {
			args = append(args, "-l", req.Language)
		}
		if req.Threads > 0 {
			args = append(args, "-t", fmt.Sprint(req.Threads))
		}
		if req.Translate {
			args = append(args, "-tr")
		}
		if req.Prompt != "" {
			args = append(args, "--prompt", req.Prompt)
		}

		if _, err := runEngine(req, dir, args...); err != nil {
			return err
		}

		data, err := os.ReadFile(base + ".json")
		if err != nil {
			return fmt.Errorf("whisper.cpp produced no JSON output: %w", err)
		}
		var parsed whisperJSON
		if err := json.Unmarshal(data, &parsed); err != nil {
			return fmt.Errorf("failed to parse whisper.cpp JSON: %w", err)
		}

		result.Language = parsed.Result.Language
		for _, seg := range parsed.Transcription {
			result.Segments = append(result.Segments, Segment{
				Start: time.Duration(seg.Offsets.From) * time.Millisecond,
				End:   time.Duration(seg.Offsets.To) * time.Millisecond,
				Text:  strings.TrimSpace(seg.Text),
			})
		}
		result.Text = textFromSegments(result.Segments)
		return nil
	})
	return result, err
}

// ---------------------------------------------------------------------------
// whisper.cpp's parakeet-cli
// ---------------------------------------------------------------------------

// transcribeParakeetCpp runs whisper.cpp's parakeet-cli. That binary only
// writes plain text — its -ps segment dump goes to stderr in a debug format
// that is not a stable interface — so this backend returns untimed text. Use
// the transcribe.cpp backend when parakeet output needs timestamps.
func transcribeParakeetCpp(req localRequest) (Transcript, error) {
	var result Transcript
	err := withTempDir(req.WAV, func(dir, wavPath string) error {
		base := filepath.Join(dir, "out")
		args := []string{
			"-m", req.Model,
			"-f", wavPath,
			"-otxt",
			"-of", base,
		}
		if req.Threads > 0 {
			args = append(args, "-t", fmt.Sprint(req.Threads))
		}

		if _, err := runEngine(req, dir, args...); err != nil {
			return err
		}

		data, err := os.ReadFile(base + ".txt")
		if err != nil {
			return fmt.Errorf("parakeet-cli produced no text output: %w", err)
		}
		// One segment per line, without timings.
		var parts []string
		for _, line := range strings.Split(string(data), "\n") {
			if l := strings.TrimSpace(line); l != "" {
				parts = append(parts, l)
			}
		}
		result.Text = strings.Join(parts, " ")
		return nil
	})
	return result, err
}

// ---------------------------------------------------------------------------
// transcribe.cpp
// ---------------------------------------------------------------------------

// transcribeJSONLRecord is one line of transcribe-cli's --batch-jsonl output.
type transcribeJSONLRecord struct {
	Type     string `json:"type"`
	File     string `json:"file"`
	Text     string `json:"text"`
	Error    string `json:"error"`
	Segments []struct {
		T0MS      int64  `json:"t0_ms"`
		T1MS      int64  `json:"t1_ms"`
		Text      string `json:"text"`
		SpeakerID int    `json:"speaker_id"`
	} `json:"segments"`
}

// transcribeCppOptions carries the settings that only transcribe.cpp exposes.
type transcribeCppOptions struct {
	Diarize        bool
	TargetLanguage string
	Backend        string
}

// transcribeTranscribeCpp runs transcribe.cpp. Even for a single file it uses
// batch mode, because --batch-jsonl is the only machine-readable output the CLI
// offers — the single-file path prints a human report that would have to be
// scraped.
//
// transcribe.cpp supports parakeet, canary, whisper and a dozen other model
// families from one binary, so this is the backend to use for timestamped
// parakeet output.
func transcribeTranscribeCpp(req localRequest, opts transcribeCppOptions) (Transcript, error) {
	var result Transcript
	err := withTempDir(req.WAV, func(dir, wavPath string) error {
		listPath := filepath.Join(dir, "batch.list")
		if err := os.WriteFile(listPath, []byte(wavPath+"\n"), 0600); err != nil {
			return fmt.Errorf("failed to write batch list: %w", err)
		}

		args := []string{
			"-m", req.Model,
			"--batch", listPath,
			"--batch-jsonl",
			"--timestamps", "segment",
			"-q",
		}
		if req.Language != "" {
			args = append(args, "-l", req.Language)
		}
		if req.Threads > 0 {
			args = append(args, "--threads", fmt.Sprint(req.Threads))
		}
		if req.Translate {
			args = append(args, "-t")
		}
		if opts.TargetLanguage != "" {
			args = append(args, "--target-language", opts.TargetLanguage)
		}
		if req.Prompt != "" {
			args = append(args, "--initial-prompt", req.Prompt)
		}
		if opts.Diarize {
			args = append(args, "--diarize")
		}
		if opts.Backend != "" {
			args = append(args, "--backend", opts.Backend)
		}

		stdout, runErr := runEngine(req, dir, args...)
		if runErr != nil {
			// transcribe.cpp still writes its JSONL record when it rejects a
			// request, and that record names the actual problem ("unsupported
			// task", "unsupported language"). Prefer it over an exit status
			// trailed by a screenful of backend initialisation logging.
			if rec, perr := parseTranscribeJSONL(stdout); perr == nil && rec.Error != "" {
				return fmt.Errorf("transcribe.cpp reported an error: %s", rec.Error)
			}
			return runErr
		}

		rec, err := parseTranscribeJSONL(stdout)
		if err != nil {
			return err
		}
		if rec.Error != "" {
			return fmt.Errorf("transcribe.cpp reported an error: %s", rec.Error)
		}

		for _, seg := range rec.Segments {
			s := Segment{
				Start: time.Duration(seg.T0MS) * time.Millisecond,
				End:   time.Duration(seg.T1MS) * time.Millisecond,
				Text:  strings.TrimSpace(seg.Text),
			}
			if seg.SpeakerID > 0 {
				s.Speaker = fmt.Sprintf("speaker_%d", seg.SpeakerID)
			}
			result.Segments = append(result.Segments, s)
		}
		result.Text = strings.TrimSpace(rec.Text)
		if result.Text == "" {
			result.Text = textFromSegments(result.Segments)
		}
		return nil
	})
	return result, err
}

// parseTranscribeJSONL pulls the single result record out of transcribe-cli's
// JSONL stream, skipping the batch_header line and anything that is not JSON.
func parseTranscribeJSONL(stdout string) (transcribeJSONLRecord, error) {
	scanner := bufio.NewScanner(strings.NewReader(stdout))
	// Transcripts of long audio comfortably exceed bufio's 64 KB default.
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var rec transcribeJSONLRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
		if rec.Type == "batch_header" {
			continue
		}
		return rec, nil
	}
	if err := scanner.Err(); err != nil {
		return transcribeJSONLRecord{}, fmt.Errorf("failed to read transcribe.cpp output: %w", err)
	}
	return transcribeJSONLRecord{}, fmt.Errorf("transcribe.cpp returned no transcription record")
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

// engineLocations describes where an engine's binary and models usually live.
type engineLocations struct {
	binEnv     string
	modelEnv   string
	binNames   []string // looked up on PATH
	binPaths   []string // explicit candidates, ~ expanded
	modelDirs  []string
	modelMatch func(name string) bool
	// modelRank orders candidates within a directory, best first. Nil falls
	// back to alphabetical order.
	modelRank func(name string) int
	install   string // shown when nothing is found
}

var whisperLocations = engineLocations{
	binEnv:   "GOLISTEN_WHISPER_BIN",
	modelEnv: "GOLISTEN_WHISPER_MODEL",
	binNames: []string{"whisper-cli", "whisper-cpp"},
	binPaths: []string{
		"~/.cache/golisten/bin/whisper-cli",
		"/opt/homebrew/bin/whisper-cli",
		"/usr/local/bin/whisper-cli",
		"~/whisper.cpp/build/bin/whisper-cli",
		"~/code/whisper.cpp/build/bin/whisper-cli",
		"~/src/whisper.cpp/build/bin/whisper-cli",
		"./build/bin/whisper-cli",
		// Pre-2024 checkouts built a binary called `main`. Only ever accepted
		// from a whisper.cpp directory — never off PATH, where the name means
		// nothing.
		"~/whisper.cpp/main",
		"~/code/whisper.cpp/main",
		"~/src/whisper.cpp/main",
	},
	modelDirs: []string{
		"~/.cache/golisten/models",
		"~/Library/Application Support/MacWhisper/models",
		"/opt/homebrew/share/whisper-cpp/models",
		"/usr/local/share/whisper-cpp/models",
		"~/.cache/whisper.cpp",
		"~/whisper.cpp/models",
		"~/code/whisper.cpp/models",
		"~/src/whisper.cpp/models",
		"./models",
	},
	modelMatch: isWhisperModel,
	modelRank:  rankWhisperModel,
	install:    "brew install whisper-cpp, then: golisten --download base.en",
}

var parakeetLocations = engineLocations{
	binEnv:   "GOLISTEN_PARAKEET_BIN",
	modelEnv: "GOLISTEN_PARAKEET_MODEL",
	binNames: []string{"parakeet-cli"},
	binPaths: []string{
		"~/.cache/golisten/bin/parakeet-cli",
		"/opt/homebrew/bin/parakeet-cli",
		"/usr/local/bin/parakeet-cli",
		"~/whisper.cpp/build/bin/parakeet-cli",
		"~/code/whisper.cpp/build/bin/parakeet-cli",
		"~/src/whisper.cpp/build/bin/parakeet-cli",
		"./build/bin/parakeet-cli",
	},
	modelDirs: []string{
		"~/.cache/golisten/models",
		"~/whisper.cpp/models",
		"~/code/whisper.cpp/models",
		"~/src/whisper.cpp/models",
		"./models",
	},
	modelMatch: isParakeetModel,
	install:    "build whisper.cpp (it ships parakeet-cli) and fetch a ggml parakeet model",
}

var transcribeLocations = engineLocations{
	binEnv:   "GOLISTEN_TRANSCRIBE_BIN",
	modelEnv: "GOLISTEN_TRANSCRIBE_MODEL",
	binNames: []string{"transcribe-cli"},
	binPaths: []string{
		"~/.cache/golisten/bin/transcribe-cli",
		"/opt/homebrew/bin/transcribe-cli",
		"/usr/local/bin/transcribe-cli",
		"~/transcribe.cpp/build/bin/transcribe-cli",
		"~/code/transcribe.cpp/build/bin/transcribe-cli",
		"~/src/transcribe.cpp/build/bin/transcribe-cli",
		"./build/bin/transcribe-cli",
	},
	modelDirs: []string{
		"~/.cache/golisten/models",
		"~/transcribe.cpp/models",
		"~/code/transcribe.cpp/models",
		"~/src/transcribe.cpp/models",
		"./models",
	},
	modelMatch: isGGUFModel,
	modelRank:  rankTranscribeModel,
	install:    "build transcribe.cpp, then: golisten --download parakeet",
}

// isWhisperModel matches whisper.cpp's own ggml naming, excluding the dummy
// fixtures and the VAD models that share the directory — auto-picking one of
// those produces confident nonsense.
func isWhisperModel(name string) bool {
	lower := strings.ToLower(name)
	if !strings.HasPrefix(lower, "ggml-") || !strings.HasSuffix(lower, ".bin") {
		return false
	}
	if strings.Contains(lower, "for-tests") || strings.Contains(lower, "silero") ||
		strings.Contains(lower, "vad") || strings.Contains(lower, "parakeet") {
		return false
	}
	return true
}

func isParakeetModel(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "parakeet") && strings.HasSuffix(lower, ".bin")
}

func isGGUFModel(name string) bool {
	return strings.HasSuffix(strings.ToLower(name), ".gguf")
}

// whisperModelRank orders models best-first so auto-discovery lands on the most
// capable model present rather than whichever sorts first alphabetically.
var whisperModelRank = []string{
	"large-v3-turbo", "large-v3", "large-v2", "large",
	"medium.en", "medium", "small.en", "small",
	"base.en", "base", "tiny.en", "tiny",
}

func rankWhisperModel(name string) int {
	lower := strings.ToLower(name)
	for i, tag := range whisperModelRank {
		if strings.Contains(lower, tag) {
			return i
		}
	}
	return len(whisperModelRank)
}

// expandHome resolves a leading ~ against the user's home directory.
func expandHome(path string) string {
	if !strings.HasPrefix(path, "~") {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, strings.TrimPrefix(path, "~"))
}

// isExecutableFile reports whether path is a regular file with an exec bit.
func isExecutableFile(path string) bool {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Mode()&0111 != 0
}

// findBinary resolves the engine binary: explicit flag, then env var, then
// PATH, then the well-known build locations.
func findBinary(loc engineLocations, explicit string, getenv func(string) string) (string, error) {
	if explicit != "" {
		if isExecutableFile(explicit) {
			return explicit, nil
		}
		if resolved, err := exec.LookPath(explicit); err == nil {
			return resolved, nil
		}
		return "", fmt.Errorf("%s is not an executable", explicit)
	}
	if env := getenv(loc.binEnv); env != "" {
		if isExecutableFile(env) {
			return env, nil
		}
		return "", fmt.Errorf("%s points at %s, which is not an executable", loc.binEnv, env)
	}
	for _, name := range loc.binNames {
		if resolved, err := exec.LookPath(name); err == nil {
			return resolved, nil
		}
	}
	for _, candidate := range loc.binPaths {
		if p := expandHome(candidate); isExecutableFile(p) {
			return p, nil
		}
	}
	return "", fmt.Errorf("could not find %s.\nSet %s, pass --bin, or install it: %s",
		strings.Join(loc.binNames, " or "), loc.binEnv, loc.install)
}

// findModel resolves the model file: explicit flag, then env var, then a scan
// of the well-known model directories.
func findModel(loc engineLocations, explicit string, getenv func(string) string) (string, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err == nil {
			return explicit, nil
		}
		// A bare name like "base.en" is resolved against the model directories.
		if found := scanModelDirs(loc, func(name string) bool {
			return loc.modelMatch(name) && strings.Contains(strings.ToLower(name), strings.ToLower(explicit))
		}); found != "" {
			return found, nil
		}
		return "", fmt.Errorf("model %q not found", explicit)
	}
	if env := getenv(loc.modelEnv); env != "" {
		if _, err := os.Stat(env); err == nil {
			return env, nil
		}
		return "", fmt.Errorf("%s points at %s, which does not exist", loc.modelEnv, env)
	}
	if found := scanModelDirs(loc, loc.modelMatch); found != "" {
		return found, nil
	}
	return "", fmt.Errorf("could not find a model file.\nSet %s, pass -m, or install one: %s",
		loc.modelEnv, loc.install)
}

// scanModelDirs walks the candidate directories in priority order and returns
// the best match in the first directory that has one.
func scanModelDirs(loc engineLocations, match func(string) bool) string {
	for _, dir := range loc.modelDirs {
		expanded := expandHome(dir)
		entries, err := os.ReadDir(expanded)
		if err != nil {
			continue
		}
		var names []string
		for _, e := range entries {
			if !e.IsDir() && match(e.Name()) {
				names = append(names, e.Name())
			}
		}
		if len(names) == 0 {
			continue
		}
		rank := loc.modelRank
		if rank == nil {
			rank = func(string) int { return 0 }
		}
		sort.Slice(names, func(i, j int) bool {
			ri, rj := rank(names[i]), rank(names[j])
			if ri != rj {
				return ri < rj
			}
			return names[i] < names[j]
		})
		return filepath.Join(expanded, names[0])
	}
	return ""
}

// transcribeModelRank orders the model families transcribe.cpp can run,
// best-first for general speech. Parakeet leads because it is both faster and
// more accurate than the whisper models most people have lying around.
var transcribeModelRank = []string{
	"parakeet-tdt", "parakeet-unified", "parakeet-rnnt", "parakeet-ctc", "parakeet",
	"canary-qwen", "canary",
	"nemotron", "gigaam", "qwen3-asr", "voxtral", "granite", "moonshine", "sensevoice",
	"whisper",
}

// transcribeQuantRank breaks ties within a family. Q8_0 keeps essentially all
// of the accuracy at a third of the size, so it is preferred over the larger
// float weights and the lossier low-bit quantisations.
var transcribeQuantRank = []string{"q8_0", "q6_k", "q5_k_m", "q4_k_m", "f16", "f32"}

// rankTranscribeModel scores a GGUF filename by family first, then by
// quantisation, so a directory holding several models yields the best general
// choice rather than whichever sorts first alphabetically.
func rankTranscribeModel(name string) int {
	lower := strings.ToLower(name)

	family := len(transcribeModelRank)
	for i, tag := range transcribeModelRank {
		if strings.Contains(lower, tag) {
			family = i
			break
		}
	}
	quant := len(transcribeQuantRank)
	for i, tag := range transcribeQuantRank {
		if strings.Contains(lower, tag) {
			quant = i
			break
		}
	}
	// Family dominates; quantisation only separates models of the same family.
	return family*(len(transcribeQuantRank)+1) + quant
}

// engineLocationsByName maps a local backend to where its binary and models
// live.
var engineLocationsByName = map[string]engineLocations{
	"whisper":    whisperLocations,
	"parakeet":   parakeetLocations,
	"transcribe": transcribeLocations,
}

// autoProviderOrder is the order backends are tried when no -p is given.
// transcribe.cpp comes first so parakeet is the default when it is installed.
// Translation reorders it: parakeet has no translate task, so asking for
// English output has to start with whisper rather than fail.
func autoProviderOrder(translate bool) []string {
	if translate {
		return []string{"whisper", "transcribe", "parakeet"}
	}
	return []string{"transcribe", "whisper", "parakeet"}
}

// resolveAutoProvider picks the first local backend that has both a binary and
// a model available. An explicit -m therefore also selects the engine: a .gguf
// name resolves to transcribe.cpp, a ggml-*.bin name to whisper.cpp.
func resolveAutoProvider(order []string, locations map[string]engineLocations,
	binary, model string, getenv func(string) string) (string, error) {

	for _, name := range order {
		loc, ok := locations[name]
		if !ok {
			continue
		}
		if _, err := findBinary(loc, binary, getenv); err != nil {
			continue
		}
		if _, err := findModel(loc, model, getenv); err != nil {
			continue
		}
		return name, nil
	}
	return "", fmt.Errorf("no local speech-to-text engine found.\n"+
		"Install one:\n"+
		"  transcribe.cpp (parakeet, the default):  build it, then golisten --download parakeet\n"+
		"  whisper.cpp:                             brew install whisper-cpp, then golisten --download base.en\n"+
		"Or use a hosted backend, e.g. golisten -p openai %s", "audio.mp3")
}
