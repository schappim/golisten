package main

import (
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

// swapURL repoints a provider endpoint at a test server and returns a restore
// function.
func swapURL(target *string, url string) func() {
	original := *target
	*target = url
	return func() { *target = original }
}

// capturedUpload is what a fake provider saw.
type capturedUpload struct {
	method  string
	path    string
	query   url.Values
	headers http.Header
	fields  map[string]string
	file    []byte
	rawBody []byte
}

// multipartServer stands in for a provider that accepts multipart uploads.
func multipartServer(t *testing.T, status int, response string) (*httptest.Server, *capturedUpload) {
	t.Helper()
	captured := &capturedUpload{fields: map[string]string{}}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.method = r.Method
		captured.path = r.URL.Path
		captured.query = r.URL.Query()
		captured.headers = r.Header.Clone()

		_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err == nil && params["boundary"] != "" {
			mr := multipart.NewReader(r.Body, params["boundary"])
			for {
				part, err := mr.NextPart()
				if err != nil {
					break
				}
				data, _ := io.ReadAll(part)
				if part.FileName() != "" {
					captured.file = data
					captured.fields["__filename"] = part.FileName()
				} else {
					captured.fields[part.FormName()] = string(data)
				}
			}
		} else {
			captured.rawBody, _ = io.ReadAll(r.Body)
		}

		w.WriteHeader(status)
		io.WriteString(w, response)
	}))
	t.Cleanup(srv.Close)
	return srv, captured
}

const openAIVerboseFixture = `{
  "task": "transcribe",
  "language": "english",
  "duration": 10.76,
  "text": "And so my fellow Americans ask what you can do for your country.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 7.96, "text": " And so my fellow Americans"},
    {"id": 1, "start": 7.96, "end": 10.76, "text": " ask what you can do for your country."}
  ]
}`

func TestTranscribeOpenAI_SendsExpectedRequest(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, openAIVerboseFixture)
	defer swapURL(&openAIAPIURL, srv.URL)()

	got, err := transcribeOpenAI(cloudRequest{
		Audio: []byte("MP3DATA"), MIME: "audio/mpeg", Ext: "mp3",
		APIKey: "sk-test", Model: "whisper-1", Language: "en",
		Prompt: "OpenAI, DALL-E", NeedSegments: true,
	})
	if err != nil {
		t.Fatalf("transcribeOpenAI: %v", err)
	}

	if captured.method != "POST" {
		t.Errorf("method = %s, want POST", captured.method)
	}
	if auth := captured.headers.Get("Authorization"); auth != "Bearer sk-test" {
		t.Errorf("Authorization = %q", auth)
	}
	if captured.fields["model"] != "whisper-1" {
		t.Errorf("model field = %q", captured.fields["model"])
	}
	if captured.fields["response_format"] != "verbose_json" {
		t.Errorf("response_format = %q, want verbose_json", captured.fields["response_format"])
	}
	if captured.fields["language"] != "en" {
		t.Errorf("language = %q", captured.fields["language"])
	}
	if captured.fields["prompt"] != "OpenAI, DALL-E" {
		t.Errorf("prompt = %q", captured.fields["prompt"])
	}
	if string(captured.file) != "MP3DATA" {
		t.Errorf("uploaded bytes = %q", captured.file)
	}
	// OpenAI validates the declared extension against its supported list.
	if captured.fields["__filename"] != "audio.mp3" {
		t.Errorf("filename = %q, want audio.mp3", captured.fields["__filename"])
	}

	if got.Language != "english" {
		t.Errorf("language = %q", got.Language)
	}
	if got.Duration != 10760*time.Millisecond {
		t.Errorf("duration = %v, want 10.76s", got.Duration)
	}
	if len(got.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(got.Segments))
	}
	if got.Segments[1].Start != 7960*time.Millisecond {
		t.Errorf("segment start = %v, want 7.96s", got.Segments[1].Start)
	}
}

// The gpt-4o transcription models cannot return timings. Asking them for SRT
// has to fail loudly rather than write an empty subtitle file.
func TestTranscribeOpenAI_RejectsTimestampsOnTextOnlyModel(t *testing.T) {
	_, err := transcribeOpenAI(cloudRequest{
		Audio: []byte("x"), Ext: "mp3", APIKey: "k",
		Model: "gpt-4o-transcribe", NeedSegments: true,
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "whisper-1") {
		t.Fatalf("the error should suggest whisper-1, got: %v", err)
	}
}

func TestTranscribeOpenAI_TextOnlyModelUsesPlainJSON(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, `{"text":"hello there"}`)
	defer swapURL(&openAIAPIURL, srv.URL)()

	got, err := transcribeOpenAI(cloudRequest{
		Audio: []byte("x"), Ext: "mp3", APIKey: "k", Model: "gpt-4o-transcribe",
	})
	if err != nil {
		t.Fatalf("transcribeOpenAI: %v", err)
	}
	if captured.fields["response_format"] != "json" {
		t.Errorf("response_format = %q, want json", captured.fields["response_format"])
	}
	if got.Text != "hello there" {
		t.Errorf("text = %q", got.Text)
	}
}

// Translation is a different endpoint that only accepts whisper-1 and takes no
// source-language hint.
func TestTranscribeOpenAI_TranslateUsesTranslationEndpoint(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, `{"text":"translated"}`)
	defer swapURL(&openAITranslate, srv.URL)()

	got, err := transcribeOpenAI(cloudRequest{
		Audio: []byte("x"), Ext: "mp3", APIKey: "k",
		Model: "gpt-4o-transcribe", Language: "de", Translate: true,
	})
	if err != nil {
		t.Fatalf("transcribeOpenAI: %v", err)
	}
	if captured.fields["model"] != "whisper-1" {
		t.Errorf("model = %q, want whisper-1 on the translation endpoint", captured.fields["model"])
	}
	if _, ok := captured.fields["language"]; ok {
		t.Error("the translation endpoint takes no language field")
	}
	if got.Text != "translated" {
		t.Errorf("text = %q", got.Text)
	}
}

func TestTranscribeOpenAI_ReturnsErrorOnNon200(t *testing.T) {
	srv, _ := multipartServer(t, http.StatusUnauthorized, `{"error":{"message":"invalid api key"}}`)
	defer swapURL(&openAIAPIURL, srv.URL)()

	_, err := transcribeOpenAI(cloudRequest{Audio: []byte("x"), Ext: "mp3", APIKey: "bad", Model: "whisper-1"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "invalid api key") {
		t.Fatalf("the provider's message should be quoted, got: %v", err)
	}
}

func TestTranscribeOpenAI_MalformedResponseIsAnError(t *testing.T) {
	srv, _ := multipartServer(t, http.StatusOK, `not json`)
	defer swapURL(&openAIAPIURL, srv.URL)()

	if _, err := transcribeOpenAI(cloudRequest{Audio: []byte("x"), Ext: "mp3", APIKey: "k", Model: "whisper-1"}); err == nil {
		t.Fatal("expected an error")
	}
}

const elevenLabsFixture = `{
  "language_code": "en",
  "language_probability": 0.98,
  "text": "Hello there. General Kenobi.",
  "words": [
    {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
    {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
    {"text": "there.", "start": 0.5, "end": 0.9, "type": "word", "speaker_id": "speaker_0"},
    {"text": "General", "start": 1.0, "end": 1.4, "type": "word", "speaker_id": "speaker_1"},
    {"text": "Kenobi.", "start": 1.4, "end": 1.9, "type": "word", "speaker_id": "speaker_1"}
  ]
}`

func TestTranscribeElevenLabs_SendsExpectedRequest(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, elevenLabsFixture)
	defer swapURL(&elevenLabsAPIURL, srv.URL)()

	got, err := transcribeElevenLabs(cloudRequest{
		Audio: []byte("MP3DATA"), MIME: "audio/mpeg", Ext: "mp3",
		APIKey: "el-test", Model: "scribe_v1", Language: "en", Diarize: true,
	})
	if err != nil {
		t.Fatalf("transcribeElevenLabs: %v", err)
	}

	if key := captured.headers.Get("xi-api-key"); key != "el-test" {
		t.Errorf("xi-api-key = %q", key)
	}
	if captured.headers.Get("Authorization") != "" {
		t.Error("ElevenLabs authenticates with xi-api-key, not Authorization")
	}
	if captured.fields["model_id"] != "scribe_v1" {
		t.Errorf("model_id = %q", captured.fields["model_id"])
	}
	if captured.fields["language_code"] != "en" {
		t.Errorf("language_code = %q", captured.fields["language_code"])
	}
	if captured.fields["diarize"] != "true" {
		t.Errorf("diarize = %q", captured.fields["diarize"])
	}

	if got.Language != "en" {
		t.Errorf("language = %q", got.Language)
	}
	// Words are grouped into segments, and the speaker change forces a break.
	if len(got.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d: %+v", len(got.Segments), got.Segments)
	}
	if got.Segments[0].Text != "Hello there." || got.Segments[0].Speaker != "speaker_0" {
		t.Errorf("unexpected first segment: %+v", got.Segments[0])
	}
	if got.Segments[1].Speaker != "speaker_1" {
		t.Errorf("unexpected second segment: %+v", got.Segments[1])
	}
}

// Spacing entries have no content; folding them in would produce doubled gaps.
func TestTranscribeElevenLabs_SkipsNonWordTokens(t *testing.T) {
	srv, _ := multipartServer(t, http.StatusOK, elevenLabsFixture)
	defer swapURL(&elevenLabsAPIURL, srv.URL)()

	got, err := transcribeElevenLabs(cloudRequest{Audio: []byte("x"), Ext: "mp3", APIKey: "k", Model: "scribe_v1"})
	if err != nil {
		t.Fatalf("transcribeElevenLabs: %v", err)
	}
	for _, s := range got.Segments {
		if strings.Contains(s.Text, "  ") {
			t.Fatalf("spacing tokens leaked into the text: %q", s.Text)
		}
	}
}

func TestTranscribeElevenLabs_ReturnsErrorOnNon200(t *testing.T) {
	srv, _ := multipartServer(t, http.StatusUnprocessableEntity, `{"detail":"invalid model"}`)
	defer swapURL(&elevenLabsAPIURL, srv.URL)()

	_, err := transcribeElevenLabs(cloudRequest{Audio: []byte("x"), Ext: "mp3", APIKey: "k", Model: "bogus"})
	if err == nil || !strings.Contains(err.Error(), "invalid model") {
		t.Fatalf("expected the provider's message, got: %v", err)
	}
}

const deepgramUtterancesFixture = `{
  "metadata": {"duration": 12.5},
  "results": {
    "channels": [{"alternatives": [{"transcript": "Hello there. General Kenobi.", "words": []}]}],
    "utterances": [
      {"start": 0.0, "end": 1.0, "transcript": "Hello there.", "speaker": 0},
      {"start": 1.0, "end": 2.0, "transcript": "General Kenobi.", "speaker": 1}
    ]
  }
}`

func TestTranscribeDeepgram_SendsExpectedRequest(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, deepgramUtterancesFixture)
	defer swapURL(&deepgramAPIURL, srv.URL)()

	got, err := transcribeDeepgram(cloudRequest{
		Audio: []byte("MP3DATA"), MIME: "audio/mpeg", Ext: "mp3",
		APIKey: "dg-test", Model: "nova-3", Language: "en",
		Diarize: true, NeedSegments: true,
	})
	if err != nil {
		t.Fatalf("transcribeDeepgram: %v", err)
	}

	if auth := captured.headers.Get("Authorization"); auth != "Token dg-test" {
		t.Errorf("Authorization = %q, want the Token scheme", auth)
	}
	// Deepgram takes the raw container, not a multipart form.
	if string(captured.rawBody) != "MP3DATA" {
		t.Errorf("body = %q, want the raw audio", captured.rawBody)
	}
	if ct := captured.headers.Get("Content-Type"); ct != "audio/mpeg" {
		t.Errorf("Content-Type = %q, want audio/mpeg", ct)
	}
	for key, want := range map[string]string{
		"model": "nova-3", "language": "en", "diarize": "true",
		"utterances": "true", "smart_format": "true", "punctuate": "true",
	} {
		if got := captured.query.Get(key); got != want {
			t.Errorf("query %s = %q, want %q", key, got, want)
		}
	}

	if got.Duration != 12500*time.Millisecond {
		t.Errorf("duration = %v", got.Duration)
	}
	if len(got.Segments) != 2 {
		t.Fatalf("expected 2 utterance segments, got %d", len(got.Segments))
	}
	if got.Segments[1].Speaker != "speaker_1" {
		t.Errorf("speaker = %q, want speaker_1", got.Segments[1].Speaker)
	}
	if got.Text != "Hello there. General Kenobi." {
		t.Errorf("text = %q", got.Text)
	}
}

// Without utterances, segments have to be rebuilt from word timings.
func TestTranscribeDeepgram_FallsBackToWordTimings(t *testing.T) {
	fixture := `{
      "metadata": {"duration": 2.0},
      "results": {"channels": [{"alternatives": [{
        "transcript": "hello there",
        "words": [
          {"word": "hello", "punctuated_word": "Hello", "start": 0.0, "end": 0.4},
          {"word": "there", "punctuated_word": "there.", "start": 0.4, "end": 0.9}
        ]}]}]}
    }`
	srv, _ := multipartServer(t, http.StatusOK, fixture)
	defer swapURL(&deepgramAPIURL, srv.URL)()

	got, err := transcribeDeepgram(cloudRequest{
		Audio: []byte("x"), MIME: "audio/mpeg", APIKey: "k", Model: "nova-3", NeedSegments: true,
	})
	if err != nil {
		t.Fatalf("transcribeDeepgram: %v", err)
	}
	if len(got.Segments) != 1 {
		t.Fatalf("expected 1 grouped segment, got %d: %+v", len(got.Segments), got.Segments)
	}
	// The punctuated form is preferred over the bare word.
	if got.Segments[0].Text != "Hello there." {
		t.Errorf("text = %q, want the punctuated words", got.Segments[0].Text)
	}
}

// Utterances are only worth requesting when the output needs timings.
func TestTranscribeDeepgram_SkipsUtterancesForPlainText(t *testing.T) {
	srv, captured := multipartServer(t, http.StatusOK, deepgramUtterancesFixture)
	defer swapURL(&deepgramAPIURL, srv.URL)()

	if _, err := transcribeDeepgram(cloudRequest{
		Audio: []byte("x"), MIME: "audio/mpeg", APIKey: "k", Model: "nova-3",
	}); err != nil {
		t.Fatalf("transcribeDeepgram: %v", err)
	}
	if captured.query.Get("utterances") != "" {
		t.Error("utterances should not be requested for plain text output")
	}
}

func TestTranscribeDeepgram_ReturnsErrorOnNon200(t *testing.T) {
	srv, _ := multipartServer(t, http.StatusBadRequest, `{"err_msg":"unsupported model"}`)
	defer swapURL(&deepgramAPIURL, srv.URL)()

	_, err := transcribeDeepgram(cloudRequest{Audio: []byte("x"), APIKey: "k", Model: "bogus"})
	if err == nil || !strings.Contains(err.Error(), "unsupported model") {
		t.Fatalf("expected the provider's message, got: %v", err)
	}
}

func TestSpeakerLabel(t *testing.T) {
	zero := 0
	if got := speakerLabel(&zero, true); got != "speaker_0" {
		t.Errorf("speakerLabel = %q", got)
	}
	// Without --diarize every segment would otherwise be labelled speaker_0.
	if got := speakerLabel(&zero, false); got != "" {
		t.Errorf("speakerLabel without diarization = %q, want empty", got)
	}
	if got := speakerLabel(nil, true); got != "" {
		t.Errorf("speakerLabel(nil) = %q, want empty", got)
	}
}

func TestBuildMultipart(t *testing.T) {
	body, contentType, err := buildMultipart("file", "audio.mp3", []byte("AUDIO"),
		map[string]string{"model": "whisper-1", "response_format": "json"})
	if err != nil {
		t.Fatalf("buildMultipart: %v", err)
	}
	if !strings.HasPrefix(contentType, "multipart/form-data; boundary=") {
		t.Fatalf("content type = %q", contentType)
	}
	text := string(body)
	for _, want := range []string{`name="model"`, "whisper-1", `filename="audio.mp3"`, "AUDIO"} {
		if !strings.Contains(text, want) {
			t.Errorf("body is missing %q", want)
		}
	}
}

func TestSortedKeys(t *testing.T) {
	got := sortedKeys(map[string]string{"c": "", "a": "", "b": ""})
	if strings.Join(got, "") != "abc" {
		t.Fatalf("sortedKeys = %v", got)
	}
}

func TestDoRequest_FailsOnUnreachableHost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close() // nothing is listening now

	req, err := http.NewRequest("POST", url, strings.NewReader("x"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := doRequest(req); err == nil {
		t.Fatal("expected a network error")
	}
}

func TestSecondsToDuration(t *testing.T) {
	if got := secondsToDuration(1.5); got != 1500*time.Millisecond {
		t.Fatalf("secondsToDuration = %v", got)
	}
}
