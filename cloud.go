package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Provider endpoint URLs. Declared as vars so tests can repoint them at an
// httptest server without touching the calling code.
var (
	openAIAPIURL     = "https://api.openai.com/v1/audio/transcriptions"
	openAITranslate  = "https://api.openai.com/v1/audio/translations"
	elevenLabsAPIURL = "https://api.elevenlabs.io/v1/speech-to-text"
	deepgramAPIURL   = "https://api.deepgram.com/v1/listen"
)

// httpClient is shared by the cloud backends. The timeout is generous because
// a chunk can be several minutes of audio.
var httpClient = &http.Client{Timeout: 300 * time.Second}

// cloudRequest is one upload's worth of work.
type cloudRequest struct {
	Audio     []byte
	MIME      string
	Ext       string
	APIKey    string
	Model     string
	Language  string
	Prompt    string
	Translate bool
	Diarize   bool
	// NeedSegments is set when the requested output format cannot be produced
	// from flat text, so a backend can ask for timed output up front.
	NeedSegments bool
}

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------

// openAITimestampModels lists the transcription models that can return
// verbose_json with segment timings. The gpt-4o transcription models answer in
// plain text or flat json only, so asking them for SRT would silently produce
// an empty file.
var openAITimestampModels = map[string]bool{
	"whisper-1": true,
}

// openAIVerboseResponse is the verbose_json schema.
type openAIVerboseResponse struct {
	Text     string  `json:"text"`
	Language string  `json:"language"`
	Duration float64 `json:"duration"`
	Segments []struct {
		Start float64 `json:"start"`
		End   float64 `json:"end"`
		Text  string  `json:"text"`
	} `json:"segments"`
}

func transcribeOpenAI(req cloudRequest) (Transcript, error) {
	supportsTimestamps := openAITimestampModels[req.Model]
	if req.NeedSegments && !supportsTimestamps {
		return Transcript{}, fmt.Errorf(
			"model %q returns text without timestamps; use -m whisper-1 for timed output", req.Model)
	}

	responseFormat := "json"
	if supportsTimestamps {
		responseFormat = "verbose_json"
	}

	fields := map[string]string{
		"model":           req.Model,
		"response_format": responseFormat,
	}
	if req.Language != "" {
		fields["language"] = req.Language
	}
	if req.Prompt != "" {
		fields["prompt"] = req.Prompt
	}

	endpoint := openAIAPIURL
	if req.Translate {
		// Translation into English is a separate endpoint, and it only accepts
		// whisper-1. It takes no language hint — the source language is
		// detected and the output is always English.
		endpoint = openAITranslate
		delete(fields, "language")
		fields["model"] = "whisper-1"
		if req.NeedSegments {
			fields["response_format"] = "verbose_json"
		}
	}

	body, contentType, err := buildMultipart("file", "audio."+req.Ext, req.Audio, fields)
	if err != nil {
		return Transcript{}, err
	}

	httpReq, err := http.NewRequest("POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return Transcript{}, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", contentType)
	httpReq.Header.Set("Authorization", "Bearer "+req.APIKey)

	data, err := doRequest(httpReq)
	if err != nil {
		return Transcript{}, err
	}

	var parsed openAIVerboseResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return Transcript{}, fmt.Errorf("failed to parse OpenAI response: %w", err)
	}

	out := Transcript{
		Text:     strings.TrimSpace(parsed.Text),
		Language: parsed.Language,
		Duration: secondsToDuration(parsed.Duration),
	}
	for _, seg := range parsed.Segments {
		out.Segments = append(out.Segments, Segment{
			Start: secondsToDuration(seg.Start),
			End:   secondsToDuration(seg.End),
			Text:  strings.TrimSpace(seg.Text),
		})
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// ElevenLabs
// ---------------------------------------------------------------------------

// elevenLabsResponse is the single-channel speech-to-text schema. ElevenLabs
// returns word-level timings only; segments are assembled from them.
type elevenLabsResponse struct {
	LanguageCode string `json:"language_code"`
	Text         string `json:"text"`
	Words        []struct {
		Text      string  `json:"text"`
		Start     float64 `json:"start"`
		End       float64 `json:"end"`
		Type      string  `json:"type"`
		SpeakerID string  `json:"speaker_id"`
	} `json:"words"`
}

func transcribeElevenLabs(req cloudRequest) (Transcript, error) {
	fields := map[string]string{
		"model_id": req.Model,
	}
	if req.Language != "" {
		fields["language_code"] = req.Language
	}
	if req.Diarize {
		fields["diarize"] = "true"
	}

	body, contentType, err := buildMultipart("file", "audio."+req.Ext, req.Audio, fields)
	if err != nil {
		return Transcript{}, err
	}

	httpReq, err := http.NewRequest("POST", elevenLabsAPIURL, bytes.NewReader(body))
	if err != nil {
		return Transcript{}, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", contentType)
	httpReq.Header.Set("xi-api-key", req.APIKey)

	data, err := doRequest(httpReq)
	if err != nil {
		return Transcript{}, err
	}

	var parsed elevenLabsResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return Transcript{}, fmt.Errorf("failed to parse ElevenLabs response: %w", err)
	}

	words := make([]Segment, 0, len(parsed.Words))
	for _, w := range parsed.Words {
		// "spacing" entries carry no content, and audio events are annotations
		// rather than speech.
		if w.Type != "" && w.Type != "word" {
			continue
		}
		words = append(words, Segment{
			Start:   secondsToDuration(w.Start),
			End:     secondsToDuration(w.End),
			Text:    w.Text,
			Speaker: w.SpeakerID,
		})
	}

	return Transcript{
		Text:     strings.TrimSpace(parsed.Text),
		Language: parsed.LanguageCode,
		Segments: groupWords(words, 600*time.Millisecond, 12*time.Second, 180),
	}, nil
}

// ---------------------------------------------------------------------------
// Deepgram
// ---------------------------------------------------------------------------

// deepgramResponse covers both the utterance view (preferred, because it is
// already segmented) and the flat channel alternative used as a fallback.
type deepgramResponse struct {
	Results struct {
		Channels []struct {
			Alternatives []struct {
				Transcript string `json:"transcript"`
				Words      []struct {
					Word           string  `json:"word"`
					PunctuatedWord string  `json:"punctuated_word"`
					Start          float64 `json:"start"`
					End            float64 `json:"end"`
					Speaker        *int    `json:"speaker"`
				} `json:"words"`
			} `json:"alternatives"`
		} `json:"channels"`
		Utterances []struct {
			Transcript string  `json:"transcript"`
			Start      float64 `json:"start"`
			End        float64 `json:"end"`
			Speaker    *int    `json:"speaker"`
		} `json:"utterances"`
	} `json:"results"`
	Metadata struct {
		Duration float64 `json:"duration"`
	} `json:"metadata"`
}

func transcribeDeepgram(req cloudRequest) (Transcript, error) {
	query := url.Values{}
	query.Set("model", req.Model)
	query.Set("smart_format", "true")
	query.Set("punctuate", "true")
	if req.Language != "" {
		query.Set("language", req.Language)
	}
	if req.Diarize {
		query.Set("diarize", "true")
	}
	// Utterances give ready-made segments with timings; without them only a
	// single flat transcript comes back.
	if req.NeedSegments || req.Diarize {
		query.Set("utterances", "true")
	}

	endpoint := deepgramAPIURL + "?" + query.Encode()
	httpReq, err := http.NewRequest("POST", endpoint, bytes.NewReader(req.Audio))
	if err != nil {
		return Transcript{}, fmt.Errorf("failed to create request: %w", err)
	}
	// Deepgram takes the raw container bytes rather than a multipart upload.
	httpReq.Header.Set("Content-Type", req.MIME)
	httpReq.Header.Set("Authorization", "Token "+req.APIKey)

	data, err := doRequest(httpReq)
	if err != nil {
		return Transcript{}, err
	}

	var parsed deepgramResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return Transcript{}, fmt.Errorf("failed to parse Deepgram response: %w", err)
	}

	out := Transcript{Duration: secondsToDuration(parsed.Metadata.Duration)}

	for _, u := range parsed.Results.Utterances {
		out.Segments = append(out.Segments, Segment{
			Start:   secondsToDuration(u.Start),
			End:     secondsToDuration(u.End),
			Text:    strings.TrimSpace(u.Transcript),
			Speaker: speakerLabel(u.Speaker, req.Diarize),
		})
	}

	if len(parsed.Results.Channels) > 0 && len(parsed.Results.Channels[0].Alternatives) > 0 {
		alt := parsed.Results.Channels[0].Alternatives[0]
		out.Text = strings.TrimSpace(alt.Transcript)
		if len(out.Segments) == 0 && len(alt.Words) > 0 {
			words := make([]Segment, 0, len(alt.Words))
			for _, w := range alt.Words {
				text := w.PunctuatedWord
				if text == "" {
					text = w.Word
				}
				words = append(words, Segment{
					Start:   secondsToDuration(w.Start),
					End:     secondsToDuration(w.End),
					Text:    text,
					Speaker: speakerLabel(w.Speaker, req.Diarize),
				})
			}
			out.Segments = groupWords(words, 600*time.Millisecond, 12*time.Second, 180)
		}
	}
	if out.Text == "" {
		out.Text = textFromSegments(out.Segments)
	}
	return out, nil
}

// speakerLabel renders Deepgram's numeric speaker index, but only when
// diarization was actually requested — otherwise every segment would be
// labelled speaker_0.
func speakerLabel(id *int, diarize bool) string {
	if !diarize || id == nil {
		return ""
	}
	return "speaker_" + strconv.Itoa(*id)
}

// ---------------------------------------------------------------------------
// Shared HTTP helpers
// ---------------------------------------------------------------------------

// buildMultipart assembles a multipart/form-data body with one file part and
// any number of simple text fields.
func buildMultipart(fileField, filename string, content []byte, fields map[string]string) ([]byte, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	for _, key := range sortedKeys(fields) {
		if err := w.WriteField(key, fields[key]); err != nil {
			return nil, "", fmt.Errorf("failed to write form field %s: %w", key, err)
		}
	}
	part, err := w.CreateFormFile(fileField, filename)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(content); err != nil {
		return nil, "", fmt.Errorf("failed to write audio to form: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, "", fmt.Errorf("failed to finalise form: %w", err)
	}
	return buf.Bytes(), w.FormDataContentType(), nil
}

// sortedKeys keeps multipart field order deterministic so tests can assert on
// the encoded body.
func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

// doRequest sends the request and returns the body, turning any non-200 into an
// error that quotes the provider's own message.
func doRequest(req *http.Request) ([]byte, error) {
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (%d): %s", resp.StatusCode, truncate(string(body), 500))
	}
	if readErr != nil {
		return nil, fmt.Errorf("failed to read response: %w", readErr)
	}
	return body, nil
}

func secondsToDuration(s float64) time.Duration {
	return time.Duration(s * float64(time.Second))
}
