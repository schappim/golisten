package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

// Segment is one timed span of transcribed speech. Speaker is empty unless the
// backend was asked for diarization and supports it.
type Segment struct {
	Start   time.Duration
	End     time.Duration
	Text    string
	Speaker string
}

// Transcript is the backend-neutral result every provider is normalised into.
// Segments may be empty when a backend only returns flat text; Text is always
// populated.
type Transcript struct {
	Text     string
	Language string
	Duration time.Duration
	Segments []Segment
}

// textFromSegments joins segment text into a single paragraph. Used when a
// backend gives us segments but no pre-joined transcript.
func textFromSegments(segs []Segment) string {
	parts := make([]string, 0, len(segs))
	for _, s := range segs {
		if t := strings.TrimSpace(s.Text); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, " ")
}

// shift returns a copy of the transcript with every segment moved later by off.
// Used when stitching the results of separately-transcribed audio chunks back
// onto a single timeline.
func (t Transcript) shift(off time.Duration) Transcript {
	out := t
	out.Segments = make([]Segment, len(t.Segments))
	for i, s := range t.Segments {
		s.Start += off
		s.End += off
		out.Segments[i] = s
	}
	return out
}

// mergeTranscripts stitches per-chunk transcripts into one. Each part is
// expected to have already been shifted onto the global timeline. The language
// of the first part that reports one wins.
func mergeTranscripts(parts []Transcript) Transcript {
	var out Transcript
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if out.Language == "" {
			out.Language = p.Language
		}
		if t := strings.TrimSpace(p.Text); t != "" {
			texts = append(texts, t)
		}
		out.Segments = append(out.Segments, p.Segments...)
		if p.Duration > out.Duration {
			out.Duration = p.Duration
		}
	}
	out.Text = strings.Join(texts, " ")
	sort.SliceStable(out.Segments, func(i, j int) bool {
		return out.Segments[i].Start < out.Segments[j].Start
	})
	return out
}

// formatTimestamp renders d as HH:MM:SS<sep>mmm. SRT uses a comma separator,
// WebVTT a period. Negative durations clamp to zero so a backend reporting a
// slightly negative start can never emit an unparseable cue.
func formatTimestamp(d time.Duration, sep string) string {
	if d < 0 {
		d = 0
	}
	ms := d.Milliseconds()
	h := ms / 3_600_000
	ms -= h * 3_600_000
	m := ms / 60_000
	ms -= m * 60_000
	s := ms / 1000
	ms -= s * 1000
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", h, m, s, sep, ms)
}

// renderTimestamps is the whisper.cpp-style "[start --> end]  text" line
// format, used for -f txt when --timestamps is set.
func renderTimestamps(t Transcript) string {
	var b strings.Builder
	for _, s := range t.Segments {
		fmt.Fprintf(&b, "[%s --> %s]  %s%s\n",
			formatTimestamp(s.Start, "."), formatTimestamp(s.End, "."),
			speakerPrefix(s), strings.TrimSpace(s.Text))
	}
	return b.String()
}

func speakerPrefix(s Segment) string {
	if s.Speaker == "" {
		return ""
	}
	return s.Speaker + ": "
}

func renderText(t Transcript) string {
	body := strings.TrimSpace(t.Text)
	if body == "" {
		return ""
	}
	return body + "\n"
}

func renderSRT(t Transcript) string {
	var b strings.Builder
	n := 0
	for _, s := range t.Segments {
		text := strings.TrimSpace(s.Text)
		if text == "" {
			continue
		}
		n++
		fmt.Fprintf(&b, "%d\n%s --> %s\n%s%s\n\n", n,
			formatTimestamp(s.Start, ","), formatTimestamp(s.End, ","),
			speakerPrefix(s), text)
	}
	return b.String()
}

func renderVTT(t Transcript) string {
	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for _, s := range t.Segments {
		text := strings.TrimSpace(s.Text)
		if text == "" {
			continue
		}
		fmt.Fprintf(&b, "%s --> %s\n%s%s\n\n",
			formatTimestamp(s.Start, "."), formatTimestamp(s.End, "."),
			speakerPrefix(s), text)
	}
	return b.String()
}

// jsonSegment / jsonTranscript are the wire shape of -f json. Kept as explicit
// structs (rather than marshalling Transcript directly) so time.Duration is
// emitted as plain milliseconds and the field names stay stable.
type jsonSegment struct {
	StartMS int64  `json:"start_ms"`
	EndMS   int64  `json:"end_ms"`
	Text    string `json:"text"`
	Speaker string `json:"speaker,omitempty"`
}

type jsonTranscript struct {
	Language   string        `json:"language,omitempty"`
	DurationMS int64         `json:"duration_ms,omitempty"`
	Text       string        `json:"text"`
	Segments   []jsonSegment `json:"segments"`
}

func renderJSON(t Transcript) (string, error) {
	out := jsonTranscript{
		Language:   t.Language,
		DurationMS: t.Duration.Milliseconds(),
		Text:       strings.TrimSpace(t.Text),
		Segments:   make([]jsonSegment, 0, len(t.Segments)),
	}
	for _, s := range t.Segments {
		out.Segments = append(out.Segments, jsonSegment{
			StartMS: s.Start.Milliseconds(),
			EndMS:   s.End.Milliseconds(),
			Text:    strings.TrimSpace(s.Text),
			Speaker: s.Speaker,
		})
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data) + "\n", nil
}

// render turns a transcript into the requested output format. showTimestamps
// only affects the txt format; the other formats are timed by definition.
func render(t Transcript, format string, showTimestamps bool) (string, error) {
	switch format {
	case "txt":
		if showTimestamps {
			return renderTimestamps(t), nil
		}
		return renderText(t), nil
	case "srt":
		return renderSRT(t), nil
	case "vtt":
		return renderVTT(t), nil
	case "json":
		return renderJSON(t)
	default:
		return "", fmt.Errorf("unknown output format %q", format)
	}
}

// formatNeedsSegments reports whether the requested output can't be produced
// from flat text alone. Used to fail early on backend/format combinations that
// would otherwise silently emit an empty file.
func formatNeedsSegments(format string, showTimestamps bool) bool {
	switch format {
	case "srt", "vtt":
		return true
	case "txt":
		return showTimestamps
	}
	return false
}

// groupWords folds word-level timings (ElevenLabs, and Deepgram when no
// utterances come back) into readable segments. It breaks on speaker change, a
// pause longer than maxGap, sentence-final punctuation, or when a segment grows
// past maxChars / maxDur — all structural signals, no vocabulary involved.
func groupWords(words []Segment, maxGap, maxDur time.Duration, maxChars int) []Segment {
	var out []Segment
	var cur *Segment

	flush := func() {
		if cur != nil {
			cur.Text = strings.TrimSpace(cur.Text)
			if cur.Text != "" {
				out = append(out, *cur)
			}
			cur = nil
		}
	}

	for _, w := range words {
		text := strings.TrimSpace(w.Text)
		if text == "" {
			continue
		}
		if cur != nil {
			tooLong := len(cur.Text)+1+len(text) > maxChars
			tooSlow := w.Start-cur.End > maxGap
			tooWide := w.End-cur.Start > maxDur
			newSpeaker := w.Speaker != cur.Speaker
			if tooLong || tooSlow || tooWide || newSpeaker {
				flush()
			}
		}
		if cur == nil {
			cur = &Segment{Start: w.Start, End: w.End, Text: text, Speaker: w.Speaker}
		} else {
			cur.Text += " " + text
			cur.End = w.End
		}
		if endsSentence(text) {
			flush()
		}
	}
	flush()
	return out
}

// endsSentence reports whether a token closes a sentence. This is typography,
// not vocabulary: it looks only at terminal punctuation, optionally wrapped in
// a closing quote or bracket.
func endsSentence(word string) bool {
	trimmed := strings.TrimRight(word, `"'”’)]}`)
	if trimmed == "" {
		return false
	}
	last, _ := utf8.DecodeLastRuneInString(trimmed)
	switch last {
	case '.', '!', '?', '。', '！', '？':
		return true
	}
	return false
}
