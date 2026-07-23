package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func ms(n int) time.Duration { return time.Duration(n) * time.Millisecond }

func sampleTranscript() Transcript {
	return Transcript{
		Text:     "Hello there. General Kenobi.",
		Language: "en",
		Duration: ms(4000),
		Segments: []Segment{
			{Start: ms(0), End: ms(1500), Text: "Hello there."},
			{Start: ms(1500), End: ms(4000), Text: "General Kenobi."},
		},
	}
}

func TestFormatTimestamp(t *testing.T) {
	cases := []struct {
		d    time.Duration
		sep  string
		want string
	}{
		{0, ",", "00:00:00,000"},
		{ms(1500), ",", "00:00:01,500"},
		{ms(61234), ".", "00:01:01.234"},
		{3*time.Hour + 25*time.Minute + 45*time.Second + ms(6), ".", "03:25:45.006"},
		{-ms(500), ",", "00:00:00,000"}, // negatives clamp rather than emit an unparseable cue
	}
	for _, tc := range cases {
		if got := formatTimestamp(tc.d, tc.sep); got != tc.want {
			t.Errorf("formatTimestamp(%v) = %q, want %q", tc.d, got, tc.want)
		}
	}
}

func TestRenderText(t *testing.T) {
	got, err := render(sampleTranscript(), "txt", false)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if got != "Hello there. General Kenobi.\n" {
		t.Fatalf("render = %q", got)
	}
}

func TestRenderText_EmptyTranscriptIsEmpty(t *testing.T) {
	got, err := render(Transcript{}, "txt", false)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if got != "" {
		t.Fatalf("render = %q, want empty", got)
	}
}

func TestRenderTimestamps(t *testing.T) {
	got, err := render(sampleTranscript(), "txt", true)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	want := "[00:00:00.000 --> 00:00:01.500]  Hello there.\n" +
		"[00:00:01.500 --> 00:00:04.000]  General Kenobi.\n"
	if got != want {
		t.Fatalf("render = %q, want %q", got, want)
	}
}

func TestRenderSRT(t *testing.T) {
	got, err := render(sampleTranscript(), "srt", false)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	want := "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n" +
		"2\n00:00:01,500 --> 00:00:04,000\nGeneral Kenobi.\n\n"
	if got != want {
		t.Fatalf("render = %q, want %q", got, want)
	}
}

// Blank segments must not consume a cue number, or the SRT ends up with gaps in
// its numbering and some players stop rendering.
func TestRenderSRT_SkipsEmptySegmentsAndRenumbers(t *testing.T) {
	tr := Transcript{Segments: []Segment{
		{Start: ms(0), End: ms(1000), Text: "  "},
		{Start: ms(1000), End: ms(2000), Text: "first"},
		{Start: ms(2000), End: ms(3000), Text: ""},
		{Start: ms(3000), End: ms(4000), Text: "second"},
	}}
	got := renderSRT(tr)
	if strings.Contains(got, "3\n") {
		t.Fatalf("empty segments were numbered:\n%s", got)
	}
	if !strings.HasPrefix(got, "1\n00:00:01,000 --> 00:00:02,000\nfirst\n") {
		t.Fatalf("unexpected first cue:\n%s", got)
	}
	if !strings.Contains(got, "2\n00:00:03,000 --> 00:00:04,000\nsecond\n") {
		t.Fatalf("second cue was not renumbered:\n%s", got)
	}
}

func TestRenderVTT(t *testing.T) {
	got, err := render(sampleTranscript(), "vtt", false)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if !strings.HasPrefix(got, "WEBVTT\n\n") {
		t.Fatalf("VTT must start with the WEBVTT header, got %q", got)
	}
	if !strings.Contains(got, "00:00:00.000 --> 00:00:01.500\nHello there.") {
		t.Fatalf("unexpected VTT body:\n%s", got)
	}
	if strings.Contains(got, ",") {
		t.Fatal("VTT timestamps must use a period, not a comma")
	}
}

func TestRenderJSON(t *testing.T) {
	got, err := render(sampleTranscript(), "json", false)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	var parsed jsonTranscript
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	if parsed.Language != "en" || parsed.DurationMS != 4000 {
		t.Fatalf("unexpected header fields: %+v", parsed)
	}
	if len(parsed.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(parsed.Segments))
	}
	if parsed.Segments[1].StartMS != 1500 || parsed.Segments[1].EndMS != 4000 {
		t.Fatalf("unexpected segment timings: %+v", parsed.Segments[1])
	}
}

// An empty transcript still has to produce a segments array rather than null,
// so consumers can iterate it unconditionally.
func TestRenderJSON_EmptySegmentsIsAnArray(t *testing.T) {
	got, err := render(Transcript{Text: "hi"}, "json", false)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if !strings.Contains(got, `"segments": []`) {
		t.Fatalf("expected an empty array, got:\n%s", got)
	}
}

func TestRenderSpeakerLabels(t *testing.T) {
	tr := Transcript{Segments: []Segment{
		{Start: 0, End: ms(1000), Text: "hello", Speaker: "speaker_0"},
	}}
	if got := renderSRT(tr); !strings.Contains(got, "speaker_0: hello") {
		t.Fatalf("SRT missing speaker label:\n%s", got)
	}
	if got := renderVTT(tr); !strings.Contains(got, "speaker_0: hello") {
		t.Fatalf("VTT missing speaker label:\n%s", got)
	}
	if got := renderTimestamps(tr); !strings.Contains(got, "speaker_0: hello") {
		t.Fatalf("timestamped text missing speaker label:\n%s", got)
	}
}

func TestRender_UnknownFormatIsAnError(t *testing.T) {
	if _, err := render(sampleTranscript(), "docx", false); err == nil {
		t.Fatal("expected an error for an unknown format")
	}
}

func TestFormatNeedsSegments(t *testing.T) {
	cases := []struct {
		format     string
		timestamps bool
		want       bool
	}{
		{"txt", false, false},
		{"txt", true, true},
		{"srt", false, true},
		{"vtt", false, true},
		{"json", false, false},
	}
	for _, tc := range cases {
		if got := formatNeedsSegments(tc.format, tc.timestamps); got != tc.want {
			t.Errorf("formatNeedsSegments(%q, %v) = %v, want %v",
				tc.format, tc.timestamps, got, tc.want)
		}
	}
}

func TestTextFromSegments(t *testing.T) {
	got := textFromSegments([]Segment{
		{Text: " one "}, {Text: ""}, {Text: "two"},
	})
	if got != "one two" {
		t.Fatalf("textFromSegments = %q, want %q", got, "one two")
	}
}

func TestShift(t *testing.T) {
	original := sampleTranscript()
	shifted := original.shift(10 * time.Second)

	if shifted.Segments[0].Start != 10*time.Second {
		t.Fatalf("start = %v, want 10s", shifted.Segments[0].Start)
	}
	if shifted.Segments[1].End != 14*time.Second {
		t.Fatalf("end = %v, want 14s", shifted.Segments[1].End)
	}
	if original.Segments[0].Start != 0 {
		t.Fatal("shift mutated the original transcript")
	}
}

func TestMergeTranscripts(t *testing.T) {
	parts := []Transcript{
		{Text: "first part", Language: "en", Segments: []Segment{{Start: 0, End: ms(1000), Text: "first part"}}},
		{Text: "second part", Segments: []Segment{{Start: ms(2000), End: ms(3000), Text: "second part"}}},
	}
	merged := mergeTranscripts(parts)

	if merged.Text != "first part second part" {
		t.Fatalf("text = %q", merged.Text)
	}
	if merged.Language != "en" {
		t.Fatalf("language = %q, want en", merged.Language)
	}
	if len(merged.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(merged.Segments))
	}
	if merged.Segments[0].Start > merged.Segments[1].Start {
		t.Fatal("segments are not in time order")
	}
}

func TestMergeTranscripts_SortsOutOfOrderParts(t *testing.T) {
	merged := mergeTranscripts([]Transcript{
		{Segments: []Segment{{Start: ms(5000), Text: "late"}}},
		{Segments: []Segment{{Start: ms(1000), Text: "early"}}},
	})
	if merged.Segments[0].Text != "early" {
		t.Fatalf("segments were not sorted: %+v", merged.Segments)
	}
}

func TestMergeTranscripts_SkipsEmptyText(t *testing.T) {
	merged := mergeTranscripts([]Transcript{
		{Text: "one"}, {Text: "   "}, {Text: "two"},
	})
	if merged.Text != "one two" {
		t.Fatalf("text = %q, want %q", merged.Text, "one two")
	}
}

func TestEndsSentence(t *testing.T) {
	cases := map[string]bool{
		"end.":     true,
		"what?":    true,
		"stop!":    true,
		`"quote."`: true,
		"(aside.)": true,
		"middle":   false,
		"":         false,
		"...":      true,
		"終わり。":     true, // CJK full stop is a multi-byte rune
		`")"`:      false,
	}
	for input, want := range cases {
		if got := endsSentence(input); got != want {
			t.Errorf("endsSentence(%q) = %v, want %v", input, got, want)
		}
	}
}

func words(specs ...Segment) []Segment { return specs }

func TestGroupWords_BreaksOnSentenceEnd(t *testing.T) {
	got := groupWords(words(
		Segment{Start: ms(0), End: ms(500), Text: "Hello"},
		Segment{Start: ms(500), End: ms(1000), Text: "there."},
		Segment{Start: ms(1000), End: ms(1500), Text: "General"},
		Segment{Start: ms(1500), End: ms(2000), Text: "Kenobi."},
	), time.Second, time.Minute, 500)

	if len(got) != 2 {
		t.Fatalf("expected 2 segments, got %d: %+v", len(got), got)
	}
	if got[0].Text != "Hello there." || got[1].Text != "General Kenobi." {
		t.Fatalf("unexpected grouping: %+v", got)
	}
	if got[0].Start != 0 || got[0].End != ms(1000) {
		t.Fatalf("unexpected timings on the first segment: %+v", got[0])
	}
}

func TestGroupWords_BreaksOnLongPause(t *testing.T) {
	got := groupWords(words(
		Segment{Start: ms(0), End: ms(500), Text: "before"},
		Segment{Start: ms(5000), End: ms(5500), Text: "after"},
	), time.Second, time.Minute, 500)

	if len(got) != 2 {
		t.Fatalf("expected a break at the pause, got %d segments: %+v", len(got), got)
	}
}

func TestGroupWords_BreaksOnSpeakerChange(t *testing.T) {
	got := groupWords(words(
		Segment{Start: ms(0), End: ms(500), Text: "mine", Speaker: "speaker_0"},
		Segment{Start: ms(500), End: ms(1000), Text: "yours", Speaker: "speaker_1"},
	), time.Second, time.Minute, 500)

	if len(got) != 2 {
		t.Fatalf("expected a break at the speaker change, got %d: %+v", len(got), got)
	}
	if got[0].Speaker != "speaker_0" || got[1].Speaker != "speaker_1" {
		t.Fatalf("speaker labels were lost: %+v", got)
	}
}

func TestGroupWords_BreaksOnLength(t *testing.T) {
	var in []Segment
	for i := 0; i < 40; i++ {
		in = append(in, Segment{
			Start: ms(i * 100), End: ms(i*100 + 100), Text: "word",
		})
	}
	got := groupWords(in, time.Second, time.Minute, 40)
	if len(got) < 2 {
		t.Fatalf("expected several segments, got %d", len(got))
	}
	for _, s := range got {
		if len(s.Text) > 40 {
			t.Fatalf("segment exceeds the character budget: %q", s.Text)
		}
	}
}

func TestGroupWords_BreaksOnDuration(t *testing.T) {
	var in []Segment
	for i := 0; i < 20; i++ {
		in = append(in, Segment{
			Start: ms(i * 1000), End: ms(i*1000 + 900), Text: "word",
		})
	}
	got := groupWords(in, 2*time.Second, 5*time.Second, 500)
	if len(got) < 3 {
		t.Fatalf("expected several segments, got %d", len(got))
	}
	for _, s := range got {
		if s.End-s.Start > 6*time.Second {
			t.Fatalf("segment spans %v, over the budget", s.End-s.Start)
		}
	}
}

func TestGroupWords_IgnoresBlankTokens(t *testing.T) {
	got := groupWords(words(
		Segment{Start: ms(0), End: ms(100), Text: "  "},
		Segment{Start: ms(100), End: ms(200), Text: "real"},
	), time.Second, time.Minute, 500)

	if len(got) != 1 || got[0].Text != "real" {
		t.Fatalf("unexpected grouping: %+v", got)
	}
}

func TestGroupWords_EmptyInput(t *testing.T) {
	if got := groupWords(nil, time.Second, time.Minute, 100); got != nil {
		t.Fatalf("expected nil, got %+v", got)
	}
}
