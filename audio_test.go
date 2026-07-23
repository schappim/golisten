package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// buildWAV assembles a WAV file for tests from raw sample bytes.
func buildWAV(formatTag, channels, rate, bits int, data []byte, extraChunks ...[]byte) []byte {
	var fmtChunk bytes.Buffer
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(formatTag))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(channels))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(rate))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(rate*channels*bits/8))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(channels*bits/8))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(bits))

	var body bytes.Buffer
	body.WriteString("fmt ")
	binary.Write(&body, binary.LittleEndian, uint32(fmtChunk.Len()))
	body.Write(fmtChunk.Bytes())
	for _, c := range extraChunks {
		body.Write(c)
	}
	body.WriteString("data")
	binary.Write(&body, binary.LittleEndian, uint32(len(data)))
	body.Write(data)

	var out bytes.Buffer
	out.WriteString("RIFF")
	binary.Write(&out, binary.LittleEndian, uint32(4+body.Len()))
	out.WriteString("WAVE")
	out.Write(body.Bytes())
	return out.Bytes()
}

func int16Bytes(samples ...int16) []byte {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(s))
	}
	return buf
}

func TestDetectFormat(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want audioFormat
	}{
		{"wav", buildWAV(wavFormatPCM, 1, 16000, 16, int16Bytes(0)), formatWAV},
		{"mp3 with ID3", append([]byte("ID3\x04\x00\x00\x00\x00\x00"), make([]byte, 8)...), formatMP3},
		{"mp3 raw frame sync", append([]byte{0xFF, 0xFB, 0x90, 0x00}, make([]byte, 8)...), formatMP3},
		{"flac", append([]byte("fLaC"), make([]byte, 8)...), formatFLAC},
		{"ogg", append([]byte("OggS"), make([]byte, 8)...), formatOgg},
		{"mp4", append([]byte("\x00\x00\x00\x20ftypM4A "), make([]byte, 4)...), formatMP4},
		{"unknown", []byte("this is plain text and not audio"), formatUnknown},
		{"too short", []byte("RIF"), formatUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := detectFormat(tc.data); got != tc.want {
				t.Fatalf("detectFormat = %q, want %q", got, tc.want)
			}
		})
	}
}

// A RIFF header alone must not be mistaken for a WAVE file — RIFF also wraps
// AVI and WebP.
func TestDetectFormat_RIFFWithoutWAVEIsNotWAV(t *testing.T) {
	data := append([]byte("RIFF\x00\x00\x00\x00AVI "), make([]byte, 8)...)
	if got := detectFormat(data); got == formatWAV {
		t.Fatal("a RIFF/AVI header was reported as WAV")
	}
}

func TestMimeAndExtForFormat(t *testing.T) {
	cases := []struct {
		format    audioFormat
		mime, ext string
	}{
		{formatWAV, "audio/wav", "wav"},
		{formatMP3, "audio/mpeg", "mp3"},
		{formatFLAC, "audio/flac", "flac"},
		{formatOgg, "audio/ogg", "ogg"},
		{formatMP4, "audio/mp4", "m4a"},
		{formatUnknown, "application/octet-stream", "bin"},
	}
	for _, tc := range cases {
		if got := mimeForFormat(tc.format); got != tc.mime {
			t.Errorf("mimeForFormat(%q) = %q, want %q", tc.format, got, tc.mime)
		}
		if got := fileExtForFormat(tc.format); got != tc.ext {
			t.Errorf("fileExtForFormat(%q) = %q, want %q", tc.format, got, tc.ext)
		}
	}
}

func TestDecodeWAV_16BitMono(t *testing.T) {
	data := buildWAV(wavFormatPCM, 1, 16000, 16, int16Bytes(0, 16384, -16384, 32767))
	audio, err := decodeWAV(data)
	if err != nil {
		t.Fatalf("decodeWAV: %v", err)
	}
	if audio.Rate != 16000 {
		t.Fatalf("rate = %d, want 16000", audio.Rate)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768}
	for i, w := range want {
		if math.Abs(float64(audio.Samples[i]-w)) > 1e-6 {
			t.Errorf("sample %d = %v, want %v", i, audio.Samples[i], w)
		}
	}
}

func TestDecodeWAV_StereoIsDownmixed(t *testing.T) {
	// Left and right are opposite, so the mono mix must be silence.
	data := buildWAV(wavFormatPCM, 2, 44100, 16, int16Bytes(16384, -16384, 8192, -8192))
	audio, err := decodeWAV(data)
	if err != nil {
		t.Fatalf("decodeWAV: %v", err)
	}
	if len(audio.Samples) != 2 {
		t.Fatalf("expected 2 frames, got %d", len(audio.Samples))
	}
	for i, s := range audio.Samples {
		if math.Abs(float64(s)) > 1e-6 {
			t.Errorf("frame %d = %v, want 0 after downmix", i, s)
		}
	}
}

func TestDecodeWAV_SampleWidths(t *testing.T) {
	cases := []struct {
		name      string
		formatTag int
		bits      int
		data      []byte
		want      float32
	}{
		{"8-bit unsigned midpoint", wavFormatPCM, 8, []byte{128}, 0},
		{"8-bit unsigned max", wavFormatPCM, 8, []byte{255}, 127.0 / 128},
		{"16-bit", wavFormatPCM, 16, int16Bytes(-32768), -1},
		{"24-bit positive", wavFormatPCM, 24, []byte{0x00, 0x00, 0x40}, 0.5},
		{"24-bit negative", wavFormatPCM, 24, []byte{0x00, 0x00, 0xC0}, -0.5},
		{"32-bit int", wavFormatPCM, 32, []byte{0x00, 0x00, 0x00, 0x40}, 0.5},
		{"32-bit float", wavFormatIEEEFloat, 32, f32Bytes(0.25), 0.25},
		{"64-bit float", wavFormatIEEEFloat, 64, f64Bytes(-0.75), -0.75},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			audio, err := decodeWAV(buildWAV(tc.formatTag, 1, 16000, tc.bits, tc.data))
			if err != nil {
				t.Fatalf("decodeWAV: %v", err)
			}
			if len(audio.Samples) != 1 {
				t.Fatalf("expected 1 sample, got %d", len(audio.Samples))
			}
			if math.Abs(float64(audio.Samples[0]-tc.want)) > 1e-6 {
				t.Fatalf("sample = %v, want %v", audio.Samples[0], tc.want)
			}
		})
	}
}

func f32Bytes(v float32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	return b
}

func f64Bytes(v float64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, math.Float64bits(v))
	return b
}

// WAVE_FORMAT_EXTENSIBLE hides the real codec in the subformat GUID; a decoder
// that only reads the format tag would reject these files.
func TestDecodeWAV_ExtensibleResolvesSubformat(t *testing.T) {
	var ext bytes.Buffer
	binary.Write(&ext, binary.LittleEndian, uint16(wavFormatExtensible))
	binary.Write(&ext, binary.LittleEndian, uint16(1))     // channels
	binary.Write(&ext, binary.LittleEndian, uint32(16000)) // rate
	binary.Write(&ext, binary.LittleEndian, uint32(32000)) // byte rate
	binary.Write(&ext, binary.LittleEndian, uint16(2))     // block align
	binary.Write(&ext, binary.LittleEndian, uint16(16))    // bits
	binary.Write(&ext, binary.LittleEndian, uint16(22))    // cbSize
	binary.Write(&ext, binary.LittleEndian, uint16(16))    // valid bits
	binary.Write(&ext, binary.LittleEndian, uint32(4))     // channel mask
	binary.Write(&ext, binary.LittleEndian, uint16(wavFormatPCM))
	ext.Write(make([]byte, 14)) // rest of the GUID

	var body bytes.Buffer
	body.WriteString("fmt ")
	binary.Write(&body, binary.LittleEndian, uint32(ext.Len()))
	body.Write(ext.Bytes())
	body.WriteString("data")
	payload := int16Bytes(16384)
	binary.Write(&body, binary.LittleEndian, uint32(len(payload)))
	body.Write(payload)

	var out bytes.Buffer
	out.WriteString("RIFF")
	binary.Write(&out, binary.LittleEndian, uint32(4+body.Len()))
	out.WriteString("WAVE")
	out.Write(body.Bytes())

	audio, err := decodeWAV(out.Bytes())
	if err != nil {
		t.Fatalf("decodeWAV: %v", err)
	}
	if math.Abs(float64(audio.Samples[0]-0.5)) > 1e-6 {
		t.Fatalf("sample = %v, want 0.5", audio.Samples[0])
	}
}

// Recorders routinely put LIST/INFO chunks between fmt and data.
func TestDecodeWAV_SkipsUnknownChunks(t *testing.T) {
	var list bytes.Buffer
	list.WriteString("LIST")
	binary.Write(&list, binary.LittleEndian, uint32(4))
	list.WriteString("INFO")

	data := buildWAV(wavFormatPCM, 1, 16000, 16, int16Bytes(16384), list.Bytes())
	audio, err := decodeWAV(data)
	if err != nil {
		t.Fatalf("decodeWAV: %v", err)
	}
	if len(audio.Samples) != 1 || math.Abs(float64(audio.Samples[0]-0.5)) > 1e-6 {
		t.Fatalf("samples = %v, want [0.5]", audio.Samples)
	}
}

// RIFF chunks are word-aligned: an odd-sized chunk is followed by a pad byte
// that is not part of any chunk. Missing that shifts every later chunk by one.
func TestDecodeWAV_HandlesOddSizedChunkPadding(t *testing.T) {
	var odd bytes.Buffer
	odd.WriteString("note")
	binary.Write(&odd, binary.LittleEndian, uint32(3))
	odd.WriteString("abc")
	odd.WriteByte(0) // pad byte

	data := buildWAV(wavFormatPCM, 1, 16000, 16, int16Bytes(-16384), odd.Bytes())
	audio, err := decodeWAV(data)
	if err != nil {
		t.Fatalf("decodeWAV: %v", err)
	}
	if len(audio.Samples) != 1 || math.Abs(float64(audio.Samples[0]+0.5)) > 1e-6 {
		t.Fatalf("samples = %v, want [-0.5]", audio.Samples)
	}
}

func TestDecodeWAV_Rejections(t *testing.T) {
	cases := []struct {
		name string
		data []byte
	}{
		{"not riff", []byte("just some bytes here at all")},
		{"no fmt chunk", func() []byte {
			var out bytes.Buffer
			out.WriteString("RIFF")
			binary.Write(&out, binary.LittleEndian, uint32(4))
			out.WriteString("WAVE")
			return out.Bytes()
		}()},
		{"unsupported bit depth", buildWAV(wavFormatPCM, 1, 16000, 12, []byte{1, 2})},
		{"unsupported codec", buildWAV(0x0055, 1, 16000, 16, int16Bytes(1))},
		{"zero channels", buildWAV(wavFormatPCM, 0, 16000, 16, int16Bytes(1))},
		{"zero sample rate", buildWAV(wavFormatPCM, 1, 0, 16, int16Bytes(1))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := decodeWAV(tc.data); err == nil {
				t.Fatal("expected an error, got nil")
			}
		})
	}
}

func TestDecodeWAV_NoDataChunkIsRejected(t *testing.T) {
	var fmtChunk bytes.Buffer
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(wavFormatPCM))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(1))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(16000))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(32000))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(2))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(16))

	var body bytes.Buffer
	body.WriteString("fmt ")
	binary.Write(&body, binary.LittleEndian, uint32(fmtChunk.Len()))
	body.Write(fmtChunk.Bytes())

	var out bytes.Buffer
	out.WriteString("RIFF")
	binary.Write(&out, binary.LittleEndian, uint32(4+body.Len()))
	out.WriteString("WAVE")
	out.Write(body.Bytes())

	if _, err := decodeWAV(out.Bytes()); err == nil {
		t.Fatal("expected an error for a WAV with no data chunk")
	}
}

func TestEncodeWAV_HeaderIsWellFormed(t *testing.T) {
	out := EncodeWAV(Audio{Samples: []float32{0, 0.5}, Rate: 16000})
	if len(out) != 44+4 {
		t.Fatalf("length = %d, want 48", len(out))
	}
	if string(out[0:4]) != "RIFF" || string(out[8:12]) != "WAVE" {
		t.Fatal("missing RIFF/WAVE markers")
	}
	if got := binary.LittleEndian.Uint32(out[4:]); got != uint32(len(out)-8) {
		t.Errorf("RIFF size = %d, want %d", got, len(out)-8)
	}
	if got := binary.LittleEndian.Uint32(out[24:]); got != 16000 {
		t.Errorf("sample rate = %d, want 16000", got)
	}
	if got := binary.LittleEndian.Uint16(out[22:]); got != 1 {
		t.Errorf("channels = %d, want 1", got)
	}
	if got := binary.LittleEndian.Uint32(out[40:]); got != 4 {
		t.Errorf("data size = %d, want 4", got)
	}
}

// Decoding then re-encoding 16-bit PCM must return the original bytes exactly.
// Scaling by 32767 on the way out instead of 32768 quietly shifts every sample
// by up to an LSB, which is enough to change a transcript.
func TestEncodeWAV_RoundTripIsBitExact(t *testing.T) {
	samples := make([]int16, 0, 1024)
	for i := -512; i < 512; i++ {
		samples = append(samples, int16(i*64))
	}
	samples = append(samples, -32768, 32767, 0, 1, -1)

	original := buildWAV(wavFormatPCM, 1, 16000, 16, int16Bytes(samples...))
	decoded, err := decodeWAV(original)
	if err != nil {
		t.Fatalf("decodeWAV: %v", err)
	}
	encoded := EncodeWAV(decoded)

	if !bytes.Equal(encoded[44:], original[len(original)-len(samples)*2:]) {
		t.Fatal("16-bit PCM did not survive a decode/encode round trip unchanged")
	}
}

func TestEncodeWAV_SaturatesInsteadOfWrapping(t *testing.T) {
	out := EncodeWAV(Audio{Samples: []float32{4, -4}, Rate: 16000})
	hi := int16(binary.LittleEndian.Uint16(out[44:]))
	lo := int16(binary.LittleEndian.Uint16(out[46:]))
	if hi != 32767 {
		t.Errorf("positive overload = %d, want 32767", hi)
	}
	if lo != -32768 {
		t.Errorf("negative overload = %d, want -32768", lo)
	}
}

func TestResample_SameRateIsAPassthrough(t *testing.T) {
	in := Audio{Samples: []float32{0.1, 0.2, 0.3}, Rate: 16000}
	out, err := Resample(in, 16000)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	if &out.Samples[0] != &in.Samples[0] {
		t.Fatal("same-rate resampling should not copy the samples")
	}
}

func TestResample_ProducesExpectedLength(t *testing.T) {
	in := Audio{Samples: make([]float32, 44100), Rate: 44100}
	out, err := Resample(in, 16000)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	if out.Rate != 16000 {
		t.Fatalf("rate = %d, want 16000", out.Rate)
	}
	if diff := len(out.Samples) - 16000; diff < -2 || diff > 2 {
		t.Fatalf("length = %d, want ~16000", len(out.Samples))
	}
}

// A constant input must come out at the same level: if the per-phase kernels
// are not normalised, the output ripples at the phase-cycle rate.
func TestResample_PreservesDCLevel(t *testing.T) {
	in := Audio{Samples: make([]float32, 44100), Rate: 44100}
	for i := range in.Samples {
		in.Samples[i] = 0.5
	}
	out, err := Resample(in, 16000)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	// Skip the edges, where the kernel runs off the end of the input.
	for i := 100; i < len(out.Samples)-100; i++ {
		if math.Abs(float64(out.Samples[i]-0.5)) > 1e-4 {
			t.Fatalf("sample %d = %v, want 0.5", i, out.Samples[i])
		}
	}
}

// A tone inside the passband must survive downsampling at full amplitude.
func TestResample_PreservesInBandTone(t *testing.T) {
	const freq = 440.0
	in := Audio{Samples: make([]float32, 44100), Rate: 44100}
	for i := range in.Samples {
		in.Samples[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/44100))
	}
	out, err := Resample(in, 16000)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	peak := 0.0
	for _, s := range out.Samples[500 : len(out.Samples)-500] {
		peak = math.Max(peak, math.Abs(float64(s)))
	}
	if peak < 0.49 || peak > 0.51 {
		t.Fatalf("peak amplitude = %v, want ~0.5", peak)
	}
}

// The anti-alias filter is what stops content above the new Nyquist folding
// back down into the speech band. Without it a 12 kHz tone would reappear at
// 4 kHz, right where it does the most damage to recognition.
func TestResample_RejectsAliasing(t *testing.T) {
	const freq = 12000.0
	in := Audio{Samples: make([]float32, 44100), Rate: 44100}
	for i := range in.Samples {
		in.Samples[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/44100))
	}
	out, err := Resample(in, 16000)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	peak := 0.0
	for _, s := range out.Samples[1000 : len(out.Samples)-1000] {
		peak = math.Max(peak, math.Abs(float64(s)))
	}
	// -60 dB relative to the 0.5 input is an ample margin; the filter measures
	// far better than this, but the bound keeps the test robust.
	if peak > 0.0005 {
		t.Fatalf("alias energy = %v (%.1f dB), want < 0.0005", peak, 20*math.Log10(peak/0.5))
	}
}

func TestResample_Upsampling(t *testing.T) {
	in := Audio{Samples: make([]float32, 8000), Rate: 8000}
	for i := range in.Samples {
		in.Samples[i] = float32(0.5 * math.Sin(2*math.Pi*300*float64(i)/8000))
	}
	out, err := Resample(in, 16000)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	if diff := len(out.Samples) - 16000; diff < -2 || diff > 2 {
		t.Fatalf("length = %d, want ~16000", len(out.Samples))
	}
	peak := 0.0
	for _, s := range out.Samples[500 : len(out.Samples)-500] {
		peak = math.Max(peak, math.Abs(float64(s)))
	}
	if peak < 0.48 || peak > 0.52 {
		t.Fatalf("peak after upsampling = %v, want ~0.5", peak)
	}
}

func TestResample_InvalidRates(t *testing.T) {
	if _, err := Resample(Audio{Samples: []float32{1}, Rate: 16000}, 0); err == nil {
		t.Error("expected an error for a zero target rate")
	}
	if _, err := Resample(Audio{Samples: []float32{1}, Rate: 0}, 16000); err == nil {
		t.Error("expected an error for a zero source rate")
	}
}

func TestAudioDuration(t *testing.T) {
	a := Audio{Samples: make([]float32, 32000), Rate: 16000}
	if got := a.Duration(); got != 2*time.Second {
		t.Fatalf("duration = %v, want 2s", got)
	}
	if got := (Audio{Rate: 0}).Duration(); got != 0 {
		t.Fatalf("duration with no rate = %v, want 0", got)
	}
}

func TestChunkAudio_ShortAudioIsASingleChunk(t *testing.T) {
	a := Audio{Samples: make([]float32, 100), Rate: 16000}
	chunks := ChunkAudio(a, 1000)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Offset != 0 {
		t.Fatalf("offset = %v, want 0", chunks[0].Offset)
	}
}

func TestChunkAudio_EmptyAudioProducesNothing(t *testing.T) {
	if got := ChunkAudio(Audio{Rate: 16000}, 1000); got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}
}

func TestChunkAudio_SplitsAndTracksOffsets(t *testing.T) {
	a := Audio{Samples: make([]float32, 16000*10), Rate: 16000}
	for i := range a.Samples {
		a.Samples[i] = 0.5
	}
	chunks := ChunkAudio(a, 16000*3)
	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks, got %d", len(chunks))
	}

	total := 0
	for i, c := range chunks {
		if len(c.Audio.Samples) > 16000*3 {
			t.Fatalf("chunk %d has %d samples, over the %d limit", i, len(c.Audio.Samples), 16000*3)
		}
		wantOffset := samplesToDuration(total, 16000)
		if c.Offset != wantOffset {
			t.Fatalf("chunk %d offset = %v, want %v", i, c.Offset, wantOffset)
		}
		total += len(c.Audio.Samples)
	}
	if total != len(a.Samples) {
		t.Fatalf("chunks cover %d samples, want %d", total, len(a.Samples))
	}
}

// Splitting mid-word costs accuracy, so the cut should land in the quiet gap
// rather than at the hard limit.
func TestChunkAudio_PrefersQuietBoundary(t *testing.T) {
	rate := 16000
	a := Audio{Samples: make([]float32, rate*10), Rate: rate}
	for i := range a.Samples {
		a.Samples[i] = float32(math.Sin(float64(i) * 0.1))
	}
	// Silence from 4.0s to 4.5s, inside the second half of a 5s window.
	silenceStart, silenceEnd := rate*4, rate*9/2
	for i := silenceStart; i < silenceEnd; i++ {
		a.Samples[i] = 0
	}

	chunks := ChunkAudio(a, rate*5)
	if len(chunks) < 2 {
		t.Fatalf("expected a split, got %d chunk(s)", len(chunks))
	}
	split := len(chunks[0].Audio.Samples)
	if split < silenceStart || split > silenceEnd {
		t.Fatalf("split at sample %d (%.2fs), want inside the silence at %.2f-%.2fs",
			split, float64(split)/float64(rate),
			float64(silenceStart)/float64(rate), float64(silenceEnd)/float64(rate))
	}
}

func TestChunkAudio_NonPositiveLimitDoesNotSplit(t *testing.T) {
	a := Audio{Samples: make([]float32, 100), Rate: 16000}
	if got := ChunkAudio(a, 0); len(got) != 1 {
		t.Fatalf("expected 1 chunk for a zero limit, got %d", len(got))
	}
}

func TestDecode_EmptyInputIsRejected(t *testing.T) {
	if _, err := Decode(nil); err == nil {
		t.Fatal("expected an error for empty input")
	}
}

func TestDecode_WAVIsResampledTo16k(t *testing.T) {
	// One second of 44.1 kHz stereo.
	payload := make([]byte, 0, 44100*4)
	for i := 0; i < 44100; i++ {
		v := int16(8000 * math.Sin(2*math.Pi*440*float64(i)/44100))
		payload = append(payload, int16Bytes(v, v)...)
	}
	audio, err := Decode(buildWAV(wavFormatPCM, 2, 44100, 16, payload))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if audio.Rate != targetSampleRate {
		t.Fatalf("rate = %d, want %d", audio.Rate, targetSampleRate)
	}
	if diff := len(audio.Samples) - 16000; diff < -2 || diff > 2 {
		t.Fatalf("length = %d, want ~16000", len(audio.Samples))
	}
}

// An unrecognised container is handed to ffmpeg; when that is unavailable the
// error has to name both problems, not just "ffmpeg missing".
func TestDecode_UnknownFormatFallsBackToFFmpeg(t *testing.T) {
	original := runFFmpeg
	defer func() { runFFmpeg = original }()

	called := false
	runFFmpeg = func(input []byte) ([]byte, error) {
		called = true
		return buildWAV(wavFormatPCM, 1, 16000, 16, int16Bytes(16384, -16384)), nil
	}

	audio, err := Decode([]byte("some exotic container format here"))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if !called {
		t.Fatal("ffmpeg fallback was not used")
	}
	if len(audio.Samples) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(audio.Samples))
	}
}

func TestDecode_ReportsFFmpegFailure(t *testing.T) {
	original := runFFmpeg
	defer func() { runFFmpeg = original }()
	runFFmpeg = func([]byte) ([]byte, error) { return nil, errors.New("ffmpeg not found in PATH") }

	_, err := Decode([]byte("some exotic container format here"))
	if err == nil {
		t.Fatal("expected an error")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("ffmpeg")) {
		t.Fatalf("error should mention ffmpeg, got: %v", err)
	}
}

func TestDecode_RejectsSilentlyEmptyStream(t *testing.T) {
	original := runFFmpeg
	defer func() { runFFmpeg = original }()
	runFFmpeg = func([]byte) ([]byte, error) {
		return buildWAV(wavFormatPCM, 1, 16000, 16, nil), nil
	}
	if _, err := Decode([]byte("some exotic container format here")); err == nil {
		t.Fatal("expected an error when decoding yields no samples")
	}
}

func TestDecodeMP3_RejectsGarbage(t *testing.T) {
	if _, err := decodeMP3([]byte("not an mp3 stream at all, just bytes")); err == nil {
		t.Fatal("expected an error for a non-MP3 payload")
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("hello", 10); got != "hello" {
		t.Errorf("truncate = %q, want %q", got, "hello")
	}
	if got := truncate("hello world", 5); got != "hello..." {
		t.Errorf("truncate = %q, want %q", got, "hello...")
	}
}

func TestSamplesToDuration(t *testing.T) {
	if got := samplesToDuration(16000, 16000); got != time.Second {
		t.Errorf("samplesToDuration = %v, want 1s", got)
	}
	if got := samplesToDuration(100, 0); got != 0 {
		t.Errorf("samplesToDuration with no rate = %v, want 0", got)
	}
}

func TestBlackmanWindowIsZeroOutsideItsSupport(t *testing.T) {
	if got := blackman(2, 1); got != 0 {
		t.Errorf("blackman(2,1) = %v, want 0", got)
	}
	if got := blackman(-2, 1); got != 0 {
		t.Errorf("blackman(-2,1) = %v, want 0", got)
	}
}

func TestSinc(t *testing.T) {
	if got := sinc(0); got != 1 {
		t.Errorf("sinc(0) = %v, want 1", got)
	}
	if got := sinc(1); math.Abs(got) > 1e-12 {
		t.Errorf("sinc(1) = %v, want 0", got)
	}
}

// Uniformly quiet audio has no single obviously-best cut point. Taking the
// first minimum would halve every chunk and double the number of API calls, so
// the splitter must use its full budget.
func TestChunkAudio_UsesFullBudgetWhenEverywhereIsQuiet(t *testing.T) {
	rate := 16000
	a := Audio{Samples: make([]float32, rate*100), Rate: rate}
	limit := rate * 10

	chunks := ChunkAudio(a, limit)
	if len(chunks) > 11 {
		t.Fatalf("100s split into %d chunks of at most 10s; the budget is being wasted", len(chunks))
	}
	for i, c := range chunks[:len(chunks)-1] {
		if got := len(c.Audio.Samples); got < limit*9/10 {
			t.Fatalf("chunk %d holds %d samples, well under the %d limit", i, got, limit)
		}
	}
}

// Between two equally quiet gaps the later one is the better cut: it keeps
// chunks long without splitting mid-word.
func TestChunkAudio_PrefersTheLaterOfTwoEqualGaps(t *testing.T) {
	rate := 16000
	a := Audio{Samples: make([]float32, rate*10), Rate: rate}
	for i := range a.Samples {
		a.Samples[i] = float32(math.Sin(float64(i) * 0.1))
	}
	// Two silent gaps, both inside the second half of a 6s window.
	for _, gap := range [][2]int{{rate * 4, rate*4 + rate/2}, {rate * 5, rate*5 + rate/2}} {
		for i := gap[0]; i < gap[1]; i++ {
			a.Samples[i] = 0
		}
	}

	chunks := ChunkAudio(a, rate*6)
	split := len(chunks[0].Audio.Samples)
	if split < rate*5 || split > rate*5+rate/2 {
		t.Fatalf("split at %.2fs, want the later gap at 5.0-5.5s", float64(split)/float64(rate))
	}
}

// Regression: MP4/M4A keep their index at the end of the file, so ffmpeg has to
// be handed a seekable input. Feeding it a pipe fails with "partial file" while
// still exiting 0, producing a header-only WAV that used to surface as the
// unhelpful "decoded audio contains no samples".
func TestRunFFmpeg_DecodesSeekOnlyContainers(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg is not installed")
	}

	dir := t.TempDir()
	src := filepath.Join(dir, "src.wav")
	tone := make([]float32, 16000)
	for i := range tone {
		tone[i] = float32(0.3 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}
	if err := os.WriteFile(src, EncodeWAV(Audio{Samples: tone, Rate: 16000}), 0644); err != nil {
		t.Fatal(err)
	}

	m4a := filepath.Join(dir, "out.m4a")
	cmd := exec.Command(ffmpeg, "-y", "-loglevel", "error", "-i", src, "-c:a", "aac", m4a)
	if err := cmd.Run(); err != nil {
		t.Skipf("this ffmpeg cannot produce AAC: %v", err)
	}
	encoded, err := os.ReadFile(m4a)
	if err != nil {
		t.Fatal(err)
	}

	audio, err := Decode(encoded)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if audio.Rate != targetSampleRate {
		t.Errorf("rate = %d, want %d", audio.Rate, targetSampleRate)
	}
	if len(audio.Samples) < 8000 {
		t.Fatalf("decoded only %d samples from a 1s clip", len(audio.Samples))
	}
	peak := 0.0
	for _, s := range audio.Samples[1000 : len(audio.Samples)-1000] {
		peak = math.Max(peak, math.Abs(float64(s)))
	}
	if peak < 0.1 {
		t.Fatalf("decoded audio is silent (peak %v)", peak)
	}
}
